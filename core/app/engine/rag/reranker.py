"""R5.01 — cross-encoder re-ranking of the hybrid retrieval pool.

Why this exists
---------------
Both retrieval arms judge a chunk WITHOUT ever reading it next to the question: the embedding
squeezes question and chunk into separate vectors (so "2.5 BHK" and "3 BHK" land a hair apart),
and the keyword arm counts shared words (so a chunk that merely MENTIONS "price" can outrank the
cost sheet that states it). RRF then fuses two rank lists that share those blind spots. The
FOCUS / PRICING / POLICY boosts in retrieval.py are hand-written patches over exactly this.

A cross-encoder reads (question, chunk) as ONE input with full attention across both and
answers "does this passage answer this question?" — the judgement the patches approximate.
It is too slow to run over a whole corpus, which is why it only re-scores the top of the pool
RRF has already narrowed.

How it runs (and why it cannot hurt a live call)
------------------------------------------------
  * ONNX Runtime on CPU, no PyTorch. onnxruntime + tokenizers already ship in the image (via
    pipecat / faster-whisper), so this adds no dependency weight. The default model file is the
    int8-quantised export: ~2.5x faster than fp32 at near-identical scores (measured in the
    core container: 12 pairs x 256 tokens, 2 threads -> ~370 ms int8 vs ~890 ms fp32).
  * Inference runs on a DEDICATED, bounded thread pool — never on the event loop, which also
    carries the voice audio. ORT's thread spin-waiting is switched off so an idle reranker does
    not burn the CPU the audio pipeline needs.
  * Every call has a hard deadline (retrieval_rerank_timeout_ms) and a concurrency cap
    (retrieval_rerank_max_concurrency). Past either, the caller gets a non-"ok" outcome and
    retrieval keeps its RRF order. A reranker problem can cost ranking quality, never a turn.
  * The model is loaded in the background (startup warm-up, or lazily on first use). A request
    never waits for a load: until the model is ready it is served RRF order ("not_ready").
    A failed load is retried after a back-off, so a transient download failure heals itself.
  * Scores are cached per (model, query, passage) — the knowledge_lookup tool re-running the
    query the speculative pass already scored this turn pays nothing.

Debugging
---------
  * Every call returns a RerankOutcome whose .status says exactly what happened
    (ok / not_ready / unavailable / skipped_busy / skipped_few / timeout / error).
  * reranker_status() is served at GET /knowledge/reranker: load state, model file + sha256,
    per-status counters and latency percentiles since process start.
  * retrieval.py logs one `rag.rerank` line per retrieval, and a per-candidate table when
    RETRIEVAL_RERANK_DEBUG_LOG=true.
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import os
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Outcome statuses. Anything other than OK means "scores unusable — keep the RRF order".
OK = "ok"
NOT_READY = "not_ready"        # model still loading (or load not started yet)
UNAVAILABLE = "unavailable"    # last load failed; will retry after LOAD_RETRY_S
SKIPPED_BUSY = "skipped_busy"  # concurrency cap reached — CPU protection
SKIPPED_FEW = "skipped_few"    # fewer than 2 candidates: nothing to reorder
SKIPPED_LANGUAGE = "skipped_language"  # query language not in retrieval_rerank_languages
TIMEOUT = "timeout"
ERROR = "error"

LOAD_RETRY_S = 60.0            # back-off before retrying a failed model load
_SCORE_CACHE_MAX = 4096        # (model, query, passage) -> logit entries kept
# Characters kept per passage BEFORE tokenisation. The tokenizer truncates to max_tokens anyway;
# this only stops a pathological multi-megabyte chunk from costing tokenisation time first.
_MAX_PASSAGE_CHARS = 4000


# --------------------------------------------------------------------------- data types

@dataclass(frozen=True)
class RerankConfig:
    """The settings that identify one loaded model + how it is run. Built from app settings
    (see config_from_settings) so the eval script can swap models per variant."""
    model: str                 # HF repo id, or a local directory holding the files
    revision: str | None       # pinned HF commit — the model can never change under us
    onnx_file: str             # path of the ONNX export inside the repo / directory
    cache_dir: str | None      # HF cache root; None = HF default (HF_HOME)
    allow_download: bool       # False = only use files already on disk (baked image)
    max_tokens: int            # (query + passage) token budget per pair
    threads: int               # ORT intra-op threads per inference
    max_concurrency: int       # simultaneous inferences per process
    timeout_ms: int

    @property
    def key(self) -> tuple:
        """What makes two configs need different loaded sessions."""
        return (self.model, self.revision, self.onnx_file, self.cache_dir,
                self.max_tokens, self.threads)

    @property
    def label(self) -> str:
        return f"{self.model}:{self.onnx_file}"


def config_from_settings(s) -> RerankConfig:
    return RerankConfig(
        model=s.retrieval_rerank_model,
        revision=(s.retrieval_rerank_revision or None),
        onnx_file=s.retrieval_rerank_onnx_file,
        cache_dir=(s.retrieval_rerank_cache_dir or os.environ.get("RETRIEVAL_RERANK_CACHE_DIR")
                   or None),
        allow_download=s.retrieval_rerank_allow_download,
        max_tokens=s.retrieval_rerank_max_tokens,
        threads=s.retrieval_rerank_threads,
        max_concurrency=s.retrieval_rerank_max_concurrency,
        timeout_ms=s.retrieval_rerank_timeout_ms,
    )


@dataclass
class RerankOutcome:
    status: str
    model: str
    scores: dict[str, float] = field(default_factory=dict)   # candidate key -> raw logit
    candidates: int = 0
    scored: int = 0            # pairs actually sent to the model (candidates - cache hits)
    cache_hits: int = 0
    ms: float = 0.0            # wall time the caller spent waiting, incl. queueing
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass
class _Handle:
    """One loaded model: ORT session + tokenizer + what we need to feed it."""
    session: object
    tokenizer: object
    input_names: tuple[str, ...]
    onnx_path: str
    sha256: str
    load_ms: float


# --------------------------------------------------------------------------- module state
#
# Everything below is process-wide. Loads and inference run on worker threads, so shared maps
# are guarded by _lock; the event loop only ever reads/writes them through the helpers here.

_lock = threading.Lock()
_handles: dict[tuple, _Handle] = {}
_loading: dict[tuple, Future] = {}             # load in progress, keyed by RerankConfig.key
_load_errors: dict[tuple, tuple[float, str]] = {}   # key -> (monotonic time, message)
_executor: ThreadPoolExecutor | None = None
_executor_size = 0
_inflight = 0
_score_cache: "OrderedDict[tuple[str, str, str], float]" = OrderedDict()
_stats: Counter = Counter()
_latencies: deque = deque(maxlen=512)          # ms of recent OK calls, for p50/p95


def _pool(size: int) -> ThreadPoolExecutor:
    """The inference pool. Sized by max_concurrency; recreated only if that setting changes."""
    global _executor, _executor_size
    with _lock:
        if _executor is None or _executor_size != size:
            old = _executor
            _executor = ThreadPoolExecutor(max_workers=size, thread_name_prefix="rerank")
            _executor_size = size
            if old is not None:
                old.shutdown(wait=False)
        return _executor


# --------------------------------------------------------------------------- loading

def _resolve_file(cfg: RerankConfig, filename: str) -> str:
    """Local path of one model file. A local directory is used as-is; otherwise the pinned HF
    revision is resolved through the HF cache (the Docker build pre-populates it, so a baked
    image never touches the network — allow_download=False enforces that)."""
    if os.path.isdir(cfg.model):
        path = os.path.join(cfg.model, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{filename} not found in {cfg.model}")
        return path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=cfg.model, filename=filename, revision=cfg.revision,
                           cache_dir=cfg.cache_dir, local_files_only=not cfg.allow_download)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_sync(cfg: RerankConfig) -> _Handle:
    """Load + warm one model. Runs on a worker thread; may take seconds (or a download)."""
    import onnxruntime as ort
    from tokenizers import Tokenizer

    started = time.perf_counter()
    onnx_path = _resolve_file(cfg, cfg.onnx_file)
    tok_path = _resolve_file(cfg, "tokenizer.json")

    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_truncation(max_length=cfg.max_tokens)
    # BERT-family vocabularies pad with [PAD] (id 0); XLM-R-family (the multilingual mMiniLM
    # rerankers) pad with <pad> (id 1). Padded positions are masked out either way, but feeding
    # the model its own pad id keeps it exactly on the distribution it was exported with.
    pad_token = next((t for t in ("[PAD]", "<pad>") if tokenizer.token_to_id(t) is not None),
                     None)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token) if pad_token else 0,
                             pad_token=pad_token or "[PAD]")

    so = ort.SessionOptions()
    so.intra_op_num_threads = cfg.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT worker threads spin-wait after each run by default, holding cores at 100% for a
    # while. That CPU belongs to the voice pipeline (VAD, audio framing) on the same box.
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    input_names = tuple(i.name for i in session.get_inputs())

    handle = _Handle(session=session, tokenizer=tokenizer, input_names=input_names,
                     onnx_path=onnx_path, sha256=_sha256(onnx_path), load_ms=0.0)
    # One throwaway inference: ORT allocates its arenas lazily, and that first-run cost
    # belongs here, not on a caller's turn.
    _infer(handle, "warm up", ["warm up passage"])
    handle.load_ms = (time.perf_counter() - started) * 1000
    return handle


def _on_load_done(key: tuple, cfg: RerankConfig, fut: Future) -> None:
    """Load finished (worker thread). Publish the handle, or record the failure for back-off."""
    with _lock:
        _loading.pop(key, None)
        try:
            handle = fut.result()
        except Exception as e:  # noqa: BLE001 — any load failure degrades to RRF order
            _load_errors[key] = (time.monotonic(), f"{e.__class__.__name__}: {e}")
            _stats["load_failed"] += 1
            log.error("rag.rerank model load FAILED model=%s — retrieval stays on RRF order, "
                      "retry in %.0fs: %s: %s", cfg.label, LOAD_RETRY_S, e.__class__.__name__, e)
            return
        _handles[key] = handle
        _load_errors.pop(key, None)
        _stats["loaded"] += 1
    log.info("rag.rerank model loaded model=%s file=%s sha256=%s inputs=%s threads=%d "
             "max_tokens=%d load_ms=%.0f", cfg.label, handle.onnx_path, handle.sha256[:12],
             ",".join(handle.input_names), cfg.threads, cfg.max_tokens, handle.load_ms)


def _ensure_loading(cfg: RerankConfig) -> tuple[_Handle | None, str | None]:
    """Non-blocking. Returns (handle, None) when ready; otherwise starts a background load if
    none is running (respecting the failure back-off) and returns (None, status)."""
    key = cfg.key
    with _lock:
        handle = _handles.get(key)
        if handle is not None:
            return handle, None
        if key in _loading:
            return None, NOT_READY
        failed = _load_errors.get(key)
        if failed and time.monotonic() - failed[0] < LOAD_RETRY_S:
            return None, UNAVAILABLE
        # Loads use their own short-lived thread, NOT the inference pool, so a slow download
        # can never occupy a scoring slot.
        fut: Future = Future()
        _loading[key] = fut

    def _run() -> None:
        try:
            fut.set_result(_load_sync(cfg))
        except BaseException as e:  # noqa: BLE001
            fut.set_exception(e)

    fut.add_done_callback(lambda f: _on_load_done(key, cfg, f))
    threading.Thread(target=_run, name="rerank-load", daemon=True).start()
    log.info("rag.rerank model load started model=%s revision=%s download=%s", cfg.label,
             (cfg.revision or "latest")[:12], cfg.allow_download)
    return None, NOT_READY


async def warm_up(cfg: RerankConfig, timeout_s: float = 300.0) -> bool:
    """Load the model now and wait for it (startup / scripts). True when ready. Never raises."""
    handle, _ = _ensure_loading(cfg)
    if handle is not None:
        return True
    with _lock:
        fut = _loading.get(cfg.key)
        if cfg.key in _handles:         # finished between the two lock sections
            return True
    if fut is None:                     # failed recently and is in back-off
        return False
    try:
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout_s)
    except Exception:  # noqa: BLE001 — failure already logged by _on_load_done
        return False
    with _lock:
        return cfg.key in _handles


# --------------------------------------------------------------------------- inference

def _infer(handle: _Handle, query: str, passages: list[str]):
    """Raw relevance logits, one per passage. Sync — call on a worker thread only."""
    import numpy as np
    enc = handle.tokenizer.encode_batch([(query, p) for p in passages])
    feed = {
        "input_ids": np.asarray([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64),
        "token_type_ids": np.asarray([e.type_ids for e in enc], dtype=np.int64),
    }
    feed = {k: v for k, v in feed.items() if k in handle.input_names}
    logits = np.asarray(handle.session.run(None, feed)[0], dtype=np.float32)
    if logits.ndim == 2 and logits.shape[1] > 1:
        # Two-class heads ([not relevant, relevant]) — the relevant-class logit is the score.
        logits = logits[:, 1]
    return logits.reshape(-1)


def _passage_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _score_job(handle: _Handle, model_label: str, query: str,
               items: list[tuple[str, str]]) -> dict[str, float]:
    """Worker-thread body: score the uncached pairs and write them to the shared cache.
    Writing here (not in the awaiting coroutine) means a call that TIMED OUT still leaves its
    scores behind for the next identical query."""
    logits = _infer(handle, query, [text for _, text in items])
    out: dict[str, float] = {}
    with _lock:
        for (key, text), score in zip(items, logits):
            out[key] = float(score)
            ck = (model_label, query, _passage_hash(text))
            _score_cache[ck] = float(score)
            _score_cache.move_to_end(ck)
        while len(_score_cache) > _SCORE_CACHE_MAX:
            _score_cache.popitem(last=False)
    return out


def _record(status: str, ms: float | None = None) -> None:
    with _lock:
        _stats[status] += 1
        if status == OK and ms is not None:
            _latencies.append(ms)


async def score(cfg: RerankConfig, query: str,
                candidates: list[tuple[str, str]]) -> RerankOutcome:
    """Cross-encoder logits for ``candidates`` = [(key, passage_text), ...].

    Never raises and never blocks the event loop. Anything but status "ok" means the caller
    must keep its existing order. Logits are unbounded (higher = more relevant); only their
    ORDER is used downstream, so no calibration is assumed.
    """
    started = time.perf_counter()
    out = RerankOutcome(status=OK, model=cfg.label, candidates=len(candidates))

    def _finish(status: str, error: str | None = None) -> RerankOutcome:
        out.status, out.error = status, error
        out.ms = (time.perf_counter() - started) * 1000
        _record(status, out.ms)
        return out

    if len(candidates) < 2:
        return _finish(SKIPPED_FEW)
    handle, status = _ensure_loading(cfg)
    if handle is None:
        return _finish(status or NOT_READY)

    items = [(key, (text or "")[:_MAX_PASSAGE_CHARS]) for key, text in candidates]

    # Cache first: only unseen (query, passage) pairs go to the model.
    todo: list[tuple[str, str]] = []
    with _lock:
        for key, text in items:
            hit = _score_cache.get((cfg.label, query, _passage_hash(text)))
            if hit is None:
                todo.append((key, text))
            else:
                out.scores[key] = hit
    out.cache_hits = len(items) - len(todo)
    out.scored = len(todo)
    if not todo:
        return _finish(OK)

    # Concurrency cap. Checked-and-claimed under the lock; released when the WORKER finishes
    # (not when we stop waiting), so a timed-out job still counts until its CPU is free again.
    global _inflight
    with _lock:
        if _inflight >= cfg.max_concurrency:
            busy = True
        else:
            busy = False
            _inflight += 1
    if busy:
        return _finish(SKIPPED_BUSY)

    def _release(_f) -> None:
        global _inflight
        with _lock:
            _inflight -= 1

    try:
        cf = _pool(cfg.max_concurrency).submit(_score_job, handle, cfg.label, query, todo)
    except Exception as e:  # noqa: BLE001 — e.g. pool shut down during reload
        _release(None)
        return _finish(ERROR, f"{e.__class__.__name__}: {e}")
    cf.add_done_callback(_release)

    try:
        fresh = await asyncio.wait_for(asyncio.wrap_future(cf), cfg.timeout_ms / 1000)
    except asyncio.TimeoutError:
        return _finish(TIMEOUT, f"exceeded {cfg.timeout_ms} ms")
    except Exception as e:  # noqa: BLE001 — a model error must not fail retrieval
        log.warning("rag.rerank inference failed: %s: %s", e.__class__.__name__, e)
        return _finish(ERROR, f"{e.__class__.__name__}: {e}")
    out.scores.update(fresh)
    return _finish(OK)


# --------------------------------------------------------------------------- fusion (pure)

def fuse_rankings(base_order: list[str], ce_scores: dict[str, float], *, mode: str,
                  k: int, base_weight: float, ce_weight: float) -> dict[str, float]:
    """New relevance score per candidate, from its pre-rerank position and its CE score.

    ``base_order``  candidate keys, best first, as retrieval ranked them (RRF + the policy /
                    pricing membership boosts). Every key must have a CE score.
    ``mode``        "fuse"    — reciprocal-rank fusion of the two orderings:
                                    base_weight/(k + base_rank) + ce_weight/(k + ce_rank)
                                The existing ranking keeps a vote, so one odd CE judgement
                                (e.g. an English-trained model on a Hindi phrasing) cannot
                                single-handedly evict what vector + keyword + business rules
                                agreed on. The default.
                    "replace" — cross-encoder order only: ce_weight/(k + ce_rank).
    Ranks, not raw logits, are fused: CE logits are unbounded and uncalibrated across queries,
    and RRF scores are not on their scale either — ranks are the one thing both share.
    Ties break on CE score, then base position, so the result is fully deterministic.
    """
    base_pos = {key: i for i, key in enumerate(base_order)}
    ce_order = sorted(base_order, key=lambda key: (-ce_scores[key], base_pos[key]))
    ce_rank = {key: i for i, key in enumerate(ce_order, start=1)}
    fused: dict[str, float] = {}
    for base_rank, key in enumerate(base_order, start=1):
        ce_part = ce_weight / (k + ce_rank[key])
        fused[key] = ce_part if mode == "replace" else base_weight / (k + base_rank) + ce_part
    # Deterministic tie-break without changing any strict ordering: nudge by CE rank at a
    # scale far below the smallest real gap between two fused scores.
    return {key: score - ce_rank[key] * 1e-12 for key, score in fused.items()}


# --------------------------------------------------------------------------- query gloss
#
# The default cross-encoder was trained on English MS MARCO. Given raw romanised Hindi
# ("corner aur premium mein charge ka farak kya hai") it sees mostly unknown tokens and scores
# on the few English nouns, which in the 2026-09-28 eval pushed evidence OUT of the window.
# A word-for-word English gloss ("corner and premium in charge of difference what is") is
# ungrammatical but gives the model the meaning of every word, and turned rerank from a net
# loss into a net gain on Hinglish. It is applied to the CE input ONLY.
#
# The built-in list is deliberately GENERAL Hindi — grammar words, question words, common verbs
# and everyday nouns — never domain vocabulary. Domain words ("khel kood" -> "sports",
# "suvidha" -> "amenities") belong to a tenant's glossary, where they can be curated per KB.

# Unambiguous Hindi words: none of them is an English word, so a query containing any of these
# is Hindi/Hinglish for certain and the ambiguous words below may be glossed too.
_GLOSS_STRONG = {
    # question words
    "kya": "what", "kia": "what", "kaun": "who", "kon": "who", "kaunsa": "which",
    "konsa": "which", "kaunse": "which", "kab": "when", "kahan": "where", "kahaan": "where",
    "kaise": "how", "kaisa": "how", "kaisi": "how", "kyun": "why", "kyu": "why", "kyon": "why",
    "kitna": "how much", "kitnaa": "how much", "kitni": "how much", "kitne": "how many",
    # copula / auxiliaries
    "hai": "is", "hei": "is", "hain": "are", "hoga": "will be", "hogi": "will be",
    "honge": "will be", "hota": "is", "hoti": "is", "hote": "are", "tha": "was", "thi": "was",
    # postpositions / conjunctions that do not collide with English
    "mein": "in", "mei": "in", "aur": "and", "bhi": "also", "liye": "for", "lie": "for",
    "wala": "", "wali": "", "wale": "", "waala": "", "waali": "", "waale": "",
    "lekin": "but", "agar": "if", "tak": "until",
    # negation
    "nahi": "not", "nahin": "not", "nhi": "not",
    # pronouns / determiners
    "mujhe": "me", "muje": "me", "mera": "my", "meri": "my", "mere": "my", "aap": "you",
    "aapka": "your", "aapki": "your", "aapke": "your", "apna": "own", "apni": "own",
    "hum": "we", "hame": "us", "humein": "us", "hamara": "our", "yeh": "this", "woh": "that",
    "iska": "its", "iski": "its", "iske": "its", "uska": "its", "uski": "its", "uske": "its",
    "koi": "any", "kuch": "some", "saare": "all", "sare": "all", "sabhi": "all",
    "sabse": "most",
    # common verbs
    "milega": "available", "milegi": "available", "milenge": "available",
    "milta": "available", "milti": "available", "chahiye": "need", "chaiye": "need",
    "batao": "tell", "btao": "tell", "bataiye": "tell", "bataye": "tell", "dikhao": "show",
    "lagega": "charged", "lagegi": "charged", "lagta": "charged", "lagti": "charged",
    "padega": "will cost", "padegi": "will cost", "padta": "costs", "banta": "comes to",
    "sakta": "can", "sakti": "can", "sakte": "can", "karna": "do", "karo": "do",
    "karein": "do", "rahe": "", "raha": "", "rha": "", "rahi": "", "gaya": "", "gayi": "",
    # everyday nouns / adjectives
    "kharcha": "cost", "kharch": "cost", "keemat": "price", "kimat": "price", "daam": "price",
    "paisa": "money", "paise": "money", "ghar": "home", "jaankari": "information",
    "jankari": "information", "farak": "difference", "fark": "difference",
    "antar": "difference", "saal": "year", "mahina": "month", "accha": "good",
    "acha": "good", "bada": "big", "badi": "big", "bade": "big", "chota": "small",
    "choti": "small", "chhota": "small", "sasta": "cheap", "sasti": "cheap",
    "mehenga": "expensive", "mehnga": "expensive", "zyada": "more", "jyada": "more",
    "pehle": "before", "baad": "after", "abhi": "now", "aaj": "today", "paas": "near",
    "milakar": "combined", "shamil": "included", "baare": "about", "thoda": "a little",
    "thora": "a little",
}
# Words that ARE English too ("me", "par", "to"...) — glossed only in a query that the strong
# set already proved is Hindi, so "tell me the price" is never rewritten.
_GLOSS_AMBIGUOUS = {
    "me": "in", "se": "from", "ka": "of", "ki": "of", "ke": "of", "ko": "to", "par": "on",
    "pe": "on", "ye": "this", "vo": "that", "wo": "that", "ya": "or", "sab": "all",
    "ek": "one", "kal": "tomorrow", "ho": "", "kar": "", "na": "",
}
_GLOSS_TOKEN = re.compile(r"[a-z]+|[^a-z]+")


@functools.lru_cache(maxsize=64)
def _compile_phrases(items: tuple[tuple[str, str], ...]) -> tuple[tuple["re.Pattern", str], ...]:
    """Longest phrase first, whole-word. Cached: a tenant's glossary rarely changes."""
    ordered = sorted(((k.strip().lower(), v) for k, v in items if k and k.strip()),
                     key=lambda kv: len(kv[0]), reverse=True)
    return tuple((re.compile(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])"), str(v))
                 for k, v in ordered)


def gloss_query(query: str, extra: dict[str, str] | None = None) -> str:
    """English gloss of a romanised Hindi/Hinglish query, for the cross-encoder input only.

    1. ``extra`` (settings + tenant glossary) phrases are replaced first, always — domain
       entries are curated, so they are trusted even in an otherwise-English query.
    2. The built-in word list applies ONLY when the query contains at least one unambiguous
       Hindi word; then the ambiguous ones (me, se, ka, par...) are glossed too.
    An English query without glossary hits is returned unchanged (lower-cased); numbers,
    punctuation and unknown words pass through as they are.
    """
    text = (query or "").lower()
    for pattern, repl in _compile_phrases(tuple(sorted((extra or {}).items()))):
        text = pattern.sub(repl, text)
    tokens = _GLOSS_TOKEN.findall(text)
    if not any(t in _GLOSS_STRONG for t in tokens):
        return re.sub(r"\s+", " ", text).strip()
    out = []
    for t in tokens:
        if t in _GLOSS_STRONG:
            out.append(_GLOSS_STRONG[t])
        elif t in _GLOSS_AMBIGUOUS:
            out.append(_GLOSS_AMBIGUOUS[t])
        else:
            out.append(t)
    return re.sub(r"\s+", " ", "".join(out)).strip()


# --------------------------------------------------------------------------- introspection

def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * len(ordered)))], 1)


def reranker_status(cfg: RerankConfig | None = None) -> dict:
    """Snapshot for GET /knowledge/reranker and scripts: what is loaded, what failed, and how
    calls have been going since the process started."""
    with _lock:
        loaded = [{"model": k[0], "revision": (k[1] or "")[:12], "file": k[2],
                   "path": h.onnx_path, "sha256": h.sha256[:12], "inputs": list(h.input_names),
                   "max_tokens": k[4], "threads": k[5], "load_ms": round(h.load_ms)}
                  for k, h in _handles.items()]
        errors = [{"model": k[0], "file": k[2], "error": msg,
                   "age_s": round(time.monotonic() - t, 1)} for k, (t, msg) in _load_errors.items()]
        lat = list(_latencies)
        snapshot = {"loaded": loaded, "loading": [k[0] + ":" + k[2] for k in _loading],
                    "load_errors": errors, "inflight": _inflight,
                    "cache_entries": len(_score_cache), "calls": dict(_stats),
                    "latency_ms": {"n": len(lat), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                                   "max": round(max(lat), 1) if lat else None}}
    if cfg is not None:
        with _lock:
            ready = cfg.key in _handles
        snapshot["configured"] = {"model": cfg.label, "revision": (cfg.revision or "")[:12],
                                  "ready": ready, "threads": cfg.threads,
                                  "max_concurrency": cfg.max_concurrency,
                                  "timeout_ms": cfg.timeout_ms, "max_tokens": cfg.max_tokens,
                                  "allow_download": cfg.allow_download,
                                  "cache_dir": cfg.cache_dir}
    return snapshot


def clear_score_cache() -> None:
    """Forget cached scores. The eval calls this so every measured rerank is a cold one —
    what a live turn's first retrieval actually pays."""
    with _lock:
        _score_cache.clear()


def _reset_for_tests() -> None:
    """Drop every loaded model, cache entry and counter. Tests only."""
    global _inflight
    with _lock:
        _handles.clear()
        _loading.clear()
        _load_errors.clear()
        _score_cache.clear()
        _stats.clear()
        _latencies.clear()
        _inflight = 0
