#!/usr/bin/env python3
"""Test the voice agent end to end from typed text -- no ElevenLabs, no browser, no mic.

    docker compose exec core python scripts/voice_harness.py --tenant "Ajmera Realty & Infra"
    docker compose exec core python scripts/voice_harness.py --tenant <uuid> --file my_calls.txt
    docker compose exec core python scripts/voice_harness.py --tenant <uuid> \
        "is it rera approved" "when is the possession" "book a visit tomorrow morning"

Each line you type goes through the same path a live voice call does, minus the speakers:

    typed text --Piper (local, free)--> audio --Deepgram--> transcript --> process_turn()
               --> printed: what Deepgram heard, flagged mishearings, tools, the agent's reply

  * Speech synthesis is Piper (already installed for pipecat, runs on this machine, costs
    nothing). ElevenLabs is never called, so this works with the ElevenLabs quota exhausted.
  * Speech recognition is REAL Deepgram, configured exactly like a call (model, language,
    numerals/smart_format/punctuate, and this tenant's call-lexicon keyterms) -- so
    mishearings show up here the way they do on a call. It uses Deepgram's prerecorded API
    rather than the live stream, so endpointing/turn-splitting is not exercised.
  * The turn is the same process_turn() TurnBridge calls, on a channel="voice" conversation
    with the per-call lexicon in working memory -- confidence banding, ASR hints, tool
    chaining and the responder all behave as on a call. Barge-in (interrupting a turn
    mid-plan) needs a live audio stream and is not exercised here.
  * A demo tenant's call starts the way a demo-link call does (sample lead + demo greeting);
    any other tenant starts like the Test Console's voice call.

Writes what a real test call writes -- a conversation, messages, decision traces and any
bookings the agent makes -- and closes the conversation at the end.

Options: --text-only skips Piper+Deepgram (the typed text IS the transcript) to separate
STT problems from engine problems; --voice picks another Piper voice (downloaded once, free,
e.g. hi_IN-pratham-medium for Hindi); --debug prints the turn's full debug envelope.
Interactive commands: /quit (or Ctrl-D) ends the call, /debug toggles the debug envelope.
"""
import argparse
import asyncio
import difflib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Conversation, DemoProspect, Message, Tenant  # noqa: E402
from app.engine.memory import working  # noqa: E402
from app.engine.planner.turn_runner import process_turn  # noqa: E402
from app.engine.voice.lexicon import build_call_lexicon  # noqa: E402

VOICE_DIR = Path(__file__).resolve().parent.parent / ".voice_models"
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


# --------------------------------------------------------------------------- speech

class Speaker:
    """Piper text-to-speech, loaded once. Downloads the voice on first use (free, local)."""

    def __init__(self, voice_name: str):
        from piper import PiperVoice
        from piper.download_voices import download_voice
        VOICE_DIR.mkdir(parents=True, exist_ok=True)
        model = VOICE_DIR / f"{voice_name}.onnx"
        if not model.exists() or not Path(f"{model}.json").exists():
            print(f"(downloading Piper voice {voice_name} -- one time)")
            download_voice(voice_name, VOICE_DIR)
        try:
            self._voice = PiperVoice.load(str(model))
        except Exception as e:  # noqa: BLE001 — a truncated earlier download fails to parse
            print(f"(Piper voice {voice_name} is unreadable ({e.__class__.__name__}) -- "
                  f"re-downloading)")
            download_voice(voice_name, VOICE_DIR, force_redownload=True)
            self._voice = PiperVoice.load(str(model))

    def wav(self, text: str) -> tuple[bytes, float]:
        """WAV bytes for ``text`` and the audio's duration in seconds."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            self._voice.synthesize_wav(text, w)
        data = buf.getvalue()
        with wave.open(io.BytesIO(data)) as r:
            seconds = r.getnframes() / float(r.getframerate() or 1)
        return data, seconds


def transcribe(wav: bytes, keyterms: list[str]) -> tuple[str, float | None, int]:
    """Deepgram, configured like a live call (voice_ws._build_stt). Returns
    (transcript, confidence, latency_ms)."""
    s = get_settings()
    params = [("model", s.deepgram_model), ("language", s.deepgram_language),
              ("smart_format", str(s.deepgram_smart_format).lower()),
              ("punctuate", str(s.deepgram_punctuate).lower()),
              ("numerals", str(s.deepgram_numerals).lower())]
    params += [("keyterm", t) for t in keyterms]
    req = urllib.request.Request(f"{DEEPGRAM_URL}?{urllib.parse.urlencode(params)}", data=wav,
                                 headers={"Authorization": f"Token {s.deepgram_api_key}",
                                          "Content-Type": "audio/wav"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            body = json.loads(res.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Deepgram HTTP {e.code}: {e.read()[:300]!r}") from None
    ms = int((time.perf_counter() - started) * 1000)
    alt = body["results"]["channels"][0]["alternatives"][0]
    return alt.get("transcript", ""), alt.get("confidence"), ms


# --------------------------------------------------------------------------- call setup

async def resolve_tenant(db, ref: str) -> Tenant:
    try:
        cond = Tenant.id == uuid.UUID(ref)
    except ValueError:
        cond = Tenant.name.ilike(ref)
    tenant = (await db.execute(select(Tenant).where(cond))).scalars().first()
    if tenant is None:
        names = (await db.execute(select(Tenant.name).order_by(Tenant.name))).scalars().all()
        sys.exit(f"No tenant matches {ref!r}. Tenants: {', '.join(names)}")
    return tenant


async def start_call(db, tenant: Tenant) -> tuple[Conversation, str]:
    """Open a voice conversation the way the real entry points do, and seed the call
    lexicon into working memory (voice_ws.py)."""
    prospect = (await db.execute(select(DemoProspect).where(
        DemoProspect.tenant_id == tenant.id))).scalars().first()
    if prospect is not None:
        from app.demo.agent import start_demo_conversation
        conv, _customer, greeting = await start_demo_conversation(db, prospect, channel="voice")
    else:
        from app.api.sessions import _default_greeting, with_recording_disclosure
        conv = Conversation(tenant_id=tenant.id, channel="voice", direction="inbound",
                            status="active", started_at=datetime.now(timezone.utc))
        db.add(conv)
        await db.flush()
        greeting = with_recording_disclosure(_default_greeting(tenant.name, None), tenant)
        db.add(Message(tenant_id=tenant.id, conversation_id=conv.id, role="agent",
                       content=greeting))
    conv.channel, conv.started_at = "voice", datetime.now(timezone.utc)
    await db.commit()
    return conv, greeting


# --------------------------------------------------------------------------- reporting

def _words(text: str) -> list[str]:
    return re.findall(r"[\w']+", (text or "").lower())


def word_accuracy(typed: str, heard: str) -> float:
    a, b = _words(typed), _words(heard)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def show_turn(n: int, typed: str, heard: str, stt: dict | None, result, wall_ms: int,
              debug: bool) -> None:
    d = result.debug or {}
    print(f"\n[{n}] you typed : {typed}")
    if stt is not None:
        acc = word_accuracy(typed, heard)
        mark = "" if acc >= 0.999 else f"   <-- STT differs (word match {acc:.0%})"
        conf = f"{stt['confidence']:.2f}" if stt["confidence"] is not None else "n/a"
        print(f"    deepgram  : {heard!r}  (confidence {conf}, {stt['seconds']:.1f}s audio, "
              f"{stt['ms']} ms){mark}")
    for h in d.get("asr_hints") or []:
        print(f"    asr hint  : {h['heard']!r} -> probably {h['term']!r} ({h['score']})")
    tools = ", ".join(f"{t['tool_name']}({t['status']})" for t in result.executed_tools) or "-"
    conf = result.confidence or {}
    adj = ",".join(a.get("reason", "") for a in conf.get("adjustments") or [])
    print(f"    intent    : {d.get('understood_intent') or '-'}   action: {result.action}   "
          f"confidence: {conf.get('band')} {conf.get('score')}" + (f" [{adj}]" if adj else ""))
    print(f"    tools     : {tools}")
    if result.pending_confirmation:
        print(f"    awaiting  : yes/no to {result.pending_confirmation.get('tool')}")
    timing = d.get("timing") or {}
    print(f"    agent     : {result.response_text}")
    print(f"    timing    : turn {timing.get('total_ms', wall_ms)} ms "
          f"(planner {timing.get('planner_ms', 0)}, tools {timing.get('tool_ms', 0)}, "
          f"responder {timing.get('responder_ms', 0)})")
    if debug:
        print(json.dumps(d, indent=2, default=str)[:6000])


# --------------------------------------------------------------------------- main loop

def _inputs(args):
    """Queries from the command line, a file, or (default) interactively."""
    if args.queries:
        yield from args.queries
        return
    if args.file:
        for line in Path(args.file).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.strip()
        return
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            return
        if line:
            yield line


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tenant", required=True, help="tenant name (case-insensitive) or id")
    ap.add_argument("--file", help="one query per line; '#' lines are comments")
    ap.add_argument("--voice", default=get_settings().piper_voice,
                    help="Piper voice (default: PIPER_VOICE setting)")
    ap.add_argument("--text-only", action="store_true",
                    help="skip Piper+Deepgram: the typed text is used as the transcript")
    ap.add_argument("--debug", action="store_true", help="print each turn's debug envelope")
    ap.add_argument("queries", nargs="*", help="queries to run in order, then hang up")
    args = ap.parse_args()

    s = get_settings()
    if not args.text_only and not s.deepgram_api_key:
        sys.exit("DEEPGRAM_API_KEY is not set -- use --text-only to skip speech recognition.")
    speaker = None if args.text_only else await asyncio.to_thread(Speaker, args.voice)

    async with get_session_factory()() as db:
        tenant = await resolve_tenant(db, args.tenant)
        conv, greeting = await start_call(db, tenant)
        lexicon = await build_call_lexicon(db, tenant.id)
        await working.save(conv.id, {"asr_lexicon": lexicon.terms()})
        keyterms = lexicon.keyterms()
        print(f"Tenant {tenant.name} | conversation {conv.id}")
        print(f"STT: {'off (--text-only)' if args.text_only else f'Deepgram {s.deepgram_model}/{s.deepgram_language}, {len(keyterms)} keyterms'}"
              f" | TTS: {'off' if args.text_only else f'Piper {args.voice}'} (ElevenLabs not used)")
        print(f"\nagent> {greeting}")

        embed_cache: dict = {}
        turns, stt_misses, debug = 0, 0, args.debug
        try:
            for typed in _inputs(args):
                if typed in ("/quit", "/exit"):
                    break
                if typed == "/debug":
                    debug = not debug
                    print(f"(debug envelope {'on' if debug else 'off'})")
                    continue
                stt = None
                heard = typed
                if speaker is not None:
                    try:
                        wav, seconds = await asyncio.to_thread(speaker.wav, typed)
                        heard, confidence, ms = await asyncio.to_thread(transcribe, wav, keyterms)
                    except Exception as e:  # noqa: BLE001 — report and keep the call going
                        print(f"\n[!] speech step failed: {e}")
                        continue
                    stt = {"confidence": confidence, "seconds": seconds, "ms": ms}
                    if not heard.strip():
                        print(f"\n[!] Deepgram heard nothing for {typed!r} -- a live call would "
                              f"not start a turn either.")
                        continue
                    stt_misses += word_accuracy(typed, heard) < 0.999
                session = await working.load(conv.id)
                as_of = session.get("as_of_date")
                started = time.perf_counter()
                result = await process_turn(
                    db, tenant_id=tenant.id, conversation=conv, user_input=heard,
                    actor="voice_harness",
                    as_of_date=datetime.fromisoformat(as_of).date() if as_of else None,
                    stt_confidence=stt["confidence"] if stt else None,
                    stt_seconds=stt["seconds"] if stt else None,
                    session_embed_cache=embed_cache)
                turns += 1
                show_turn(turns, typed, heard, stt, result,
                          int((time.perf_counter() - started) * 1000), debug)
                if result.end_call:
                    print("\n(the agent ended the call)")
                    break
        finally:
            await db.refresh(conv)
            conv.status, conv.ended_at = "completed", datetime.now(timezone.utc)
            await db.commit()
            await working.clear(conv.id)
            print(f"\n--- call ended: {turns} turn(s)"
                  + (f", {stt_misses} with STT differences" if speaker is not None else "")
                  + f" | conversation {conv.id}")


if __name__ == "__main__":
    asyncio.run(main())
