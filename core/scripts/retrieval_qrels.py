"""Frozen retrieval relevance labels ("qrels") for the 50 golden questions (R5.01).

Used by eval_retrieval.py to score RETRIEVAL ONLY — no agent turn and no LLM judge. The only
paid call is one embedding per question, and those are cached on disk, so a re-run costs $0.

What a label is
---------------
For each question: the pieces of EVIDENCE the knowledge base must put in front of the model
for the golden answer to be possible. Each piece is a GROUP of alternative anchors; a chunk
supplies that piece when its text contains ANY anchor of the group.

    "EN006": [["11,000 rupees per square foot"],      # evidence 1: the 2 BHK rate
              ["11,499 rupees per square foot"]],     # evidence 2: the 3 BHK rate

  * Anchors are matched against CHUNK CONTENT (case-insensitive; dashes and whitespace
    normalised — see norm()). Content-based on purpose: chunk ids change on every re-ingest,
    the facts do not. If a document is edited so an anchor no longer exists anywhere in the KB,
    eval_retrieval.py reports it as STALE instead of silently scoring the question as a miss.
  * Groups mirror the golden rubric's core / any_of facts (rag_accuracy_golden.py) — the
    things the answer is graded on. Where the rubric's fact is a derived statement ("the 2BHK
    starts at approx Rs.1.04 crore"), the anchor is the KB figure it comes from ("1,03,93,750").
  * None = an escalation question whose correct answer is "I don't have that" — there is no
    KB evidence to retrieve, so it is excluded from retrieval metrics.
  * "re:<pattern>" is a regular expression (on normalised text) for the rare anchor a plain
    substring cannot express.

Hand-written 2026-09-28 against the seeded DSR Highland Greenz KB (44 chunks, 16 documents).
Tied to GOLDEN_VERSION: if the golden set changes, re-derive these and bump QRELS_VERSION.
"""
import re

QRELS_VERSION = "qrels-v1-2026-09-28"
FOR_GOLDEN_VERSION = "hg-golden-v5-2026-09-25"

_RATE_2BHK = ["11,000 rupees per square foot"]
_RATE_3BHK = ["11,499 rupees per square foot"]
_CHEAPEST = ["1,03,93,750"]
_3BHK_ENTRY = ["1,54,32,011"]
_CORNER = ["corner flats, 50 rupees"]
_RERA = ["prm/ka/rera/1251/446/pr/110522/004876"]
# The KB does not hold a payment plan; the SOP chunk saying the agent never quotes one is the
# evidence that lets it decline correctly instead of improvising.
_PAYMENT_SOP = ["payment schedule"]

QRELS: dict[str, list[list[str]] | None] = {
    # ── English
    "EN004": [_CHEAPEST],
    "EN006": [_RATE_2BHK, _RATE_3BHK],
    "EN011": [["1,560"], ["1,92,53,440"]],
    "EN012": [["30 rupees per square foot for each floor"]],
    "EN016": [["corpus fund - 20 rupees"]],
    "EN018": [["95,000"]],
    "EN027": [["900 flats"]],
    "EN032": [["cards room", "elders seating plaza", "seniors seating plaza"]],
    "EN038": [["video door phone", "cctv at entry"]],
    "EN042": [["electronic city 9 km"]],
    "EN045": [["manipal hospital 5.5 km", "sakra hospital"]],
    "EN048": None,
    "EN050": None,
    "EN053": None,
    "EN055": [_PAYMENT_SOP],
    "EN137": [["three clubhouses"], ["squash court", "indoor gym", "banquet hall", "library"]],
    "EN139": [["dsr techno cube", "jubilee hills"]],
    # ── Hinglish
    "HG001": [_3BHK_ENTRY],
    "HG002": [_RATE_2BHK, _RATE_3BHK],
    "HG005": [_CHEAPEST],
    "HG017": [["external electrification"]],
    # All five configurations: 1 BHK (619 sft), 1.5 BHK (780 sft), 2 / 2.5 / 3 BHK.
    "HG021": [["619"], ["780"], ["re:(?<![.\\d])2 bhk"], ["2.5 bhk"], ["3 bhk"]],
    "HG024": [["875 to 1,21"]],
    "HG028": [["swimming pool"]],
    "HG030": [["three clubhouses"], ["23,156"]],
    "HG033": [["dog park"]],
    "HG034": [["80cm by 80cm vitrified"]],
    "HG035": [["water purifier provision"]],
    "HG043": [["primus international school"]],
    "HG047": [_RERA],
    "HG057": [["8792582433"]],
    "HG058": [_PAYMENT_SOP],
    "HG138": [["wipro corporate office 2.5 km", "rga tech park 2.2 km", "decathlon"]],
    # ── Hindi (romanised)
    "HI003": [_3BHK_ENTRY],
    "HI008": [["1,72,06,110"]],
    "HI010": [["250 rupees per square foot"]],
    "HI013": [_CORNER],
    "HI014": [["clubhouse charges are 2,50,000", "clubhouse charges 3,25,000",
               "clubhouse charges of 3,25,000"]],
    "HI019": [["included in the published totals", "clubhouse charges, car parking"]],
    "HI020": [_CORNER, ["premium flats, 75 rupees"]],
    "HI025": [["tower d only", "except tower d"]],
    "HI029": [["tennis court", "basketball court", "cricket practice nets", "mini soccer"]],
    "HI036": [["windows upvc sliding"]],
    "HI040": [["chikkanayakanahalli, off sarjapur road"]],
    "HI041": [["wipro corporate office 2.5 km"]],
    "HI044": None,
    "HI046": [["prm/ka/rera/1251"]],
    "HI049": None,
    "HI052": None,
    "HI059": [["as per actuals"]],
}

_DASHES = re.compile(r"[‐-―−]")
_SPACES = re.compile(r"\s+")


def norm(text: str) -> str:
    """Lower-case, every dash variant -> '-', whitespace runs -> one space."""
    return _SPACES.sub(" ", _DASHES.sub("-", (text or "").lower())).strip()


def anchor_matches(anchor: str, normalised_text: str) -> bool:
    if anchor.startswith("re:"):
        return re.search(anchor[3:], normalised_text) is not None
    return norm(anchor) in normalised_text


def groups_in(text: str, groups: list[list[str]]) -> set[int]:
    """Indexes of the evidence groups this chunk text supplies."""
    t = norm(text)
    return {i for i, group in enumerate(groups) if any(anchor_matches(a, t) for a in group)}
