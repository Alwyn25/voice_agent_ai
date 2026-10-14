"""Likely-mishearing detection (app/engine/voice/asr_hints.py) and the call lexicon it
matches against (app/engine/voice/lexicon.py). Pure -- no DB, no pipecat, no model.

Positives are real Deepgram transcripts from recorded calls; negatives are ordinary
sentences that must never be flagged. Other-industry cases prove nothing here is
real-estate specific: the matcher only knows what the lexicon hands it.

    cd core && python -m pytest tests/test_asr_hints.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice import asr_hints as ah  # noqa: E402
from app.engine.voice import lexicon as lx  # noqa: E402
from app.packs.base import Vocabulary  # noqa: E402
from app.packs.healthcare.pack import HealthcarePack  # noqa: E402
from app.packs.real_estate_india.pack import RealEstateIndiaPack  # noqa: E402
from app.packs.saas_sales.pack import SaasSalesPack  # noqa: E402

_RE_LEXICON = lx.CallLexicon(
    ["Ajmera Realty & Infra", "Ajmera Manhattan", "Wadala East", "Bhakti Park"],
    lx.domain_terms(RealEstateIndiaPack().vocabulary)).terms()


def _pairs(text, lexicon=_RE_LEXICON):
    return [(h.heard, h.term) for h in ah.find_hints(text, lexicon)]


@pytest.mark.parametrize("transcript,heard,term", [
    # Every one of these is a real transcript from a recorded voice call.
    ("is it rare approved", "rare", "RERA"),
    ("is it herrera approved", "herrera", "RERA"),
    ("when is the position", "position", "possession"),
    ("3 bsk का price क्या है", "bsk", "BHK"),
    ("i want to mokai visit tomorrow morning", "mokai visit", "book a visit"),
    ("bouquet visite tomorrow", "bouquet visite", "book a visit"),
    # Same shape, other domain words.
    ("what is the carpet erea", "carpet erea", "carpet area"),
    ("i want to see the flour plan", "flour plan", "floor plan"),
])
def test_recorded_mishearings_are_flagged(transcript, heard, term):
    assert (heard, term) in _pairs(transcript)


@pytest.mark.parametrize("transcript", [
    "what is the price", "is it approved", "i want to book a visit tomorrow", "okay",
    "yes yes", "hello", "when is the possession", "what's the rate", "what is the area",
    "is there a bank loan", "which banks are approved", "i am looking for a flat", "4 pm",
    "my name is rahul", "can you call me back tomorrow", "what are the amenities",
    "where is the site office", "thank you so much", "what is the total cost",
    "is it ready to move", "what time can i come",
    # Real transcript: "person" must not read as "possession" (prsn vs psn), nor "session".
    "when is the time and who is the person coding", "how long is the session",
])
def test_ordinary_sentences_are_not_flagged(transcript):
    assert _pairs(transcript) == []


def test_a_term_heard_correctly_is_never_hinted():
    # "book a visit" is right; the overlapping "to book a visit" must not read as a garbled
    # "book a site visit".
    assert _pairs("i want to book a visit tomorrow") == []
    assert _pairs("is it rera approved") == []


def test_other_scripts_are_left_alone():
    assert _pairs("क्या यह रेरा approved है") == []


def test_hints_are_capped_and_best_first():
    hints = ah.find_hints("is it rare approved when is the position 3 bsk flour plan",
                          _RE_LEXICON)
    assert len(hints) <= ah.MAX_HINTS
    assert [h.score for h in hints] == sorted((h.score for h in hints), reverse=True)


def test_empty_inputs():
    assert ah.find_hints("", _RE_LEXICON) == []
    assert ah.find_hints("is it rare approved", []) == []
    assert ah.find_hints("is it rare approved", None) == []


def test_phonetic_key_merges_sounds_not_spelling():
    assert ah.phonetic_key("position") == ah.phonetic_key("possession")
    assert ah.phonetic_key("rare") == ah.phonetic_key("RERA")
    assert ah.phonetic_key("") == ""


# --- not real-estate specific: the lexicon decides what is hearable --------------------

def test_healthcare_pack_phrases_are_caught():
    lexicon = lx.CallLexicon([], lx.domain_terms(HealthcarePack().vocabulary)).terms()
    assert "book a consultation" in lexicon
    assert ("book a consultashun", "book a consultation") in _pairs(
        "i want to book a consultashun", lexicon)


def test_saas_pack_phrases_are_caught():
    lexicon = lx.CallLexicon([], lx.domain_terms(SaasSalesPack().vocabulary)).terms()
    assert "book a demo call" in lexicon and "book a call" in lexicon
    assert ("onboarting", "onboarding") in _pairs("how long does onboarting take", lexicon)


def test_tenant_custom_intent_is_hearable():
    class _E:
        def __init__(self, name, aliases, source):
            self.name, self.aliases, self.source = name, aliases, source
    entries = [_E("loan_enquiry", ["home loan", "emi"], "tenant"),
               _E("dnc_request", ["dnc", "opt_out"], "default")]
    terms = lx.intent_terms(entries)
    assert terms == ["loan enquiry", "home loan", "emi"]   # default labels are not speech
    assert ("lone enquiry", "loan enquiry") in _pairs("i have a lone enquiry", terms)


# --- lexicon construction ----------------------------------------------------------------

def test_action_phrases_include_head_noun_form():
    assert lx.action_phrases("site visit") == [
        "book a site visit", "schedule a site visit", "reschedule my site visit",
        "cancel my site visit", "book a visit", "schedule a visit", "reschedule my visit",
        "cancel my visit"]
    assert lx.action_phrases("consultation")[0] == "book a consultation"


def test_placeholder_vocabulary_yields_no_terms():
    # The generic pack's Vocabulary() defaults ("commitment", "product") are not speech.
    assert lx.domain_terms(Vocabulary()) == []
    assert lx.domain_terms(None) == []


def test_real_estate_domain_terms_lead_with_curated_stt_terms():
    terms = lx.domain_terms(RealEstateIndiaPack().vocabulary)
    assert terms[:3] == ["RERA", "BHK", "possession"]
    assert "book a visit" in terms and "site visit" in terms


def test_lexicon_keyterms_respect_deepgram_limit():
    lexicon = lx.CallLexicon([f"Project{i:03d}word" for i in range(80)],
                             [f"term{i}" for i in range(80)])
    assert len(lexicon.keyterms()) <= 100
    assert len(lexicon.terms()) == 160          # matching is uncapped
