"""Secrets in request URLs never reach the logs (app/observability/log_redaction.py)."""
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.observability.log_redaction import RedactSecretsFilter, redact  # noqa: E402


def test_demo_token_keeps_only_its_prefix():
    assert redact('"GET /public/demo/bTZ2PRCFzohOG2KExTaa5A/visit HTTP/1.1" 200') == \
        '"GET /public/demo/bTZ2PR…/visit HTTP/1.1" 200'
    assert redact("POST /public/demo/contact") == "POST /public/demo/contact"   # not a token


def test_query_tokens_are_removed():
    assert redact('"WebSocket /ws/voice?token=eyJhbGciOi.x.y" [accepted]') == \
        '"WebSocket /ws/voice?token=<redacted>" [accepted]'
    assert redact("/x?a=1&access_token=abc&b=2") == "/x?a=1&access_token=<redacted>&b=2"


def test_filter_rewrites_uvicorn_style_records():
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:1", "GET", "/public/demo/bTZ2PRCFzohOG2KExTaa5A?r=DIFS0DjRVA", "1.1", 200),
                            None)
    assert RedactSecretsFilter().filter(rec)
    assert rec.getMessage() == '127.0.0.1:1 - "GET /public/demo/bTZ2PR…?r=DIFS0DjRVA HTTP/1.1" 200'


def test_uvicorn_access_formatter_still_works():
    """uvicorn's AccessFormatter unpacks record.args; redaction must keep them a tuple."""
    AccessFormatter = pytest.importorskip("uvicorn.logging").AccessFormatter
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:1", "GET", "/public/demo/bTZ2PRCFzohOG2KExTaa5A", "1.1", 200), None)
    RedactSecretsFilter().filter(rec)
    out = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False).format(rec)
    assert out == '127.0.0.1:1 - "GET /public/demo/bTZ2PR… HTTP/1.1" 200 OK'
