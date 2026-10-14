"""Unit tests for PlivoTelephonyProvider.

Uses unittest.mock / respx to avoid real HTTP calls and keep these fast/offline.
Run with: pytest core/tests/telephony/test_plivo.py -v
"""
from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# anyio is already installed (bundled with pipecat-ai); use its pytest marker.
pytest_plugins = ("anyio",)

from app.services.telephony.base import (
    STATUS_ANSWERED,
    STATUS_BUSY,
    STATUS_COMPLETED,
    STATUS_NO_ANSWER,
    STATUS_RINGING,
    DialRequest,
)
from app.services.telephony.plivo import PlivoTelephonyProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(
    auth_id: str = "TESTID",
    auth_token: str = "TESTTOKEN",
    webhook_url: str = "https://example.com/webhooks/telephony/plivo",
) -> PlivoTelephonyProvider:
    settings = MagicMock()
    settings.plivo_auth_id = auth_id
    settings.plivo_auth_token = auth_token
    settings.telephony_caller_id = "+910000000000"
    settings.telephony_webhook_url = webhook_url
    settings.telephony_media_stream_base_url = ""
    return PlivoTelephonyProvider(settings)


def _make_v2_signature(auth_token: str, webhook_url: str, nonce: str, body: bytes) -> str:
    body_sha256 = hashlib.sha256(body).hexdigest()
    message = webhook_url + nonce + body_sha256
    return hmac.new(auth_token.encode(), message.encode(), hashlib.sha256).hexdigest()


def _make_dial_request(to: str = "+919999999999") -> DialRequest:
    return DialRequest(
        call_id="call-uuid-1",
        tenant_id="tenant-1",
        to_number=to,
        from_number="+910000000000",
        purpose="marketing",
    )


# ---------------------------------------------------------------------------
# originate
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_originate_success():
    provider = _make_provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 201
    mock_resp.json.return_value = {"request_uuid": "plivo-req-uuid-123", "message": "Call fired"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.telephony.plivo.httpx.AsyncClient", return_value=mock_client):
        result = await provider.originate(_make_dial_request())

    assert result.ok is True
    assert result.provider == "plivo"
    assert result.provider_call_id == "plivo-req-uuid-123"
    assert result.error is None


@pytest.mark.anyio
async def test_originate_failure_4xx():
    provider = _make_provider()
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"error": "invalid number"}'

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    with patch("app.services.telephony.plivo.httpx.AsyncClient", return_value=mock_client):
        result = await provider.originate(_make_dial_request())

    assert result.ok is False
    assert result.provider == "plivo"
    assert result.provider_call_id is None
    assert "invalid number" in (result.error or "")


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------

def test_verify_signature_valid():
    auth_token = "MYSECRETTOKEN"
    webhook_url = "https://example.com/webhooks/telephony/plivo"
    provider = _make_provider(auth_token=auth_token, webhook_url=webhook_url)

    body = b"CallUUID=abc123&CallStatus=answered"
    nonce = "test-nonce-12345"
    sig = _make_v2_signature(auth_token, webhook_url, nonce, body)

    headers = {
        "x-plivo-signature-v2": sig,
        "x-plivo-signature-v2-nonce": nonce,
    }
    assert provider.verify_signature(headers=headers, body=body) is True


def test_verify_signature_invalid_sig():
    provider = _make_provider()
    body = b"CallUUID=abc123&CallStatus=answered"
    headers = {
        "x-plivo-signature-v2": "badsignature",
        "x-plivo-signature-v2-nonce": "some-nonce",
    }
    assert provider.verify_signature(headers=headers, body=body) is False


def test_verify_signature_missing_header():
    provider = _make_provider()
    body = b"CallUUID=abc123&CallStatus=answered"
    assert provider.verify_signature(headers={}, body=body) is False


# ---------------------------------------------------------------------------
# parse_webhook
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("call_status,expected_status", [
    ("ringing",     STATUS_RINGING),
    ("in-progress", STATUS_ANSWERED),
    ("answered",    STATUS_ANSWERED),
    ("busy",        STATUS_BUSY),
    ("no-answer",   STATUS_NO_ANSWER),
    ("canceled",    STATUS_NO_ANSWER),
    ("completed",   STATUS_COMPLETED),
])
def test_parse_webhook_status_mapping(call_status: str, expected_status: str):
    provider = _make_provider()
    payload = {"CallUUID": "uuid-abc", "CallStatus": call_status}
    event = provider.parse_webhook(payload=payload)
    assert event is not None
    assert event.status == expected_status
    assert event.provider == "plivo"
    assert event.provider_call_id == "uuid-abc"


def test_parse_webhook_unknown_status_returns_none():
    provider = _make_provider()
    payload = {"CallUUID": "uuid-xyz", "CallStatus": "totally-unknown-event"}
    assert provider.parse_webhook(payload=payload) is None


def test_parse_webhook_missing_uuid_returns_none():
    provider = _make_provider()
    assert provider.parse_webhook(payload={"CallStatus": "answered"}) is None


def test_parse_webhook_missing_status_returns_none():
    provider = _make_provider()
    assert provider.parse_webhook(payload={"CallUUID": "uuid-abc"}) is None


def test_parse_webhook_with_duration():
    provider = _make_provider()
    payload = {"CallUUID": "uuid-abc", "CallStatus": "completed", "Duration": "45"}
    event = provider.parse_webhook(payload=payload)
    assert event is not None
    assert event.duration_seconds == 45
