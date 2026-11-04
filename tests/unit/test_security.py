import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from api import auth
from api.schemas import CaseIn, TransferIn
from models.registry import artifact_path


class _Keys:
    def api_keys(self):
        return {"k-alice": "alice", "k-bob": "bob"}


@pytest.fixture()
def keys(monkeypatch):
    monkeypatch.setattr(auth, "get_settings", lambda: _Keys())


def test_header_key_maps_to_its_operator(keys):
    assert auth.operator("k-bob") == "bob"
    for bad in (None, "", "k-carol", "k-alicé"):  # non-ASCII must be a 401, not a 500 from compare_digest
        with pytest.raises(HTTPException) as e:
            auth.operator(bad)
        assert e.value.status_code == 401


def test_websocket_auth_message(keys):
    assert auth.message_operator('{"type": "auth", "api_key": "k-alice"}') == "alice"
    for bad in (
        '{"type": "auth", "api_key": "nope"}',
        '{"type": "ping", "api_key": "k-alice"}',
        '{"type": "auth", "api_key": 5}',
        '["k-alice"]',
        "k-alice",
        "ping",
        "",
    ):
        assert auth.message_operator(bad) is None


def test_model_versions_cannot_leave_the_artifacts_dir():
    assert artifact_path("v12").name == "v12"
    for bad in ("../v1", "v1/../../etc", "/tmp/v1", "C:\\models\\v1", "v", "v1 ", "", None):
        with pytest.raises(ValueError):
            artifact_path(bad)


def _transfer(**kw):
    base = dict(transfer_id="t1", ts=1, payer_id="a", payer_fsp="fsp-01", payee_id="b", payee_fsp="fsp-02", amount=10)
    return TransferIn(**{**base, **kw})


def test_request_sizes_are_bounded():
    _transfer(device_id="d" * 128)
    with pytest.raises(ValidationError):
        _transfer(device_id="d" * 10_000)
    with pytest.raises(ValidationError):
        CaseIn(title="x", transfer_ids=["t"] * 5000)
    with pytest.raises(ValidationError):
        CaseIn(title="x", transfer_ids=["t" * 1000])
