"""UserCredential, get_credential and public_config (core/authz.py)."""

import copy
import json
import pickle

import pytest
from pydantic import SecretStr

from tenxgraph.core.authz import (
    CREDENTIAL_KEY,
    UserCredential,
    get_credential,
    public_config,
)


TOKEN = "tok-secret-1234567890"  # noqa: S105 - test value


def test_repr_and_str_are_redacted():
    cred = UserCredential(TOKEN)
    assert repr(cred) == "UserCredential(scheme='Bearer', token='**********')"
    assert str(cred) == repr(cred)
    assert TOKEN not in f"{cred} {cred!r} {[cred]} { {'c': cred} }"


def test_token_is_secret_and_readable():
    cred = UserCredential(SecretStr(TOKEN), scheme="DPoP")
    assert cred.scheme == "DPoP"
    assert cred.token.get_secret_value() == TOKEN
    assert TOKEN not in str(cred.token)


def test_pickle_raises():
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(UserCredential(TOKEN))
    with pytest.raises(TypeError):
        pickle.dumps({"credential": UserCredential(TOKEN)})


def test_copies_return_the_same_object():
    cred = UserCredential(TOKEN)
    assert copy.copy(cred) is cred
    assert copy.deepcopy(cred) is cred
    assert copy.deepcopy({"credential": cred})["credential"] is cred


def test_json_dumps_never_writes_the_token():
    cred = UserCredential(TOKEN)
    with pytest.raises(TypeError):
        json.dumps({"credential": cred})
    assert TOKEN not in json.dumps({"credential": cred}, default=str)


def test_immutable():
    cred = UserCredential(TOKEN)
    with pytest.raises(AttributeError):
        cred.scheme = "Basic"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cred._token = SecretStr("other")  # type: ignore[misc]
    with pytest.raises(AttributeError):
        del cred._scheme


@pytest.mark.parametrize("token", ["", SecretStr(""), 42])
def test_refuses_empty_or_wrong_token(token):
    with pytest.raises(ValueError):
        UserCredential(token)  # type: ignore[arg-type]


@pytest.mark.parametrize("scheme", ["", "Bear er", "Bearer\r\n"])
def test_refuses_bad_scheme(scheme):
    with pytest.raises(ValueError):
        UserCredential(TOKEN, scheme=scheme)


def test_get_credential_present_and_absent():
    cred = UserCredential(TOKEN)
    assert get_credential({CREDENTIAL_KEY: cred}) is cred
    assert get_credential({}) is None
    assert get_credential({CREDENTIAL_KEY: None}) is None
    assert get_credential(None) is None


@pytest.mark.parametrize("value", [TOKEN, {"token": TOKEN}, SecretStr(TOKEN)])
def test_get_credential_wrong_type_raises_without_the_value(value):
    with pytest.raises(TypeError) as err:
        get_credential({"credential": value})
    assert TOKEN not in str(err.value)


def test_public_config_drops_only_the_credential():
    config = {"thread_id": "t1", "user": {"user_id": "u1"}, "credential": UserCredential(TOKEN)}
    public = public_config(config)
    assert public == {"thread_id": "t1", "user": {"user_id": "u1"}}
    assert "credential" in config  # the run's own config keeps it
    assert public_config(None) == {}
