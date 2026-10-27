"""Tests for user_id_mode (observability identity transform) and Kafka TLS/SASL options."""

import asyncio
import hashlib
import hmac
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.runtime.publisher.events import Event, EventType
from tenxgraph.runtime.publisher.exporters import setup_observability
from tenxgraph.runtime.publisher.identity import (
    USER_ID_KEY_ENV,
    hash_user_id,
    mask_user_id,
    transform_user_id,
    validate_user_id_mode,
)
from tenxgraph.runtime.publisher.kafka_publisher import KafkaPublisher
from tenxgraph.runtime.publisher.otel_publisher import OtelPublisher

from .test_otel_publisher import _evt, _install_fake_otel


def _user_attr(mode, user_id="alice-1234"):
    tracer = _install_fake_otel()
    pub = OtelPublisher(user_id_mode=mode)
    ev = _evt(Event.GRAPH_EXECUTION, EventType.START)
    ev.user_id = user_id
    asyncio.run(pub.publish(ev))
    return tracer.spans[0].attrs["tenxgraph.graph.user_id"]


class TestHelpers:
    def test_raw_unchanged(self):
        assert transform_user_id("alice", "raw") == "alice"
        assert transform_user_id(42) == "42"

    def test_hash_plain_sha256_16_hex(self, monkeypatch):
        monkeypatch.delenv(USER_ID_KEY_ENV, raising=False)
        out = transform_user_id("alice", "hash")
        assert out == hashlib.sha256(b"alice").hexdigest()[:16]
        assert len(out) == 16

    def test_hash_hmac_with_env_key(self, monkeypatch):
        monkeypatch.setenv(USER_ID_KEY_ENV, "secret")
        out = transform_user_id("alice", "hash")
        assert out == hmac.new(b"secret", b"alice", hashlib.sha256).hexdigest()[:16]
        assert out != hashlib.sha256(b"alice").hexdigest()[:16]

    def test_hash_deterministic_and_key_changes_output(self):
        assert hash_user_id("a", "k1") == hash_user_id("a", "k1")
        assert hash_user_id("a", "k1") != hash_user_id("a", "k2")

    @pytest.mark.parametrize(
        ("uid", "expected"),
        [
            ("", "***"),
            ("abcd", "***"),
            ("abcde", "a***e"),
            ("abcdefg", "a***g"),
            ("alice42", "a***2"),
            ("alice-42", "al***42"),
            ("x" * 40, "xx***xx"),
        ],
    )
    def test_mask(self, uid, expected):
        assert mask_user_id(uid) == expected

    def test_validate(self):
        assert validate_user_id_mode("hash") == "hash"
        with pytest.raises(ValueError, match="raw, hash, mask"):
            validate_user_id_mode("nope")


class TestOtelPublisher:
    def test_default_raw(self):
        assert _user_attr("raw") == "alice-1234"
        _install_fake_otel()
        assert OtelPublisher()._user_id_mode == "raw"

    def test_hash(self, monkeypatch):
        monkeypatch.delenv(USER_ID_KEY_ENV, raising=False)
        assert _user_attr("hash") == hashlib.sha256(b"alice-1234").hexdigest()[:16]

    def test_mask(self):
        assert _user_attr("mask") == "al***34"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="raw, hash, mask"):
            OtelPublisher(user_id_mode="bad")  # type: ignore[arg-type]


class TestSetupObservability:
    def test_passes_through_logfire(self):
        with patch("tenxgraph.runtime.publisher.exporters.setup_logfire") as lf:
            setup_observability(MagicMock(), {"user_id_mode": "mask", "logfire": {"enabled": True}})
        assert lf.call_args.kwargs["user_id_mode"] == "mask"

    def test_passes_through_langsmith(self):
        with patch("tenxgraph.runtime.publisher.exporters.setup_langsmith") as ls:
            setup_observability(
                MagicMock(), {"user_id_mode": "hash", "langsmith": {"enabled": True}}
            )
        assert ls.call_args.kwargs["user_id_mode"] == "hash"

    def test_passes_through_both(self):
        with (
            patch("tenxgraph.runtime.publisher.exporters.setup_logfire") as lf,
            patch("tenxgraph.runtime.publisher.exporters._guard_logfire"),
            patch("tenxgraph.runtime.publisher.exporters._build_langsmith_processor"),
        ):
            setup_observability(
                MagicMock(),
                {
                    "user_id_mode": "hash",
                    "logfire": {"enabled": True},
                    "langsmith": {"enabled": True},
                },
            )
        assert lf.call_args.kwargs["user_id_mode"] == "hash"

    def test_default_raw(self):
        with patch("tenxgraph.runtime.publisher.exporters.setup_logfire") as lf:
            setup_observability(MagicMock(), {"logfire": {"enabled": True}})
        assert lf.call_args.kwargs["user_id_mode"] == "raw"

    def test_invalid_raises_even_when_disabled(self):
        with pytest.raises(ValueError, match="raw, hash, mask"):
            setup_observability(MagicMock(), {"user_id_mode": "bad"})


class TestKafkaSecurity:
    def _start(self, config):
        producer_cls = MagicMock()
        producer_cls.return_value.start = AsyncMock()
        module = MagicMock(AIOKafkaProducer=producer_cls)
        pub = KafkaPublisher(config)
        with patch(
            "tenxgraph.runtime.publisher.kafka_publisher.importlib.import_module",
            return_value=module,
        ):
            asyncio.run(pub._get_producer())
        return producer_cls.call_args.kwargs

    def test_default_has_no_security_kwargs(self):
        kwargs = self._start({})
        for key in (
            "security_protocol",
            "ssl_context",
            "sasl_mechanism",
            "sasl_plain_username",
            "sasl_plain_password",
        ):
            assert key not in kwargs

    def test_sasl_ssl_passthrough(self):
        ctx = ssl.create_default_context()
        kwargs = self._start(
            {
                "security_protocol": "SASL_SSL",
                "ssl_context": ctx,
                "sasl_mechanism": "PLAIN",
                "sasl_plain_username": "u",
                "sasl_plain_password": "p",
            }
        )
        assert kwargs["security_protocol"] == "SASL_SSL"
        assert kwargs["ssl_context"] is ctx
        assert kwargs["sasl_mechanism"] == "PLAIN"
        assert kwargs["sasl_plain_username"] == "u"
        assert kwargs["sasl_plain_password"] == "p"  # noqa: S105

    def test_only_set_keys_pass(self):
        kwargs = self._start({"security_protocol": "SSL"})
        assert kwargs["security_protocol"] == "SSL"
        assert "sasl_mechanism" not in kwargs

    def test_cafile_builds_context(self):
        with patch("tenxgraph.runtime.publisher.kafka_publisher.ssl.create_default_context") as m:
            ctx = m.return_value
            kwargs = self._start(
                {"ssl_cafile": "/ca.pem", "ssl_certfile": "/c.pem", "ssl_keyfile": "/k.pem"}
            )
        m.assert_called_once_with(cafile="/ca.pem")
        ctx.load_cert_chain.assert_called_once_with("/c.pem", keyfile="/k.pem")
        assert kwargs["ssl_context"] is ctx

    def test_ssl_context_wins_over_files(self):
        ctx = ssl.create_default_context()
        kwargs = self._start({"ssl_context": ctx, "ssl_cafile": "/ca.pem"})
        assert kwargs["ssl_context"] is ctx

    def test_password_not_logged(self, caplog):
        caplog.set_level("DEBUG")
        self._start({"sasl_plain_password": "hunter2"})
        assert "hunter2" not in caplog.text
