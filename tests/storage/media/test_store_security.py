"""Cloud key validation (B12), owner-only files (B18) and update_metadata (Q15)."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tenxgraph.storage.media import (
    BaseMediaStore,
    CloudMediaStore,
    InMemoryMediaStore,
    LocalFileMediaStore,
)


posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")

BAD_KEYS = [
    "../../../other-tenant/secrets/report",
    "a/../a",
    "short",
    "A" * 32,
    "g" * 32,
    "a" * 31,
    "a" * 33,
    "a" * 32 + "\n",
    "",
    "a" * 30 + "/.",
]
GOOD_KEY = "0123456789abcdef" * 2


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# B12: cloud keys
# ---------------------------------------------------------------------------


class TestCloudKeyValidation:
    @pytest.fixture()
    def storage(self):
        storage = AsyncMock()
        storage.upload = AsyncMock(side_effect=lambda fp, cp: cp)
        storage.delete = AsyncMock(return_value=True)
        storage.get_public_url = AsyncMock(return_value="https://signed.example.com/x")
        return storage

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", BAD_KEYS)
    async def test_every_entry_point_rejects_bad_keys(self, storage, key) -> None:
        store = CloudMediaStore(storage, prefix="p")
        calls = [
            store.retrieve(key),
            store.delete(key),
            store.exists(key),
            store.get_metadata(key),
            store.get_public_url(key),
            store.get_direct_url(key),
            store.get_direct_url(key, mime_type="image/png"),
            store.update_metadata(key, {"a": 1}),
        ]
        for call in calls:
            with pytest.raises(ValueError, match="Invalid storage key"):
                await call
        storage.get_public_url.assert_not_called()
        storage.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_direct_url_with_mime_never_signs_a_bad_path(self, storage) -> None:
        store = CloudMediaStore(storage)  # default prefix, so the legacy path is active
        with pytest.raises(ValueError):
            await store.get_direct_url("../../x", mime_type="image/png")
        storage.get_public_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_valid_key_still_works_on_the_new_prefix(self, storage) -> None:
        store = CloudMediaStore(storage)
        meta = json.dumps({"mime_type": "image/png", "ext": ".png"}).encode()
        with patch.object(CloudMediaStore, "_download_from_url", AsyncMock(return_value=meta)):
            url = await store.get_direct_url(GOOD_KEY, mime_type="image/png")
        assert url == "https://signed.example.com/x"
        signed = storage.get_public_url.call_args_list[-1].args[0]
        assert signed == f"10xgraph-media/01/23/{GOOD_KEY}.png"

    @pytest.mark.asyncio
    async def test_legacy_prefix_still_found_for_valid_keys(self, storage) -> None:
        store = CloudMediaStore(storage)
        meta = json.dumps({"mime_type": "image/png", "ext": ".png"}).encode()

        async def download(url):
            return meta

        async def public_url(path, expiration=3600):
            return f"https://signed.example.com/{path}"

        storage.get_public_url = AsyncMock(side_effect=public_url)

        async def fake_download(url):
            if "agentflow-media" in url:
                return meta
            raise FileNotFoundError(url)

        with patch.object(CloudMediaStore, "_download_from_url", staticmethod(fake_download)):
            assert await store.exists(GOOD_KEY) is True
            direct = await store.get_direct_url(GOOD_KEY, mime_type="image/png")
        assert direct == f"https://signed.example.com/agentflow-media/01/23/{GOOD_KEY}.png"


# ---------------------------------------------------------------------------
# B18: file modes
# ---------------------------------------------------------------------------


@posix_only
class TestFileModes:
    @pytest.mark.asyncio
    async def test_local_store_creates_owner_only_files(self, tmp_path) -> None:
        old = os.umask(0o022)
        try:
            base = tmp_path / "media"
            store = LocalFileMediaStore(str(base))
            key = await store.store(b"data", "image/png", {"owner_id": "u1"})
        finally:
            os.umask(old)

        assert _mode(base) == 0o700
        assert _mode(base / key[:2]) == 0o700
        assert _mode(base / key[:2] / key[2:4]) == 0o700
        assert _mode(base / key[:2] / key[2:4] / f"{key}.png") == 0o600
        assert _mode(base / key[:2] / key[2:4] / f"{key}.meta.json") == 0o600

    @pytest.mark.asyncio
    async def test_nested_new_base_dirs_are_all_private(self, tmp_path) -> None:
        old = os.umask(0o022)
        try:
            await LocalFileMediaStore(str(tmp_path / "a" / "b" / "media")).store(b"x", "image/png")
        finally:
            os.umask(old)
        for rel in ("a", "a/b", "a/b/media"):
            assert _mode(tmp_path / rel) == 0o700

    @pytest.mark.asyncio
    async def test_existing_base_dir_keeps_its_mode(self, tmp_path) -> None:
        base = tmp_path / "media"
        base.mkdir()
        base.chmod(0o755)
        LocalFileMediaStore(str(base))
        assert _mode(base) == 0o755

    @pytest.mark.asyncio
    async def test_update_metadata_keeps_0600(self, tmp_path) -> None:
        old = os.umask(0o022)
        try:
            store = LocalFileMediaStore(str(tmp_path))
            key = await store.store(b"data", "image/png")
            await store.update_metadata(key, {"threads": ["t1"]})
        finally:
            os.umask(old)
        assert _mode(tmp_path / key[:2] / key[2:4] / f"{key}.meta.json") == 0o600
        assert [
            p.name for p in (tmp_path / key[:2] / key[2:4]).iterdir() if p.name.endswith(".tmp")
        ] == []

    @pytest.mark.asyncio
    async def test_sqlite_checkpointer_file_and_directory_modes(self, tmp_path) -> None:
        pytest.importorskip("aiosqlite")
        from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer

        db = tmp_path / "state" / "cp.db"
        old = os.umask(0o022)
        try:
            cp = SqliteCheckpointer(db)
            await cp.asetup()
            await cp.arelease()
        finally:
            os.umask(old)
        assert _mode(db.parent) == 0o700
        assert _mode(db) == 0o600
        for suffix in ("-wal", "-shm"):
            side = Path(str(db) + suffix)
            if side.exists():
                assert _mode(side) == 0o600

    @pytest.mark.asyncio
    async def test_sqlite_existing_file_keeps_its_mode(self, tmp_path) -> None:
        pytest.importorskip("aiosqlite")
        from tenxgraph.storage.checkpointer.sqlite_checkpointer import SqliteCheckpointer

        db = tmp_path / "cp.db"
        db.touch()
        db.chmod(0o644)
        cp = SqliteCheckpointer(db)
        await cp.asetup()
        await cp.arelease()
        assert _mode(db) == 0o644


# ---------------------------------------------------------------------------
# Q15: update_metadata
# ---------------------------------------------------------------------------


class _Plain(BaseMediaStore):
    async def store(self, data, mime_type, metadata=None):
        return GOOD_KEY

    async def retrieve(self, storage_key):
        return b"", ""

    async def delete(self, storage_key):
        return False

    async def exists(self, storage_key):
        return False


@pytest.mark.asyncio
async def test_base_default_raises() -> None:
    with pytest.raises(NotImplementedError):
        await _Plain().update_metadata(GOOD_KEY, {"a": 1})


class TestUpdateMetadata:
    @pytest.mark.asyncio
    async def test_memory(self) -> None:
        store = InMemoryMediaStore()
        key = await store.store(b"x", "image/png", {"owner_id": "u1", "gone": 1})
        assert await store.update_metadata(key, {"threads": ["t1"], "gone": None}) is True
        meta = await store.get_metadata(key)
        assert meta == {
            "mime_type": "image/png",
            "size_bytes": 1,
            "owner_id": "u1",
            "threads": ["t1"],
        }
        assert await store.update_metadata(key, {"threads": ["t1", "t2"]}) is True
        assert (await store.get_metadata(key))["threads"] == ["t1", "t2"]
        assert await store.update_metadata("f" * 32, {"a": 1}) is False
        assert (await store.retrieve(key)) == (b"x", "image/png")

    @pytest.mark.asyncio
    async def test_local(self, tmp_path) -> None:
        store = LocalFileMediaStore(str(tmp_path))
        key = await store.store(b"x", "image/png", {"owner_id": "u1", "gone": 1})
        assert await store.update_metadata(key, {"threads": ["t1"], "gone": None}) is True
        meta = await store.get_metadata(key)
        assert meta == {
            "mime_type": "image/png",
            "size_bytes": 1,
            "owner_id": "u1",
            "threads": ["t1"],
        }
        assert await store.update_metadata("f" * 32, {"a": 1}) is False
        assert (await store.retrieve(key)) == (b"x", "image/png")
        with pytest.raises(ValueError):
            await store.update_metadata("../x", {"a": 1})

    @pytest.mark.asyncio
    async def test_cloud(self) -> None:
        uploads: list[tuple[str, dict]] = []

        def upload(local_path, cloud_path):
            uploads.append((cloud_path, json.loads(Path(local_path).read_text())))
            return cloud_path

        storage = AsyncMock()
        storage.upload = AsyncMock(side_effect=upload)
        storage.get_public_url = AsyncMock(return_value="https://signed.example.com/x")
        store = CloudMediaStore(storage, prefix="p")

        existing = {"mime_type": "image/png", "ext": ".png", "owner_id": "u1", "gone": 1}
        with patch.object(
            CloudMediaStore,
            "_download_from_url",
            AsyncMock(return_value=json.dumps(existing).encode()),
        ):
            assert await store.update_metadata(GOOD_KEY, {"threads": ["t1"], "gone": None}) is True
        path, written = uploads[-1]
        assert path == f"p/01/23/{GOOD_KEY}.meta.json"
        assert written == {
            "mime_type": "image/png",
            "ext": ".png",
            "owner_id": "u1",
            "threads": ["t1"],
        }

        with patch.object(
            CloudMediaStore, "_download_from_url", AsyncMock(side_effect=OSError("404"))
        ):
            assert await store.update_metadata(GOOD_KEY, {"a": 1}) is False
