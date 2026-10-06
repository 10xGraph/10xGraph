"""Tests for Sprint 3: Media Storage Layer.

Covers:
- InMemoryMediaStore: store/retrieve/delete/exists roundtrip
- LocalFileMediaStore: same + path traversal prevention + cleanup
- MediaRefResolver: all MediaRef kinds -> correct OpenAI format
- Auto-offload: large inline -> auto-replaced with store reference
- resolve_media_refs: pre-resolves agentflow:// URLs in messages
- Message.with_image / Message.with_file: store-backed constructors
- End-to-end: upload -> store -> message -> verify MediaRef is small
"""

import base64
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tenxgraph.storage.checkpointer import InMemoryCheckpointer
from tenxgraph.storage.media.config import MultimodalConfig
from tenxgraph.storage.media.offload import (
    MediaOffloadPolicy,
    ensure_media_offloaded,
)
from tenxgraph.storage.media.resolver import MediaRefResolver
from tenxgraph.storage.media.storage.base import BaseMediaStore
from tenxgraph.storage.media.storage.local_store import LocalFileMediaStore, _mime_to_ext
from tenxgraph.storage.media.storage.memory_store import InMemoryMediaStore
from tenxgraph.core.state.message import Message
from tenxgraph.core.state.message_block import (
    AudioBlock,
    DocumentBlock,
    ImageBlock,
    MediaRef,
    TextBlock,
    VideoBlock,
)
from tenxgraph.utils.converter import resolve_media_refs


# ===========================================================================
# InMemoryMediaStore
# ===========================================================================


class TestInMemoryMediaStore:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        store = InMemoryMediaStore()
        key = await store.store(b"hello", "text/plain")
        assert isinstance(key, str)
        assert len(key) == 32  # hex UUID

        data, mime = await store.retrieve(key)
        assert data == b"hello"
        assert mime == "text/plain"

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_raises(self):
        store = InMemoryMediaStore()
        with pytest.raises(KeyError, match="Media not found"):
            await store.retrieve("nonexistent")

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryMediaStore()
        key = await store.store(b"data", "image/png")
        assert await store.exists(key)

        deleted = await store.delete(key)
        assert deleted is True
        assert not await store.exists(key)

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self):
        store = InMemoryMediaStore()
        assert await store.delete("nonexistent") is False

    @pytest.mark.asyncio
    async def test_exists(self):
        store = InMemoryMediaStore()
        key = await store.store(b"data", "image/jpeg")
        assert await store.exists(key)
        assert not await store.exists("nope")

    @pytest.mark.asyncio
    async def test_len_and_clear(self):
        store = InMemoryMediaStore()
        await store.store(b"a", "text/plain")
        await store.store(b"b", "text/plain")
        assert len(store) == 2
        store.clear()
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_store_with_metadata(self):
        store = InMemoryMediaStore()
        key = await store.store(b"data", "image/png", metadata={"source": "test"})
        assert await store.exists(key)

    def test_to_media_ref(self):
        store = InMemoryMediaStore()
        ref = store.to_media_ref("abc123", "image/jpeg")
        assert ref.kind == "url"
        assert ref.url == "graph://media/abc123"
        assert ref.mime_type == "image/jpeg"

    def test_to_media_ref_with_extras(self):
        store = InMemoryMediaStore()
        ref = store.to_media_ref("abc", "image/png", width=100, height=200, filename="test.png")
        assert ref.width == 100
        assert ref.height == 200
        assert ref.filename == "test.png"


# ===========================================================================
# LocalFileMediaStore
# ===========================================================================


class TestLocalFileMediaStore:
    @pytest.fixture
    def tmp_dir(self):
        d = tempfile.mkdtemp()
        yield d
        shutil.rmtree(d, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        key = await store.store(b"pdf bytes", "application/pdf")

        data, mime = await store.retrieve(key)
        assert data == b"pdf bytes"
        assert mime == "application/pdf"

    @pytest.mark.asyncio
    async def test_sharded_directory_structure(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        key = await store.store(b"data", "image/jpeg")

        # Verify sharded path exists
        shard1 = key[:2]
        shard2 = key[2:4]
        shard_dir = Path(tmp_dir) / shard1 / shard2
        assert shard_dir.exists()

        # Both data file and meta file should exist
        files = list(shard_dir.iterdir())
        assert len(files) == 2  # data file + meta.json

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_raises(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        with pytest.raises(KeyError, match="Media not found"):
            await store.retrieve("a" * 32)

    @pytest.mark.asyncio
    async def test_delete(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        key = await store.store(b"data", "image/png")
        assert await store.exists(key)

        deleted = await store.delete(key)
        assert deleted is True
        assert not await store.exists(key)

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        assert await store.delete("a" * 32) is False

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        with pytest.raises(ValueError, match="Invalid storage key"):
            await store.retrieve("../../etc/passwd")

    @pytest.mark.asyncio
    async def test_path_traversal_dots(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        with pytest.raises(ValueError, match="Invalid storage key"):
            await store.retrieve("../../../etc/shadow")

    @pytest.mark.asyncio
    async def test_path_traversal_slashes(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        with pytest.raises(ValueError, match="Invalid storage key"):
            await store.retrieve("abc/def/ghi")

    @pytest.mark.asyncio
    async def test_metadata_stored(self, tmp_dir):
        store = LocalFileMediaStore(base_dir=tmp_dir)
        key = await store.store(b"data", "image/png", metadata={"tag": "test"})

        meta_path = store._meta_path(key)
        meta = json.loads(meta_path.read_text())
        assert meta["mime_type"] == "image/png"
        assert meta["size_bytes"] == 4
        assert meta["tag"] == "test"


class TestMimeToExt:
    def test_jpeg(self):
        assert _mime_to_ext("image/jpeg") in (".jpg", ".jpeg")

    def test_png(self):
        assert _mime_to_ext("image/png") == ".png"

    def test_pdf(self):
        assert _mime_to_ext("application/pdf") == ".pdf"

    def test_unknown(self):
        assert _mime_to_ext("application/x-unknown-thing") == ".bin"


# ===========================================================================
# MediaRefResolver (OpenAI format only — Google requires google-genai)
# ===========================================================================


class TestMediaRefResolverOpenAI:
    @pytest.mark.asyncio
    async def test_resolve_external_url(self):
        resolver = MediaRefResolver()
        ref = MediaRef(kind="url", url="https://example.com/img.jpg", mime_type="image/jpeg")
        result = await resolver.resolve_for_openai(ref)
        assert result == {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}

    @pytest.mark.asyncio
    async def test_resolve_inline_base64(self):
        resolver = MediaRefResolver()
        ref = MediaRef(kind="data", data_base64="abc123", mime_type="image/png")
        result = await resolver.resolve_for_openai(ref)
        assert result["image_url"]["url"] == "data:image/png;base64,abc123"

    @pytest.mark.asyncio
    async def test_resolve_file_id(self):
        resolver = MediaRefResolver()
        ref = MediaRef(kind="file_id", file_id="file-abc")
        result = await resolver.resolve_for_openai(ref)
        assert result["image_url"]["url"] == "file-abc"

    @pytest.mark.asyncio
    async def test_resolve_internal_url(self):
        store = InMemoryMediaStore()
        key = await store.store(b"image data", "image/jpeg")

        resolver = MediaRefResolver(media_store=store)
        ref = MediaRef(kind="url", url=f"graph://media/{key}", mime_type="image/jpeg")
        result = await resolver.resolve_for_openai(ref)

        expected_b64 = base64.b64encode(b"image data").decode()
        assert result["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    @pytest.mark.asyncio
    async def test_resolve_internal_url_prefers_direct_store_url(self):
        store = MagicMock(spec=BaseMediaStore)
        store.get_direct_url = AsyncMock(return_value="https://signed.example.com/image.jpg")

        resolver = MediaRefResolver(media_store=store)
        ref = MediaRef(kind="url", url="graph://media/abc123", mime_type="image/jpeg")
        result = await resolver.resolve_for_openai(ref)

        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://signed.example.com/image.jpg"},
        }
        store.get_direct_url.assert_awaited_once_with("abc123", mime_type="image/jpeg")
        store.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_internal_url_caches_direct_store_url(self):
        store = MagicMock(spec=BaseMediaStore)
        store.get_direct_url = AsyncMock(return_value="https://signed.example.com/image.jpg")
        cache = InMemoryCheckpointer()

        resolver = MediaRefResolver(
            media_store=store,
            cache_backend=cache,
            direct_url_expiration_seconds=3600,
        )
        ref = MediaRef(kind="url", url="graph://media/abc123", mime_type="image/jpeg")

        first = await resolver.resolve_for_openai(ref)
        second = await resolver.resolve_for_openai(ref)

        assert first == second
        store.get_direct_url.assert_awaited_once_with(
            "abc123",
            mime_type="image/jpeg",
            expiration=3600,
        )

    @pytest.mark.asyncio
    async def test_resolve_internal_url_no_store_raises(self):
        resolver = MediaRefResolver()  # no store
        ref = MediaRef(kind="url", url="graph://media/abc123")
        with pytest.raises(RuntimeError, match="no MediaStore configured"):
            await resolver.resolve_for_openai(ref)

    @pytest.mark.asyncio
    async def test_resolve_inline_default_mime(self):
        resolver = MediaRefResolver()
        ref = MediaRef(kind="data", data_base64="xyz")
        result = await resolver.resolve_for_openai(ref)
        assert "application/octet-stream" in result["image_url"]["url"]


# ===========================================================================
# Auto-offload (ensure_media_offloaded)
# ===========================================================================


class TestAutoOffload:
    @pytest.mark.asyncio
    async def test_offload_large_image(self):
        store = InMemoryMediaStore()
        # Create a message with a ~100KB inline image
        large_b64 = base64.b64encode(b"x" * 100_000).decode()
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Describe this"),
                ImageBlock(media=MediaRef(kind="data", data_base64=large_b64, mime_type="image/jpeg")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, max_inline_bytes=50_000)

        # Image should now reference the store
        img_block = result.content[1]
        assert isinstance(img_block, ImageBlock)
        assert img_block.media.kind == "url"
        assert img_block.media.url.startswith("graph://media/")
        assert img_block.media.data_base64 is None

        # Store should have the data
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_no_offload_small_image(self):
        store = InMemoryMediaStore()
        small_b64 = base64.b64encode(b"tiny").decode()
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="data", data_base64=small_b64, mime_type="image/png")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, max_inline_bytes=50_000)

        img_block = result.content[0]
        assert isinstance(img_block, ImageBlock)
        assert img_block.media.kind == "data"  # NOT offloaded
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_offload_policy_never(self):
        store = InMemoryMediaStore()
        large_b64 = base64.b64encode(b"x" * 100_000).decode()
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="data", data_base64=large_b64, mime_type="image/jpeg")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, policy=MediaOffloadPolicy.NEVER)
        img_block = result.content[0]
        assert img_block.media.kind == "data"  # NOT offloaded
        assert len(store) == 0

    @pytest.mark.asyncio
    async def test_offload_policy_always(self):
        store = InMemoryMediaStore()
        tiny_b64 = base64.b64encode(b"a").decode()
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="data", data_base64=tiny_b64, mime_type="image/png")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, policy=MediaOffloadPolicy.ALWAYS)
        img_block = result.content[0]
        assert img_block.media.kind == "url"  # Always offloaded
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_offload_preserves_metadata(self):
        store = InMemoryMediaStore()
        b64 = base64.b64encode(b"x" * 100_000).decode()
        msg = Message(
            role="user",
            content=[
                ImageBlock(
                    media=MediaRef(
                        kind="data",
                        data_base64=b64,
                        mime_type="image/jpeg",
                        filename="photo.jpg",
                        width=800,
                        height=600,
                    )
                ),
            ],
        )

        result = await ensure_media_offloaded(msg, store, max_inline_bytes=1000)
        ref = result.content[0].media
        assert ref.filename == "photo.jpg"
        assert ref.width == 800
        assert ref.height == 600
        assert ref.size_bytes == 100_000

    @pytest.mark.asyncio
    async def test_offload_skips_url_refs(self):
        store = InMemoryMediaStore()
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.jpg")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, policy=MediaOffloadPolicy.ALWAYS)
        assert result.content[0].media.kind == "url"
        assert result.content[0].media.url == "https://example.com/img.jpg"
        assert len(store) == 0  # Nothing offloaded

    @pytest.mark.asyncio
    async def test_offload_audio_and_document(self):
        store = InMemoryMediaStore()
        b64 = base64.b64encode(b"x" * 100_000).decode()
        msg = Message(
            role="user",
            content=[
                AudioBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="audio/wav")),
                DocumentBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="application/pdf")),
            ],
        )

        result = await ensure_media_offloaded(msg, store, max_inline_bytes=1000)
        assert result.content[0].media.kind == "url"
        assert result.content[1].media.kind == "url"
        assert len(store) == 2

    @pytest.mark.asyncio
    async def test_offload_text_blocks_untouched(self):
        store = InMemoryMediaStore()
        msg = Message(
            role="user",
            content=[TextBlock(text="Hello world")],
        )

        result = await ensure_media_offloaded(msg, store, policy=MediaOffloadPolicy.ALWAYS)
        assert isinstance(result.content[0], TextBlock)
        assert result.content[0].text == "Hello world"
        assert len(store) == 0


# ===========================================================================
# resolve_media_refs
# ===========================================================================


class TestResolveMediaRefs:
    @pytest.mark.asyncio
    async def test_resolve_internal_url_to_base64(self):
        store = InMemoryMediaStore()
        key = await store.store(b"raw image data", "image/jpeg")

        msg = Message(
            role="user",
            content=[
                TextBlock(text="Describe"),
                ImageBlock(media=MediaRef(kind="url", url=f"graph://media/{key}")),
            ],
        )

        resolver = MediaRefResolver(media_store=store)
        result = await resolve_media_refs([msg], resolver)

        img_block = result[0].content[1]
        assert isinstance(img_block, ImageBlock)
        assert img_block.media.kind == "data"
        assert img_block.media.data_base64 is not None
        assert img_block.media.mime_type == "image/jpeg"
        assert img_block.media.size_bytes == len(b"raw image data")

    @pytest.mark.asyncio
    async def test_external_urls_untouched(self):
        store = InMemoryMediaStore()
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="url", url="https://example.com/img.jpg")),
            ],
        )

        resolver = MediaRefResolver(media_store=store)
        result = await resolve_media_refs([msg], resolver)

        assert result[0].content[0].media.url == "https://example.com/img.jpg"
        assert result[0].content[0].media.kind == "url"

    @pytest.mark.asyncio
    async def test_base64_refs_untouched(self):
        msg = Message(
            role="user",
            content=[
                ImageBlock(media=MediaRef(kind="data", data_base64="abc", mime_type="image/png")),
            ],
        )

        store = InMemoryMediaStore()
        resolver = MediaRefResolver(media_store=store)
        result = await resolve_media_refs([msg], resolver)

        assert result[0].content[0].media.kind == "data"
        assert result[0].content[0].media.data_base64 == "abc"


# ===========================================================================
# Message.with_image / Message.with_file
# ===========================================================================


class TestMessageWithImage:
    @pytest.mark.asyncio
    async def test_with_image_stores_and_references(self):
        store = InMemoryMediaStore()
        msg = await Message.with_image(
            data=b"jpeg data",
            mime_type="image/jpeg",
            store=store,
            text="Describe this",
        )

        assert msg.role == "user"
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert msg.content[0].text == "Describe this"
        assert isinstance(msg.content[1], ImageBlock)
        assert msg.content[1].media.kind == "url"
        assert msg.content[1].media.url.startswith("graph://media/")
        assert msg.content[1].media.size_bytes == 9
        assert len(store) == 1

    @pytest.mark.asyncio
    async def test_with_image_no_text(self):
        store = InMemoryMediaStore()
        msg = await Message.with_image(data=b"png", mime_type="image/png", store=store)

        assert len(msg.content) == 1
        assert isinstance(msg.content[0], ImageBlock)


class TestMessageWithFile:
    @pytest.mark.asyncio
    async def test_with_file_image(self):
        store = InMemoryMediaStore()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpeg")
            f.flush()
            path = f.name

        try:
            msg = await Message.with_file(path, store=store, text="Check this")
            assert len(msg.content) == 2
            assert isinstance(msg.content[0], TextBlock)
            assert isinstance(msg.content[1], ImageBlock)
            assert msg.content[1].media.url.startswith("graph://media/")
            assert msg.content[1].media.filename == os.path.basename(path)
            assert len(store) == 1
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_with_file_pdf(self):
        store = InMemoryMediaStore()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf")
            f.flush()
            path = f.name

        try:
            msg = await Message.with_file(path, store=store)
            assert len(msg.content) == 1
            assert isinstance(msg.content[0], DocumentBlock)
            assert msg.content[0].media.url.startswith("graph://media/")
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_with_file_not_found(self):
        store = InMemoryMediaStore()
        with pytest.raises(FileNotFoundError):
            await Message.with_file("/nonexistent/file.txt", store=store)

    @pytest.mark.asyncio
    async def test_with_file_audio(self):
        store = InMemoryMediaStore()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"fake wav")
            f.flush()
            path = f.name

        try:
            msg = await Message.with_file(path, store=store)
            assert isinstance(msg.content[0], AudioBlock)
        finally:
            os.unlink(path)


# ===========================================================================
# End-to-end: upload -> store -> message -> verify small reference
# ===========================================================================


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_pipeline_keeps_state_small(self):
        """Verify that after storing via MediaStore, the message serialization is tiny."""
        store = InMemoryMediaStore()

        # Simulate a 1MB image
        big_data = b"x" * 1_000_000
        msg = await Message.with_image(
            data=big_data,
            mime_type="image/jpeg",
            store=store,
            text="Analyze this large image",
        )

        # Serialize the message (as would happen in checkpointer)
        serialized = msg.model_dump(mode="json")
        serialized_json = json.dumps(serialized)

        # The serialized message should be tiny (no 1MB base64 blob)
        assert len(serialized_json) < 1000, (
            f"Serialized message too large: {len(serialized_json)} bytes. "
            "MediaRef should be a small reference, not inline data."
        )

        # But the actual data is still retrievable
        img_ref = msg.content[1].media
        key = img_ref.url.removeprefix("graph://media/")
        data, mime = await store.retrieve(key)
        assert len(data) == 1_000_000
        assert mime == "image/jpeg"

    @pytest.mark.asyncio
    async def test_offload_then_resolve_roundtrip(self):
        """Offload large inline -> store -> resolve back for LLM call."""
        store = InMemoryMediaStore()

        # Start with inline base64 (simulating client upload)
        raw = b"image_data_here"
        b64 = base64.b64encode(raw).decode()
        msg = Message(
            role="user",
            content=[
                TextBlock(text="Describe"),
                ImageBlock(media=MediaRef(kind="data", data_base64=b64, mime_type="image/png")),
            ],
        )

        # Step 1: Offload to store
        await ensure_media_offloaded(msg, store, policy=MediaOffloadPolicy.ALWAYS)
        assert msg.content[1].media.kind == "url"

        # Step 2: Resolve back for OpenAI
        resolver = MediaRefResolver(media_store=store)
        result = await resolver.resolve_for_openai(msg.content[1].media)
        expected_b64 = base64.b64encode(raw).decode()
        assert result["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"

    @pytest.mark.asyncio
    async def test_local_file_store_roundtrip(self):
        """Full roundtrip with LocalFileMediaStore."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = LocalFileMediaStore(base_dir=tmp_dir)

            key = await store.store(b"hello world", "text/plain", metadata={"test": True})
            assert await store.exists(key)

            data, mime = await store.retrieve(key)
            assert data == b"hello world"
            assert mime == "text/plain"

            ref = store.to_media_ref(key, "text/plain")
            assert ref.url == f"graph://media/{key}"

            await store.delete(key)
            assert not await store.exists(key)
