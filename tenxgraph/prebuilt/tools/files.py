"""Workspace-scoped file tools for AgentFlow agents."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any, Literal

from agentflow.utils.decorators import tool


_DEFAULT_MAX_READ_CHARS = 20_000
_DEFAULT_MAX_WRITE_CHARS = 200_000
_DEFAULT_MAX_SEARCH_RESULTS = 20
_MAX_SEARCH_FILE_SIZE = 1_000_000
_MAX_SEARCH_PREVIEW_CHARS = 240
_SEARCH_PREVIEW_ELLIPSIS_CHARS = 3
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
}


def _configured_root(config: dict[str, Any] | None) -> Path:
    cfg = config or {}
    root = cfg.get("file_tool_root") or cfg.get("workspace_root") or "."
    return Path(str(root)).expanduser().resolve()


def _resolve_under_root(path: str, root: Path) -> Path:
    if not path or not path.strip():
        raise ValueError("path is required")

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"path must stay within the configured root: {root}") from None

    return resolved


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _is_probably_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except OSError:
        return False
    return b"\x00" not in chunk


def _write_file_content(
    target: Path, content: str, mode: Literal["create", "overwrite", "append"]
) -> None:
    if mode == "append":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        return
    if mode in {"overwrite", "create"}:
        target.write_text(content, encoding="utf-8")
        return
    raise ValueError(f"unsupported mode: {mode}")


def _is_search_candidate(candidate: Path, glob: str) -> bool:
    if any(part in _SKIP_DIRS for part in candidate.parts):
        return False
    if not candidate.is_file():
        return False
    if candidate.stat().st_size > _MAX_SEARCH_FILE_SIZE:
        return False
    return glob == "**/*" or fnmatch.fnmatch(candidate.name, Path(glob).name)


def _trim_search_preview(line: str) -> str:
    preview = line.strip()
    if len(preview) > _MAX_SEARCH_PREVIEW_CHARS:
        preview_limit = _MAX_SEARCH_PREVIEW_CHARS - _SEARCH_PREVIEW_ELLIPSIS_CHARS
        return f"{preview[:preview_limit]}..."
    return preview


def _append_content_matches(
    results: list[dict[str, Any]],
    candidate: Path,
    relative_path: str,
    query_lower: str,
    limit: int,
) -> None:
    try:
        for line_no, line in enumerate(
            candidate.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if query_lower not in line.lower():
                continue
            results.append(
                {
                    "path": relative_path,
                    "match_type": "content",
                    "line": line_no,
                    "preview": _trim_search_preview(line),
                }
            )
            if len(results) >= limit:
                break
    except OSError:
        return


@tool(
    name="file_read",
    description=(
        "Read a UTF-8 text file from the configured workspace root. Supports optional "
        "1-based start_line/end_line and truncates long output."
    ),
    tags=["file", "filesystem", "read"],
    capabilities=["read_files"],
)
def file_read(
    path: str,
    start_line: int = 1,
    end_line: int = 0,
    max_chars: int = _DEFAULT_MAX_READ_CHARS,
    config: dict[str, Any] | None = None,
) -> str:
    """Read a workspace-scoped text file."""
    root = _configured_root(config)
    try:
        target = _resolve_under_root(path, root)
        if not target.exists():
            return json.dumps({"error": "file does not exist", "path": _relative(target, root)})
        if not target.is_file():
            return json.dumps({"error": "path is not a file", "path": _relative(target, root)})
        if not _is_probably_text(target):
            return json.dumps(
                {"error": "file appears to be binary", "path": _relative(target, root)}
            )

        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        end = int(end_line) if end_line else len(lines)
        if end < start:
            return json.dumps({"error": "end_line must be greater than or equal to start_line"})

        selected = lines[start - 1 : end]
        text = "\n".join(selected)
        limit = max(1, min(int(max_chars), _DEFAULT_MAX_READ_CHARS))
        truncated = len(text) > limit
        if truncated:
            text = text[:limit]

        return json.dumps(
            {
                "path": _relative(target, root),
                "start_line": start,
                "end_line": min(end, len(lines)),
                "content": text,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(
    name="file_write",
    description=(
        "Write UTF-8 text to a file under the configured workspace root. Use mode='create' "
        "to avoid overwriting, mode='overwrite' to replace, or mode='append' to append."
    ),
    tags=["file", "filesystem", "write"],
    capabilities=["write_files"],
)
def file_write(
    path: str,
    content: str,
    mode: Literal["create", "overwrite", "append"] = "create",
    create_dirs: bool = False,
    config: dict[str, Any] | None = None,
) -> str:
    """Write UTF-8 text to a workspace-scoped file."""
    root = _configured_root(config)
    try:
        target = _resolve_under_root(path, root)
        if len(content) > _DEFAULT_MAX_WRITE_CHARS:
            result = {"error": "content is too large"}
        elif target.exists() and not target.is_file():
            result = {"error": "path exists and is not a file", "path": _relative(target, root)}
        elif target.exists() and mode == "create":
            result = {"error": "file already exists", "path": _relative(target, root)}
        elif mode not in {"create", "overwrite", "append"}:
            result = {"error": f"unsupported mode: {mode}"}
        elif not target.parent.exists() and not create_dirs:
            result = {"error": "parent directory does not exist"}
        else:
            if not target.parent.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
            _write_file_content(target, content, mode)
            result = {
                "status": "written",
                "path": _relative(target, root),
                "bytes": len(content.encode("utf-8")),
                "mode": mode,
            }
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(
    name="file_search",
    description=(
        "Search text files under the configured workspace root by filename and content. "
        "Returns relative paths, line numbers, and short previews."
    ),
    tags=["file", "filesystem", "search"],
    capabilities=["read_files"],
)
def file_search(
    query: str,
    path: str = "",
    glob: str = "**/*",
    max_results: int = _DEFAULT_MAX_SEARCH_RESULTS,
    config: dict[str, Any] | None = None,
) -> str:
    """Search workspace-scoped text files by filename and content."""
    if not query:
        return json.dumps({"error": "query is required"})

    root = _configured_root(config)
    try:
        search_root = _resolve_under_root(path or ".", root)
        if not search_root.exists():
            return json.dumps({"error": "search path does not exist"})

        query_lower = query.lower()
        limit = max(1, min(int(max_results), 100))
        results: list[dict[str, Any]] = []
        candidates = [search_root] if search_root.is_file() else search_root.rglob(glob)

        for candidate in candidates:
            if len(results) >= limit:
                break
            if not _is_search_candidate(candidate, glob):
                continue

            relative_path = _relative(candidate, root)
            if query_lower in candidate.name.lower():
                results.append(
                    {
                        "path": relative_path,
                        "match_type": "filename",
                        "line": None,
                        "preview": candidate.name,
                    }
                )
                if len(results) >= limit:
                    break

            if not _is_probably_text(candidate):
                continue

            _append_content_matches(results, candidate, relative_path, query_lower, limit)

        return json.dumps(
            {"query": query, "root": _relative(search_root, root), "results": results}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})
