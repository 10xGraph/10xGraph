"""Workspace-scoped file tools for AgentFlow agents."""

from __future__ import annotations

import fnmatch
import json
import os
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, NamedTuple

from tenxgraph.utils.decorators import tool


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
_DEFAULT_DENY_FILES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*key*.json",
    "service_account*.json",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    ".npmrc",
    ".pypirc",
)
_DEFAULT_DENY_DIRS = (".ssh", ".aws", ".gnupg", ".git")
_NOT_ALLOWED = "path is not allowed"


class _DenyRules(NamedTuple):
    """Casefolded rules for paths the file tools must hide.

    ``files`` match the final path component, ``dirs`` are names that hide everything under a
    directory (or file) with that name, and ``any_part`` patterns match every component and the
    whole relative path.
    """

    files: tuple[str, ...]
    dirs: tuple[str, ...]
    any_part: tuple[str, ...]


_DEFAULT_RULES = _DenyRules(
    tuple(p.casefold() for p in _DEFAULT_DENY_FILES),
    tuple(d.casefold() for d in _DEFAULT_DENY_DIRS),
    (),
)


class _HiddenPathError(ValueError):
    """Raised for a hidden path. The text is the same whether or not the path exists."""

    def __init__(self) -> None:
        super().__init__(_NOT_ALLOWED)


def _build_rules(deny: Iterable[str], use_default_deny: bool) -> _DenyRules:
    extra = tuple(str(p).casefold() for p in deny)
    base = _DEFAULT_RULES if use_default_deny else _DenyRules((), (), ())
    return _DenyRules(base.files, base.dirs, base.any_part + extra)


def _is_hidden(path: Path, root: Path, rules: _DenyRules) -> bool:
    """Return True if ``path`` (under ``root``) matches a deny rule, ignoring case."""
    try:
        parts = [part.casefold() for part in path.relative_to(root).parts]
    except ValueError:
        return False
    if not parts:
        return False
    relative = "/".join(parts)
    if any(part in rules.dirs for part in parts):
        return True
    if any(fnmatch.fnmatchcase(parts[-1], pattern) for pattern in rules.files):
        return True
    return any(
        fnmatch.fnmatchcase(part, pattern) or fnmatch.fnmatchcase(relative, pattern)
        for pattern in rules.any_part
        for part in parts
    )


def _configured_root(config: dict[str, Any] | None) -> Path:
    cfg = config or {}
    root = cfg.get("file_tool_root") or cfg.get("workspace_root")
    if root:
        warnings.warn(
            "Setting the file tool root through the run config (file_tool_root / "
            "workspace_root) is deprecated and will be removed in 0.11: anyone who can set "
            "the run config can point the tools at any directory. Use "
            "create_file_tools(root=...) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    return Path(str(root or ".")).expanduser().resolve()


def _resolve_under_root(path: str, root: Path, rules: _DenyRules = _DEFAULT_RULES) -> Path:
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

    # Check the path as typed (``..`` collapsed, symlinks kept) and the resolved path, so a
    # hidden name is caught whether it is the link or its target.
    if _is_hidden(Path(os.path.normpath(candidate)), root, rules) or _is_hidden(
        resolved, root, rules
    ):
        raise _HiddenPathError

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


def _search_target(
    candidate: Path, glob: str, root: Path, rules: _DenyRules = _DEFAULT_RULES
) -> Path | None:
    """Return the file to search for ``candidate``, or None to skip it.

    A symlink inside the workspace can point outside it, so the candidate is resolved and
    must still be under ``root``, the same rule ``file_read`` applies. Hidden files and
    directories (see ``_DenyRules``) are skipped, by their own name and by their target.
    """
    if any(part in _SKIP_DIRS for part in candidate.parts) or _is_hidden(candidate, root, rules):
        return None
    try:
        target = candidate.resolve()
        target.relative_to(root)
    except (OSError, ValueError):
        return None
    if _is_hidden(target, root, rules) or not target.is_file():
        return None
    if target.stat().st_size > _MAX_SEARCH_FILE_SIZE:
        return None
    if glob != "**/*" and not fnmatch.fnmatch(candidate.name, Path(glob).name):
        return None
    return target


def _trim_search_preview(line: str) -> str:
    preview = line.strip()
    if len(preview) > _MAX_SEARCH_PREVIEW_CHARS:
        preview_limit = _MAX_SEARCH_PREVIEW_CHARS - _SEARCH_PREVIEW_ELLIPSIS_CHARS
        return f"{preview[:preview_limit]}..."
    return preview


def _append_content_matches(
    results: list[dict[str, Any]],
    target: Path,
    relative_path: str,
    query_lower: str,
    limit: int,
) -> None:
    try:
        for line_no, line in enumerate(
            target.read_text(encoding="utf-8", errors="replace").splitlines(),
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


_READ_DESCRIPTION = (
    "Read a UTF-8 text file from the configured workspace root. Supports optional "
    "1-based start_line/end_line and truncates long output."
)
_WRITE_DESCRIPTION = (
    "Write UTF-8 text to a file under the configured workspace root. Use mode='create' "
    "to avoid overwriting, mode='overwrite' to replace, or mode='append' to append."
)
_SEARCH_DESCRIPTION = (
    "Search text files under the configured workspace root by filename and content. "
    "Returns relative paths, line numbers, and short previews."
)
_READ_TAGS = ["file", "filesystem", "read"]
_WRITE_TAGS = ["file", "filesystem", "write"]
_SEARCH_TAGS = ["file", "filesystem", "search"]


def _read(
    path: str,
    start_line: int,
    end_line: int,
    max_chars: int,
    root: Path,
    rules: _DenyRules = _DEFAULT_RULES,
) -> str:
    try:
        target = _resolve_under_root(path, root, rules)
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


def _write(
    path: str,
    content: str,
    mode: Literal["create", "overwrite", "append"],
    create_dirs: bool,
    root: Path,
    rules: _DenyRules = _DEFAULT_RULES,
) -> str:
    try:
        target = _resolve_under_root(path, root, rules)
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


def _search(
    query: str,
    path: str,
    glob: str,
    max_results: int,
    root: Path,
    rules: _DenyRules = _DEFAULT_RULES,
) -> str:
    if not query:
        return json.dumps({"error": "query is required"})

    try:
        search_root = _resolve_under_root(path or ".", root, rules)
        if not search_root.exists():
            return json.dumps({"error": "search path does not exist"})

        query_lower = query.lower()
        limit = max(1, min(int(max_results), 100))
        results: list[dict[str, Any]] = []
        candidates = [search_root] if search_root.is_file() else search_root.rglob(glob)

        for candidate in candidates:
            if len(results) >= limit:
                break
            target = _search_target(candidate, glob, root, rules)
            if target is None:
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

            if not _is_probably_text(target):
                continue

            _append_content_matches(results, target, relative_path, query_lower, limit)

        return json.dumps(
            {"query": query, "root": _relative(search_root, root), "results": results}
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool(
    name="file_read",
    description=_READ_DESCRIPTION,
    tags=_READ_TAGS,
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
    return _read(path, start_line, end_line, max_chars, _configured_root(config))


@tool(
    name="file_write",
    description=_WRITE_DESCRIPTION,
    tags=_WRITE_TAGS,
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
    return _write(path, content, mode, create_dirs, _configured_root(config))


@tool(
    name="file_search",
    description=_SEARCH_DESCRIPTION,
    tags=_SEARCH_TAGS,
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
    return _search(query, path, glob, max_results, _configured_root(config))


def create_file_tools(
    root: str | Path,
    *,
    read: bool = True,
    write: bool = True,
    search: bool = True,
    deny: Iterable[str] = (),
    use_default_deny: bool = True,
) -> list[Callable[..., str]]:
    """Create ``file_read``, ``file_write`` and ``file_search`` locked to ``root``.

    The root is fixed when the tools are created, so nothing in the run config can move it.
    Prefer these over the module-level tools in any graph that serves requests.

    Secret files are hidden by default, whatever the root: ``.env``, ``.env.*``, ``*.pem``,
    ``*.key``, ``*.p12``, ``*.pfx``, ``*key*.json``, ``service_account*.json``, ``id_rsa*``,
    ``id_ed25519*``, ``.netrc``, ``.npmrc``, ``.pypirc`` and anything under a directory named
    ``.ssh``, ``.aws``, ``.gnupg`` or ``.git``. Matching ignores case and uses the resolved
    path, so symlinks and ``..`` cannot reach them. Reading or writing a hidden path returns
    the same "not allowed" error whether or not it exists, and search skips it.

    Args:
        root: Directory the tools may read and write under. Must exist.
        read: Include ``file_read``.
        write: Include ``file_write``.
        search: Include ``file_search``.
        deny: Extra glob patterns to hide. Each is matched, ignoring case, against every
            component of the path relative to ``root`` and against the whole relative path.
        use_default_deny: Set to False to drop the default patterns above. Only do this when
            the root holds nothing sensitive.

    Returns:
        The selected tools, ready to pass to ``ToolNode``.

    Raises:
        ValueError: If ``root`` is not an existing directory.

    Example:
        ```python
        from tenxgraph.core.graph import ToolNode
        from tenxgraph.prebuilt.tools import create_file_tools

        tool_node = ToolNode(create_file_tools("/srv/workspace", write=False))
        ```
    """
    fixed_root = Path(root).expanduser().resolve()
    if not fixed_root.is_dir():
        raise ValueError(f"file tool root is not a directory: {fixed_root}")

    rules = _build_rules(deny, use_default_deny)
    tools: list[Callable[..., str]] = []

    if read:

        @tool(
            name="file_read",
            description=_READ_DESCRIPTION,
            tags=_READ_TAGS,
            capabilities=["read_files"],
        )
        def scoped_file_read(
            path: str,
            start_line: int = 1,
            end_line: int = 0,
            max_chars: int = _DEFAULT_MAX_READ_CHARS,
        ) -> str:
            """Read a workspace-scoped text file."""
            return _read(path, start_line, end_line, max_chars, fixed_root, rules)

        tools.append(scoped_file_read)

    if write:

        @tool(
            name="file_write",
            description=_WRITE_DESCRIPTION,
            tags=_WRITE_TAGS,
            capabilities=["write_files"],
        )
        def scoped_file_write(
            path: str,
            content: str,
            mode: Literal["create", "overwrite", "append"] = "create",
            create_dirs: bool = False,
        ) -> str:
            """Write UTF-8 text to a workspace-scoped file."""
            return _write(path, content, mode, create_dirs, fixed_root, rules)

        tools.append(scoped_file_write)

    if search:

        @tool(
            name="file_search",
            description=_SEARCH_DESCRIPTION,
            tags=_SEARCH_TAGS,
            capabilities=["read_files"],
        )
        def scoped_file_search(
            query: str,
            path: str = "",
            glob: str = "**/*",
            max_results: int = _DEFAULT_MAX_SEARCH_RESULTS,
        ) -> str:
            """Search workspace-scoped text files by filename and content."""
            return _search(query, path, glob, max_results, fixed_root, rules)

        tools.append(scoped_file_search)

    return tools
