"""File tools hide secret files (security review Q2)."""

from __future__ import annotations

import json

import pytest

from tenxgraph.prebuilt.tools import create_file_tools, file_read, file_search, file_write


MARKER = "TOPSECRET-VALUE"
NOT_ALLOWED = {"error": "path is not allowed"}

DEFAULT_HIDDEN = [
    ".env",
    ".env.local",
    "server.pem",
    "tls.key",
    "bundle.p12",
    "bundle.pfx",
    "api-key.json",
    "my_KEY_file.json",
    "service_account.json",
    "service_account_prod.json",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ed25519.pub",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".ssh/config",
    ".aws/credentials",
    ".gnupg/pubring.kbx",
    ".git/config",
    ".git-credentials",
    "terraform.tfstate",
    "terraform.tfstate.backup",
    "prod.tfvars",
]


def _tools(root, **kwargs):
    return {fn.__name__: fn for fn in create_file_tools(root, **kwargs)}


def _make(root, rel, text=MARKER):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _search_dump(tools, query=MARKER, **kwargs):
    """Search and return the raw result list (the reply also echoes the query)."""
    reply = json.loads(tools["scoped_file_search"](query, **kwargs))
    return json.dumps(reply.get("results", reply))


@pytest.fixture
def root(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.mark.parametrize("rel", DEFAULT_HIDDEN)
def test_default_patterns_block_read_write_and_search(root, rel) -> None:
    _make(root, rel)
    _make(root, "visible.txt", "plain")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"](rel)) == NOT_ALLOWED
    for mode in ("create", "overwrite", "append"):
        assert json.loads(tools["scoped_file_write"](rel, "x", mode=mode)) == NOT_ALLOWED
    assert (root / rel).read_text() == MARKER

    dump = _search_dump(tools)
    assert MARKER not in dump
    leaf = rel.rsplit("/", 1)[-1]
    assert _search_dump(tools, leaf) == "[]"
    assert json.loads(tools["scoped_file_search"](MARKER, path=rel)) == NOT_ALLOWED


@pytest.mark.parametrize("rel", [".ssh/new", ".aws/new", ".gnupg/new", ".git/new", "x/.env"])
def test_write_cannot_create_hidden_paths(root, rel) -> None:
    tools = _tools(root)
    result = tools["scoped_file_write"](rel, "x", create_dirs=True)
    assert json.loads(result) == NOT_ALLOWED
    assert not (root / rel).exists()


def test_nested_hidden_files(root) -> None:
    _make(root, "a/b/.env")
    _make(root, "a/b/deep/.ssh/id_rsa")
    _make(root, "a/b/ok.txt", "fine")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"]("a/b/.env")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"]("a/b/deep/.ssh/id_rsa")) == NOT_ALLOWED
    assert MARKER not in _search_dump(tools)
    assert json.loads(tools["scoped_file_read"]("a/b/ok.txt"))["content"] == "fine"


def test_dotdot_cannot_reach_hidden_files(root) -> None:
    _make(root, ".env")
    _make(root, "sub/ok.txt", "fine")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"]("sub/../.env")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"]("sub/./../.env")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"](str(root / "sub" / ".." / ".env"))) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_write"]("sub/../.env", "x", mode="overwrite")) == (
        NOT_ALLOWED
    )
    assert json.loads(tools["scoped_file_read"]("sub/../sub/ok.txt"))["content"] == "fine"


def test_symlink_to_hidden_file_is_hidden(root) -> None:
    _make(root, ".env")
    (root / "innocent.txt").symlink_to(root / ".env")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"]("innocent.txt")) == NOT_ALLOWED
    assert (
        json.loads(tools["scoped_file_write"]("innocent.txt", "x", mode="overwrite")) == NOT_ALLOWED
    )
    assert MARKER not in _search_dump(tools)
    assert _search_dump(tools, "innocent") == "[]"


def test_symlinked_directory_into_hidden_directory(root) -> None:
    _make(root, ".ssh/config")
    (root / "shortcut").symlink_to(root / ".ssh", target_is_directory=True)
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"]("shortcut/config")) == NOT_ALLOWED
    assert (
        json.loads(tools["scoped_file_write"]("shortcut/new", "x", create_dirs=True)) == NOT_ALLOWED
    )
    assert MARKER not in _search_dump(tools)
    assert json.loads(tools["scoped_file_search"](MARKER, path="shortcut")) == NOT_ALLOWED


def test_hidden_symlink_name_is_hidden_even_for_plain_target(root) -> None:
    _make(root, "plain.txt", "fine")
    (root / ".env").symlink_to(root / "plain.txt")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"](".env")) == NOT_ALLOWED


@pytest.mark.parametrize(
    "rel", [".ENV", ".Env.Local", "SERVER.PEM", "Api-KEY.json", "ID_RSA", "x/.SSH/config", ".Git/x"]
)
def test_case_variants_are_hidden(root, rel) -> None:
    _make(root, rel)
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"](rel)) == NOT_ALLOWED
    assert MARKER not in _search_dump(tools)


def test_case_variant_request_for_hidden_file_is_hidden(root) -> None:
    _make(root, ".env")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"](".ENV")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"](".eNv.Production")) == NOT_ALLOWED


def test_error_is_identical_for_existing_and_missing_hidden_paths(root) -> None:
    _make(root, ".env")
    tools = _tools(root)

    for existing, missing in [(".env", ".env.missing"), (".env", "sub/.env")]:
        a = tools["scoped_file_read"](existing)
        b = tools["scoped_file_read"](missing)
        assert a == b
        assert existing not in a and missing not in b
    a = tools["scoped_file_write"](".env", "x", mode="create")
    b = tools["scoped_file_write"](".env.nothing", "x", mode="create")
    assert a == b
    assert tools["scoped_file_read"](".ssh/id") == tools["scoped_file_read"](".env")
    assert tools["scoped_file_read"]("server.pem") == tools["scoped_file_read"]("gone/x.pem")


def test_custom_deny_adds_patterns(root) -> None:
    _make(root, "notes/private.txt")
    _make(root, "secrets/a.txt")
    _make(root, "public.txt", "fine")
    _make(root, ".env")
    tools = _tools(root, deny=["private*", "SECRETS"])

    assert json.loads(tools["scoped_file_read"]("notes/private.txt")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"]("secrets/a.txt")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_write"]("notes/private.txt", "x")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"](".env")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"]("public.txt"))["content"] == "fine"
    assert MARKER not in _search_dump(tools)


def test_custom_deny_matches_relative_path(root) -> None:
    _make(root, "notes/private.txt")
    _make(root, "other/private.txt", "fine")
    tools = _tools(root, deny=["notes/*"])

    assert json.loads(tools["scoped_file_read"]("notes/private.txt")) == NOT_ALLOWED
    assert json.loads(tools["scoped_file_read"]("other/private.txt"))["content"] == "fine"


def test_use_default_deny_false_opts_out(root) -> None:
    _make(root, ".env")
    _make(root, "x/.ssh/id")
    tools = _tools(root, use_default_deny=False)

    assert json.loads(tools["scoped_file_read"](".env"))["content"] == MARKER
    assert json.loads(tools["scoped_file_read"]("x/.ssh/id"))["content"] == MARKER
    assert json.loads(tools["scoped_file_write"](".env", "y", mode="overwrite"))["status"] == (
        "written"
    )


def test_use_default_deny_false_still_applies_custom_deny(root) -> None:
    _make(root, ".env")
    _make(root, "mine.txt")
    tools = _tools(root, deny=["mine.txt"], use_default_deny=False)

    assert json.loads(tools["scoped_file_read"](".env"))["content"] == MARKER
    assert json.loads(tools["scoped_file_read"]("mine.txt")) == NOT_ALLOWED


def test_normal_files_are_unaffected(root) -> None:
    _make(root, "readme.md", "hello")
    _make(root, "src/app.py", "print('hi')")
    _make(root, "data/keys.txt", "not a json key file")
    _make(root, "data/key.jsonl", "close but not matching")
    _make(root, "environment.txt", "env-like name")
    _make(root, "monkey.json", "contains key as a substring only")
    tools = _tools(root)

    assert json.loads(tools["scoped_file_read"]("readme.md"))["content"] == "hello"
    assert json.loads(tools["scoped_file_read"]("environment.txt"))["content"] == "env-like name"
    assert json.loads(tools["scoped_file_read"]("data/keys.txt"))["content"]
    assert json.loads(tools["scoped_file_read"]("data/key.jsonl"))["content"]
    assert json.loads(tools["scoped_file_write"]("src/new.py", "x"))["status"] == "written"
    found = json.loads(tools["scoped_file_search"]("hello"))
    assert [r["path"] for r in found["results"]] == ["readme.md"]


def test_search_with_glob_and_by_name_skips_hidden(root) -> None:
    _make(root, ".env")
    _make(root, "a/.env.prod")
    _make(root, "keys/server.pem")
    tools = _tools(root)

    for query in ("env", "pem", ".env"):
        assert json.loads(tools["scoped_file_search"](query))["results"] == []
    found = json.loads(tools["scoped_file_search"](MARKER, glob="**/*.pem"))
    assert found["results"] == []


def test_module_level_tools_use_defaults(tmp_path, monkeypatch) -> None:
    _make(tmp_path, ".env")
    _make(tmp_path, "ok.txt", "fine")
    monkeypatch.chdir(tmp_path)

    assert json.loads(file_read(".env")) == NOT_ALLOWED
    assert json.loads(file_read("sub/../.env")) == NOT_ALLOWED
    assert json.loads(file_write(".env", "x", mode="overwrite")) == NOT_ALLOWED
    assert json.loads(file_write(".env.new", "x")) == NOT_ALLOWED
    assert json.loads(file_read("ok.txt"))["content"] == "fine"
    assert json.loads(file_search(MARKER))["results"] == []
    assert (tmp_path / ".env").read_text() == MARKER


def test_root_inside_hidden_named_directory_is_not_self_hidden(tmp_path) -> None:
    ws = tmp_path / ".git-mirror" / "work"
    ws.mkdir(parents=True)
    _make(ws, "a.txt", "fine")
    root_in_ssh = tmp_path / ".ssh" / "proj"
    root_in_ssh.mkdir(parents=True)
    _make(root_in_ssh, "b.txt", "fine")

    assert json.loads(_tools(ws)["scoped_file_read"]("a.txt"))["content"] == "fine"
    assert json.loads(_tools(root_in_ssh)["scoped_file_read"]("b.txt"))["content"] == "fine"
