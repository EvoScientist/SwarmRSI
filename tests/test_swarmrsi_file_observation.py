"""Forward inspect ranges and lossless continuations; no model or benchmark."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

import pytest

from swarmrsi.coder_engine import ContractBoundedCoder
from swarmrsi.coder_openrouter import OpenRouterCoderModel
from swarmrsi.file_observation import read_text_page
from test_swarmrsi_coder import config, workspace


def source_text(page):
    text = re.sub(r"(?m)^\d+: ", "", page["content"])
    return text + ("\n" if page["last_line_terminated"] else "")


def collect(path, *, budget=256, **request):
    pieces, pages = [], []
    action = {"display_path": path.name, **request}
    prior = None
    for _ in range(3000):
        page = read_text_page(path, max_chars=budget, **action)
        assert len(page["content"]) <= budget
        assert len(json.dumps({k: v for k, v in page.items() if k != "content"})) < 2000
        pieces.append(source_text(page)); pages.append(page)
        if page["next_read"] is None:
            break
        next_action = page["next_read"]
        cursor = (next_action["start_line"], next_action["start_column"])
        assert prior is None or cursor > prior
        prior = cursor
        action = {"display_path": next_action["path"],
                  **{k: v for k, v in next_action.items() if k not in ("tool", "path")}}
    else:
        pytest.fail("non-advancing or unexpectedly long pagination")
    return "".join(pieces), pages


@pytest.mark.parametrize("header,row", [
    ("MARKER_A,MARKER_B,MARKER_C", "1.123456,2.234567,3.345678"),
    ("def important_entrypoint():", "    value = 'long implementation detail'"),
])
def test_old_tail_loses_requested_header_but_new_page_keeps_it(tmp_path, header, row):
    path = tmp_path / "input.txt"
    lines = [header, *[row for _ in range(420)]]
    path.write_text("\n".join(lines) + "\n")
    old_rendered = "\n".join(f"{i}: {line}" for i, line in enumerate(lines[:400], 1))
    assert header not in old_rendered[-256:]
    page = read_text_page(path, display_path=path.name, max_chars=256)
    assert page["content"].startswith("1: " + header)
    assert page["truncated"] and page["next_read"]
    assert page["returned_range"]["start"] == {"line": 1, "column": 1}
    restored, _ = collect(path)
    assert restored == path.read_text()


@pytest.mark.parametrize("text", [
    "", "\n", "a", "a\n", "a\n\n", "\n\nabc", "短🙂列\r\n下一行\r\n",
    "HEADER," + "长🙂x," * 700 + "\nlast line", "x" * 5000,
    "a\n" + "z" * 999 + "\n\nlast\n", "1: literal source prefix\n2: keep this too\n",
])
@pytest.mark.parametrize("budget", [64, 65, 256, 12000])
def test_forward_pages_reconstruct_all_text_without_loss(tmp_path, text, budget):
    path = tmp_path / "range.txt"
    path.write_bytes(text.encode("utf-8"))
    restored, pages = collect(path, budget=budget)
    assert restored == text.replace("\r\n", "\n")
    assert pages[-1]["eof"] and pages[-1]["next_read"] is None


def test_explicit_range_and_mid_line_cursor(tmp_path):
    path = tmp_path / "range.txt"
    path.write_text("before\n" + "🙂abc" * 400 + "\nthird\nafter\n")
    text, pages = collect(path, start_line=2, start_column=3, end_line=3)
    assert text == ("🙂abc" * 400)[2:] + "\nthird\n"
    assert pages[0]["returned_range"]["start"] == {"line": 2, "column": 3}
    assert pages[-1]["returned_range"]["end"] == {"line": 4, "column": 1}
    assert not pages[-1]["truncated"] and not pages[-1]["eof"]


def test_line_window_continuation_respects_original_requested_end(tmp_path):
    path = tmp_path / "many.txt"
    path.write_text("x\n" * 500)
    first = read_text_page(path, display_path=path.name, end_line=450, max_chars=12000)
    assert first["returned_range"]["end"] == {"line": 401, "column": 1}
    assert first["truncation_reason"] == "line_limit"
    assert first["next_read"] == {"tool": "inspect", "path": path.name,
                                  "start_line": 401, "start_column": 1, "end_line": 450}
    text, pages = collect(path, end_line=450, budget=12000)
    assert text == "x\n" * 450 and len(pages) == 2


def test_short_range_and_eof_coordinates_are_truthful(tmp_path):
    path = tmp_path / "small.txt"; path.write_text("header\nvalue\n")
    first = read_text_page(path, display_path=path.name, end_line=1)
    assert first["content"] == "1: header" and not first["truncated"]
    assert not first["eof"] and first["last_line_terminated"]
    empty = read_text_page(path, display_path=path.name, start_line=20)
    assert empty["returned_range"] is None and empty["content"] == "" and empty["eof"]
    at_end = read_text_page(path, display_path=path.name, start_line=2, start_column=6)
    assert source_text(at_end) == "\n"


@pytest.mark.parametrize("kwargs", [
    {"start_line": 0}, {"start_line": True}, {"start_column": -1}, {"start_column": 1.5},
    {"start_line": 2, "end_line": 1}, {"end_line": False}, {"max_chars": 63},
    {"start_column": 5},
])
def test_invalid_indices_fail_explicitly(tmp_path, kwargs):
    path = tmp_path / "small.txt"; path.write_text("abc\nnext\n")
    with pytest.raises(ValueError):
        read_text_page(path, display_path=path.name, **kwargs)


def test_inspect_consumes_native_cursor_and_keeps_path_policy(tmp_path):
    root = workspace(tmp_path)
    text = "MARKER," + "αβγ," * 200 + "\nsecond\n"
    (root / "solution.py").write_text(text)
    coder = ContractBoundedCoder(replace(config(root), max_observation_chars=256, config_sha256=""), None)
    ok, page = coder._apply_tool(root, {"tool": "inspect", "path": "solution.py"}, {})
    assert ok and page["content"].startswith("1: MARKER,")
    pieces = [source_text(page)]
    while page["next_read"]:
        ok, page = coder._apply_tool(root, page["next_read"], {})
        assert ok; pieces.append(source_text(page))
    assert "".join(pieces) == text
    ok, bad = coder._apply_tool(root, {"tool": "inspect", "path": "../escape", "start_column": 1}, {})
    assert not ok and bad.get("error")
    schema = next(item["function"]["parameters"] for item in OpenRouterCoderModel._tool_definitions()
                  if item["function"]["name"] == "inspect")
    assert schema["properties"]["start_column"] == {"type": "integer"}
    assert "start_column" not in schema["required"]


def test_repeated_reads_are_deterministic_and_new_module_is_source_bound(tmp_path):
    path = tmp_path / "input.txt"; path.write_text("line\n" * 300)
    assert read_text_page(path, display_path=path.name) == read_text_page(path, display_path=path.name)
    from swarmrsi import naturebench_session
    from swarmrsi import file_observation
    assert file_observation.__file__
    assert naturebench_session.PACKAGE_ROOT == Path(file_observation.__file__).resolve().parent


@pytest.mark.parametrize("alias_name", ["lib64", "relocated-package"])
def test_source_boundary_accepts_same_package_filesystem_alias(tmp_path, monkeypatch, alias_name):
    from swarmrsi import naturebench_session, file_observation
    original = Path(file_observation.__file__)
    alias = tmp_path / alias_name
    alias.symlink_to(original.parent, target_is_directory=True)
    alternate = alias / original.name
    assert alternate.samefile(original)
    monkeypatch.setattr(file_observation, "__file__", str(alternate))
    assert Path(file_observation.__file__).resolve().parent == naturebench_session.PACKAGE_ROOT
    # Actual configured source map is covered by test_naturebench_entrypoint.
