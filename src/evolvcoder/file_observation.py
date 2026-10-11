"""Bounded forward text pages for inspect, with recoverable truncation.

Coordinates are one-based Unicode character positions in decoded text, with
universal newlines normalized to LF. Returned ranges are half-open and exclude
the rendered line-number prefixes. No source file or whole long line is loaded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _index(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_text_page(path: Path, *, display_path: str, start_line: int = 1,
                   start_column: int = 1, end_line: int | None = None,
                   max_chars: int = 12000) -> dict[str, Any]:
    start_line = _index(start_line, "start_line")
    start_column = _index(start_column, "start_column")
    if end_line is not None and _index(end_line, "end_line") < start_line:
        raise ValueError("end_line precedes start_line")
    if type(max_chars) is not int or max_chars < 64:
        raise ValueError("inspect character budget must be at least 64")
    window_end = min(start_line + 399, end_line if end_line is not None else start_line + 399)
    result: dict[str, Any] = {
        "path": display_path, "start_line": start_line, "start_column": start_column,
        "end_line": window_end, "requested_end_line": end_line,
        "coordinate_system": "one_based_unicode_characters_normalized_newlines",
        "returned_range": None, "range_end_exclusive": True,
        "content": "", "last_line_terminated": False,
        "truncated": False, "truncation_reason": None, "next_read": None, "eof": True,
    }
    with path.open("r", encoding="utf-8", errors="replace", newline=None) as stream:
        # Skip in bounded chunks, even if a skipped line is enormous.
        line_number = 1
        while line_number < start_line:
            chunk = stream.readline(8192)
            if not chunk:
                return result
            if chunk.endswith("\n"):
                line_number += 1
        remaining = start_column - 1
        while remaining:
            chunk = stream.readline(min(8192, remaining))
            if not chunk or chunk.endswith("\n"):
                raise ValueError("start_column exceeds the requested source line")
            remaining -= len(chunk)

        column = start_column
        parts: list[str] = []
        rendered_chars = 0
        first = {"line": line_number, "column": column}
        position = dict(first)
        partial = False
        terminated = False
        while line_number <= window_end:
            prefix = f"{line_number}: "
            separator = int(bool(parts))
            available = max_chars - rendered_chars - len(prefix) - separator
            if available <= 0:
                break
            # One lookahead character distinguishes a full line from truncation;
            # a continuation reopens the source at the declared cursor.
            chunk = stream.readline(available + 1)
            if not chunk:
                break
            terminated = chunk.endswith("\n")
            partial = not terminated and len(chunk) > available
            text = chunk[:-1] if terminated else chunk[:available]
            parts.append(prefix + text)
            rendered_chars += separator + len(prefix) + len(text)
            if terminated:
                line_number += 1
                column = 1
            else:
                column += len(text)
            position = {"line": line_number, "column": column}
            if not terminated:
                break  # Partial long line or an unterminated final line.

        # A consumed lookahead still belongs to the next page, even when the
        # file handle now happens to be at EOF.
        more = partial or bool(stream.read(1))
        fulfilled_range = end_line is not None and line_number > end_line
        next_read = None
        if more and not fulfilled_range:
            next_read = {"tool": "inspect", "path": display_path,
                         "start_line": position["line"], "start_column": position["column"]}
            if end_line is not None:
                next_read["end_line"] = end_line
        result.update(
            content="\n".join(parts),
            returned_range={"start": first, "end": position} if parts else None,
            last_line_terminated=terminated if parts else False,
            truncated=next_read is not None,
            truncation_reason=("line_limit" if line_number > window_end else "character_limit") if next_read else None,
            next_read=next_read, eof=not more,
        )
    return result
