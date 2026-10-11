#!/usr/bin/env python3
"""Sandbox-side adapter; run the pinned upstream Interpreter exactly once.

Program text arrives on stdin. No key, grader, journal, or scoring attestation
is mounted here. The caller captures this helper's receipt outside the sandbox.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--interpreter-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("positive execution timeout required")
    source = args.upstream_root / "aide/interpreter.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.interpreter_sha256:
        raise ValueError("upstream AIDE Interpreter source mismatch")
    code = sys.stdin.read()
    # Upstream Rich logging writes timeout/cleanup warnings to stdout. Keep
    # stdout exclusively for the one machine-readable receipt, retaining those
    # unmodified diagnostics on stderr instead of stripping arbitrary log lines.
    with redirect_stdout(sys.stderr):
        sys.path.insert(0, str(args.upstream_root))
        from aide.interpreter import Interpreter

        interpreter = Interpreter(
            Path.cwd(), timeout=args.timeout_seconds,
            agent_file_name="runfile.py", format_tb_ipython=False,
        )
        try:
            result = interpreter.run(code, reset_session=True)
        finally:
            interpreter.cleanup_session()
    receipt = {
        "schema_version": "aide-original-interpreter-receipt-v1",
        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "interpreter_sha256": args.interpreter_sha256,
        "reset_session": True,
        "execution_result": asdict(result),
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
