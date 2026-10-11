#!/usr/bin/env python3
"""Controller-only worker: one SDK attempt, with no SDK retries.

Unlike run_aide_interpreter_once.py, this worker is NOT candidate-facing and
must never be mounted in the candidate sandbox. The parent owns the hard
deadline and durable attempt ledger. Request JSON arrives on stdin; credentials
arrive only via environment. No task files, generated code or grader are read.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import sys


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, dict) or request.get("stream"):
            raise ValueError("non-streaming Chat Completions request required")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("missing OpenRouter credential")
        # No SDK logging or exception text can contaminate the receipt or leak
        # credentials into a retained stderr log. Only a type label escapes.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            from openai import OpenAI

            with OpenAI(
                base_url=os.environ.get("AIDE_PROVIDER_BASE_URL", "https://openrouter.ai/api/v1"), api_key=key,
                max_retries=0,
            ) as client:
                response = client.chat.completions.create(**request)
                raw = response.model_dump(mode="json")
        encoded = json.dumps({"schema_version": "aide-provider-worker-v1",
                              "response": raw}, ensure_ascii=False, allow_nan=False)
    except BaseException as exc:
        print(json.dumps({"schema_version": "aide-provider-worker-v1",
                          "error_type": type(exc).__name__}))
        return 1
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
