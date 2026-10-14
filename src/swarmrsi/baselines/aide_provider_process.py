"""Hard process deadline around one already-reserved OpenRouter attempt.

This is a transport boundary, not a replacement for AIDE's query construction,
search policy or durable accounting. Install it as AideCallLedger's ``create``;
the ledger records timeout/worker errors as unresolved and never retries them.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from .aide_resources import run_bounded_process
from .aide_transport import AideResourceStop


class AideProviderProcess:
    def __init__(
        self, *, python: Path, helper: Path,
        remaining_seconds: Callable[[], float],
        environment: dict[str, str] | None = None,
    ) -> None:
        # Preserve a virtualenv's invocation path: dereferencing bin/python
        # can switch to the base interpreter and lose the qualified SDK env.
        self.python, self.helper = python.absolute(), helper.resolve()
        if not self.python.is_file() or not self.helper.is_file():
            raise ValueError("AIDE provider runtime/helper is missing")
        self.remaining_seconds = remaining_seconds
        self.environment = dict(os.environ if environment is None else environment)

    def __call__(self, **request: Any) -> Any:
        # Prepare serialization/environment before the late deadline check.
        # Credentials remain in this controller-only child's environment and
        # are never placed in argv, the request body, or error diagnostics.
        if not self.environment.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required by the provider worker")
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False)
        configured = request.get("timeout")
        if (isinstance(configured, bool) or not isinstance(configured, (int, float))
                or not math.isfinite(configured) or configured <= 0):
            raise ValueError("provider worker requires a finite positive timeout")
        remaining = self.remaining_seconds()
        if not math.isfinite(remaining):
            raise ValueError("provider worker requires a finite remaining deadline")
        timeout = min(float(configured), remaining)
        if timeout <= 1.0:
            raise AideResourceStop("insufficient_remaining_provider_wall")
        # SDK timeout remains in the original reserved body; the owned process
        # group enforces a harder outer deadline including a second of teardown.
        result = run_bounded_process(
            [str(self.python), str(self.helper)], input=payload,
            env=self.environment, timeout=timeout, teardown_seconds=1.0,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            # Do not propagate stdout/stderr: SDK messages can contain secrets.
            raise RuntimeError("AIDE provider worker failed; acceptance remains unresolved")
        try:
            receipt = json.loads(result.stdout)
            if (receipt.get("schema_version") != "aide-provider-worker-v1"
                    or not isinstance(receipt.get("response"), dict)):
                raise ValueError("invalid worker receipt")
            # Return raw JSON, not an SDK model. The ledger must persist it
            # before AidePhaseCalls constructs/validates the response object.
            return receipt["response"]
        except Exception:
            raise RuntimeError("AIDE provider worker returned an invalid receipt") from None
