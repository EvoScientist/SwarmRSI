"""Receipt accounting is distinct from successful response/token accounting."""
import json

import pytest

from swarmrsi.contracts import canonical_sha256
from swarmrsi.naturebench_session import atomic_json
from swarmrsi.session_runtime import safe_transport_usage, transport_snapshot, verify_transport_snapshot


def attempt(root, outcome=None):
    request = {"synthetic": True}
    atomic_json(root / "batch_000000.request.json", request)
    base = {"schema_version": "swarmrsi-planner-transport-attempt-v1",
            "batch": 0, "attempt": 0, "request_sha256": canonical_sha256(request), "timestamp_unix": 1000}
    atomic_json(root / "transport_attempts/batch_000000.attempt_000.started.json",
                {**base, "provider_cost_status": "unknown"})
    if outcome is not None:
        atomic_json(root / "transport_attempts/batch_000000.attempt_000.outcome.json", {**base, **outcome})


def test_started_only_is_unknown_not_a_free_call(tmp_path):
    attempt(tmp_path)
    receipt = transport_snapshot(tmp_path, 0)
    assert receipt["transport_attempts"] == receipt["unresolved_cost_attempts"] == 1
    assert receipt["known_cost_usd_subtotal"] == receipt["responses_received"] == 0
    assert receipt["pending_request_batches"] == [0]
    verify_transport_snapshot(tmp_path, 0, receipt)
    (tmp_path / "transport_attempts/extra.json").write_text('{}')
    with pytest.raises(ValueError, match="unexpected"):
        verify_transport_snapshot(tmp_path, 0, receipt)


@pytest.mark.parametrize("usage,total,unknown", [({"cost": 7}, 0, 1),
    ({"cost_usd": .2, "cost": 7}, .2, 0), ({"input_tokens": 4}, 0, 1)])
def test_only_explicit_usd_enters_subtotal(tmp_path, usage, total, unknown):
    attempt(tmp_path, {"status": "response_received", "usage": safe_transport_usage(usage),
                       "provider_cost_status": "reported" if "cost" in usage or "cost_usd" in usage else "unknown"})
    atomic_json(tmp_path / "batch_000000.response.json", {"response": {}, "usage": usage})
    atomic_json(tmp_path / "batch_000000.record.json", {"synthetic": True})
    receipt = transport_snapshot(tmp_path, 1)
    assert receipt["known_cost_usd_subtotal"] == total
    assert receipt["unresolved_cost_attempts"] == unknown
    assert receipt["pending_request_batches"] == []
    altered = json.loads((tmp_path / "batch_000000.response.json").read_text())
    altered["usage"]["cost_usd"] = 9
    atomic_json(tmp_path / "batch_000000.response.json", altered)
    with pytest.raises(ValueError, match="usage mismatch"):
        transport_snapshot(tmp_path, 1)
