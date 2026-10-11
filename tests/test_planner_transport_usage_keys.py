"""The planner transport audit ignores response-envelope usage keys, and nothing else."""
import hashlib
import json

import pytest

from evolvcoder import learning_evidence
from evolvcoder.coder_openrouter import OpenRouterCoderModel, TRANSPORT_USAGE_KEYS

REQUEST = {"planner_payload": {"task": "synthetic"}}
DECODED = {"proposals": [{"id": "p0", "operator": "improve_program"}], "_provider_finish_reason": "tool_calls"}


def parsed_response(extra):
    transport = json.dumps({"model": "m", "messages": [{"role": "system", "content": "s"},
                                                      {"role": "user", "content": json.dumps(REQUEST)}]})
    raw = json.dumps({"content": "synthetic"})
    return {**DECODED, **extra, "_raw_content": raw, "_raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "_transport_request_body_json": transport,
            "_transport_request_body_sha256": hashlib.sha256(transport.encode()).hexdigest()}


@pytest.fixture(autouse=True)
def fixed_decoder(monkeypatch):
    def decode(self, message, version, finish_reason):
        raw = json.dumps(message)
        return {**DECODED, "_raw_content": raw, "_raw_sha256": hashlib.sha256(raw.encode()).hexdigest()}
    monkeypatch.setattr(OpenRouterCoderModel, "_decode_message", decode)


def test_usage_keys_from_the_response_envelope_do_not_fail_the_audit():
    usage = {"_provider": "Together", "_reasoning_tokens": 812, "_cost_usd": 0.0012, "_cached_input_tokens": 4096}
    assert set(usage) == set(TRANSPORT_USAGE_KEYS)
    assert learning_evidence.verify_planner_transport(REQUEST, parsed_response(usage), "m")["request_sha256"]


def test_changed_proposals_or_unknown_keys_still_fail():
    tampered = parsed_response({})
    tampered["proposals"] = [{"id": "p0", "operator": "combine_programs"}]
    with pytest.raises(ValueError, match="planner decoded proposals differ"):
        learning_evidence.verify_planner_transport(REQUEST, tampered, "m")
    with pytest.raises(ValueError, match="planner decoded proposals differ"):
        learning_evidence.verify_planner_transport(REQUEST, parsed_response({"_injected": 1}), "m")
