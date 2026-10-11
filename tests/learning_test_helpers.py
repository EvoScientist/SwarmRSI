"""Synthetic fixtures migrated from the original planner/tool tests."""
from evolvcoder.naturebench_session import Planner
from evolvcoder.coder_openrouter import OpenRouterCoderModel

class EmptyModel:
    def __init__(self):
        self.requests = []

    def respond(self, request):
        self.requests.append(request)
        return {"proposals": []}, {"model_calls": 1, "input_tokens": 3, "output_tokens": 5}

def planner(seed=7):
    value = Planner.__new__(Planner)
    value.model, value.seed, value.batch = EmptyModel(), seed, 0
    value.required_symbols = ("main",)
    value.preserved_source_patterns = ("DATA_DIR",)
    value.task_context = "Public synthetic task only."
    value.available_import_roots = ("numpy",)
    value.unavailable_import_roots = ()
    return value

def raw(index=0, operator='improve_program', diagnostic='', family='linear'):
    return {'slot_index': index, 'operator': operator, 'diagnostic_source_action_id': diagnostic,
            'hypothesis': 'SYNTHETIC offline proposal, not reconstructed historical prose',
            'description': 'Retain the assigned executable parent.',
            'family': {'input_transform': 'raw', 'primary_operator': family,
                       'output_construction': 'direct', 'objective_or_test': 'public metric',
                       'capacity_controls': []},
            'predictions': dict.fromkeys(['quality', 'information_gain', 'reuse', 'validity', 'cost'], .5)}

def configure(loop, rows, *, audit_root=None):
    value = planner()
    value.audit_root = audit_root
    calls = []
    def respond(request):
        calls.append(request)
        return {'proposals': rows}, {'input_tokens': 3, 'output_tokens': 5}
    value.model.respond = respond
    loop.proposer = value
    return value, calls

def client(native=True):
    return OpenRouterCoderModel(api_key="offline-only", model="test/model", base_url="https://example.invalid",
                                max_tokens_per_turn=100, native_tools=native, parallel_tool_calls=False)
