

def test_accelerator_section_states_what_the_candidate_can_see(monkeypatch):
    """The prompt listed torch as importable and never said a GPU existed.

    Across eighteen cells not one winning program imported torch, on tasks whose own
    metadata asks for a GPU, while a card sat idle in every cell. The inventory is
    measured in the candidate runtime rather than assumed, so a CPU-only deployment
    still gets told the truth.
    """
    import swarmrsi.naturebench_session as ns

    class Probe:
        def __init__(self, out, rc=0): self.stdout, self.returncode = out, rc

    monkeypatch.setattr(ns, "runtime_probe", lambda script: Probe(
        '{"cuda": true, "devices": 1, "name": "NVIDIA H200", '
        '"total_memory_bytes": 153391689728, "torch_version": "2.8.0+cu128"}'))
    seen = ns.accelerator_inventory()
    assert seen["cuda"] is True and seen["devices"] == 1

    monkeypatch.setattr(ns, "runtime_probe", lambda script: Probe('{"cuda": false, "devices": 0}'))
    assert ns.accelerator_inventory()["cuda"] is False

    monkeypatch.setattr(ns, "runtime_probe", lambda script: Probe("not json"))
    assert ns.accelerator_inventory() == {"cuda": False, "devices": 0, "error": "unparseable_probe_output"}

    monkeypatch.setattr(ns, "runtime_probe", lambda script: Probe("", rc=1))
    assert ns.accelerator_inventory() == {"cuda": False, "devices": 0, "error": "probe_failed"}
