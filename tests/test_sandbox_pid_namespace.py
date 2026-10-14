import shutil
from pathlib import Path

import pytest

from swarmrsi import naturebench_session as session


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap not installed")
def test_candidate_sandbox_gets_its_own_pid_namespace(tmp_path, monkeypatch):
    (tmp_path / "problem/data").mkdir(parents=True)
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    wrapper = session.coder_sandbox_config(tmp_path)["wrapper"]
    # Without --unshare-pid the candidate could enumerate the controller's /proc/<pid>/fd.
    assert wrapper[1:4] == ("--die-with-parent", "--unshare-net", "--unshare-pid")


def test_pid_flag_is_in_the_source(tmp_path):
    source = Path(session.__file__).read_text()
    assert '"--die-with-parent", "--unshare-net", "--unshare-pid"]' in source
