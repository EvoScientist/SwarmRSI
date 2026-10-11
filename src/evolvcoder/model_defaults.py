"""One place that decides which model everything defaults to.

The backbone used to be written out in nine files -- the session, the two root
builders, both baseline runners -- so changing it meant finding all of them, and
missing one meant a run that quietly used the old model. Deployments that do not
have access to these particular models need a single edit, or no edit at all: both
values are overridable by environment variable.

DEFAULT_BACKBONE is what the search and the baselines run on. REFERENCE_BACKBONE is
what the Claude Code reference traces were taken with; it only ever labels a claim,
and when the two differ the profile's claim scope says so explicitly, because the
`*_beats_claude_code_same_backbone` fields keep their names for compatibility.
"""
from __future__ import annotations

import os

DEFAULT_BACKBONE = os.environ.get("EVOLVCODER_BACKBONE", "deepseek/deepseek-v4.1-flash")
REFERENCE_BACKBONE = os.environ.get("EVOLVCODER_REFERENCE_BACKBONE", "deepseek/deepseek-v4-pro")
DEFAULT_BASE_URL = os.environ.get("EVOLVCODER_BASE_URL", "https://openrouter.ai/api/v1")
