#!/usr/bin/env python3
"""Offline, source-bound audit label import and deterministic aggregation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolvcoder.project_audit import export_project_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = export_project_audit(args.bundle, evidence_root=args.evidence_root, output=args.output)
    print(json.dumps({"summary_sha256": summary["summary_sha256"],
                      "projects": len(summary["projects"]), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
