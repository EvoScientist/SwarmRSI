"""Installed utility entrypoint. Only implemented operations are advertised."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .configuration import ConfigurationError, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evolvcoder", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evolvcoder {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "preflight", "status"):
        operation = commands.add_parser(name, help=f"{name} the configured benchmark/method through the public SDK")
        operation.add_argument("--config", type=Path, required=True)
        operation.add_argument("--run-id", help="stable output identity; default is derived from resolved configuration")
    check = commands.add_parser("check-config", help="validate JSON/TOML configuration without model calls")
    check.add_argument("config", type=Path)
    analysis = commands.add_parser("analyze-learning", help="export source-verified RSI/Wisdom journals without executing experiments")
    analysis.add_argument("--chain-root", type=Path, help="RSI chain, optional for standalone runner/run analysis")
    analysis.add_argument("--runner-root", type=Path, action="append", default=[])
    analysis.add_argument("--wisdom-snapshot", type=Path)
    analysis.add_argument("--run-evidence", type=Path, action="append", default=[])
    analysis.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit", help="import and summarize source-bound project audit labels")
    audit.add_argument("--bundle", type=Path, required=True)
    audit.add_argument("--evidence-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    naturebench = commands.add_parser("run-naturebench", help="run the full installed NatureBench session with explicit local inputs")
    from .naturebench_cli import add_arguments, execute
    add_arguments(naturebench)
    report = commands.add_parser("report", help="uniform delivery summary across runs and methods; read-only")
    report.add_argument("--run", type=Path, action="append", required=True, dest="runs",
                        help="a run output directory; repeat for a campaign table with a fixed denominator")
    report.add_argument("--plan-id", help="optional identifier recorded in the report")
    report.add_argument("--output", type=Path, help="write the report to a new file instead of stdout")
    learning = commands.add_parser("learn", help="prepare, validate, run or inspect the complete RSI/Wisdom workflow")
    from . import learning_cli
    learning_cli.add_arguments(learning)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze-learning":
            from .learning_analysis import export_learning_analysis
            manifest = export_learning_analysis(args.output, chain_root=args.chain_root, runner_roots=args.runner_root,
                wisdom_snapshot=args.wisdom_snapshot, run_evidence=args.run_evidence)
            print(json.dumps({"output": str(args.output.resolve()), "analysis_sha256": manifest["analysis_sha256"],
                "observed_round_rows": manifest["observed_round_rows"], "zero_model_calls": True}))
            return 0
        if args.command == "report":
            from .run_report import campaign_report, write_campaign_report
            if args.output is None:
                print(json.dumps(campaign_report(args.runs, plan_id=args.plan_id), indent=2, sort_keys=True))
            else:
                result = write_campaign_report(args.output, args.runs, plan_id=args.plan_id)
                print(json.dumps({"output": str(args.output.resolve()), "report_sha256": result["report_sha256"],
                                  "denominator": result["denominator"], "new_program_wins": result["new_program_wins"],
                                  "zero_model_calls": True}))
            return 0
        if args.command in {"run", "preflight", "status"}:
            from .api import Application
            from .unified_run import context_for, result_dict
            context = context_for(args.config, args.run_id)
            result = getattr(Application(), args.command)(context)
            payload = result_dict(result) if args.command == "run" else result
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            if args.command == "run":
                return 0 if result.status in {"completed", "retained"} else 3 if result.status == "pending" else 1
            return 1 if args.command == "preflight" and not result.get("ready") else 0
        if args.command == "learn":
            return learning_cli.execute(args)
        if args.command == "run-naturebench":
            return execute(args)
        if args.command == "check-config":
            config = load_config(args.config)
            print(json.dumps({"configuration_valid": True, "runtime_ready": None,
                              "note": "Schema validation only; runtime, data and adapter readiness are separate.",
                              "config": config.to_dict()}, ensure_ascii=False, indent=2))
        elif args.command == "audit":
            from .project_audit import export_project_audit
            summary = export_project_audit(args.bundle, evidence_root=args.evidence_root, output=args.output)
            print(json.dumps({"summary_sha256": summary["summary_sha256"], "projects": len(summary["projects"]),
                              "output": str(args.output)}, ensure_ascii=False))
    except learning_cli.ValidationPending as exc:
        print(json.dumps({"status": "pending", "reason": str(exc)}))
        return 3
    except (ConfigurationError, ValueError, KeyError, OSError) as exc:
        print(f"evolvcoder: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
