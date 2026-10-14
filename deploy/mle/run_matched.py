"""Assemble one MLE-Bench Lite run: task package, BRS profile, config, launcher.

Three things this deployment has to get right, and each is a decision rather than a
default.

The default has no score-triggered early stop. MLE-Bench publishes gold/silver/bronze
cut-offs on its private official test set, while search observes a carved public-train
split. Historical MLE-Lite candidates show that these quantities can differ in scale
and reverse candidate order, so an official threshold cannot safely stop public search.
Gold and MLEvolve targets remain explicit diagnostic overrides; the frozen envelope is
the default stopping rule.

The candidate envelope matches MLEvolve's, not ours. The earlier insults comparison
gave our candidates two cores and no GPU against MLEvolve's sixteen cores, 64 GB and
one card, so the two numbers were never comparable. The defaults here are MLEvolve's.

The wall budget deliberately does NOT match MLEvolve's twelve hours per task. Twenty
one tasks at twelve hours over two cards is more than a week, and the point of the
panel is a medal rate, not a wall-clock tie. The budget is stated in the profile's
claim scope so the comparison is reported as what it is: a lower budget than the
reference's, which makes a matched or better medal rate a stronger claim and a worse
one uninterpretable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from swarmrsi.model_defaults import DEFAULT_BACKBONE, DEFAULT_BASE_URL  # noqa: E402

DONOR = "config/naturebench/profile.json"

# Competitions whose examples live in image directories rather than in train.csv. The
# carve reads train.csv alone, so a package for one of these would hand the candidate
# labels with no pixels. They are refused until the public tree is mounted as well.
# Competitions whose payload the package has to carry a copy of, with the approximate
# size of that copy. They are not refused any more -- carve_split copies the payload in
# -- but a package is refused when its copy would not fit the disk budget, because this
# local disk is shared with another user's runs.
# Competitions with no training table at all, where the label is the file itself. The
# carve splits rows, so it has nothing to split and would seal an empty package.
# dogs-vs-cats is no longer here: make_task_root transcribes its filename labels into a
# real train table before the carve, so the package builds like any other.
# Birds uses the native record/multi-label adapter, not a synthetic flat table.
NO_TRAINING_TABLE = {
    "the-icml-2013-whale-challenge-right-whale-redux": "the label is the training file's directory",
}

LARGE_PAYLOAD_GB = {
    "ranzcr-clip-catheter-line-classification": 13.0,
    "aptos2019-blindness-detection": 10.0,
    "histopathologic-cancer-detection": 7.0,
    "dogs-vs-cats-redux-kernels-edition": 0.9,
    "dog-breed-identification": 0.7,
    "the-icml-2013-whale-challenge-right-whale-redux": 0.8,
    "mlsp-2013-birds": 0.7,  # Live prepared inventory is approximately 0.66 decimal GB.
    "plant-pathology-2020-fgvc7": 0.8,
}


def load_envelope(path: Path) -> dict:
    """A benchmark's budget, stated once and declaratively.

    A run's envelope has to come from whatever its benchmark publishes as the reference,
    and that differs per dataset: NatureBench hard-matches a Claude Code trace item by
    item, while MLE-Bench publishes a wall clock and nothing else. Inheriting one shape
    for the other is how a per-action wall of 3600 and an evaluation timeout of 1800 --
    both measured against NatureBench's small, fast tasks -- ended up governing image
    competitions they cannot fit.

    So each dataset gets a file under deploy/envelopes/ recording its reference, what we
    actually run, and every deviation between the two. Adding a dataset means writing
    that file, not editing this one.
    """
    envelope = json.loads(path.read_text())
    require = ("schema_version", "benchmark", "reference", "ours")
    missing = [k for k in require if k not in envelope]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    if envelope["schema_version"] != "swarmrsi-benchmark-envelope-v1":
        raise SystemExit(f"{path}: unknown envelope schema {envelope['schema_version']!r}")
    return envelope


def mlevolve_target(reports_root: Path, competition: str, lower_is_better: bool):
    """The score to beat: the best of MLEvolve's three submitted seeds for this task.

    Its grading reports are the leaderboard's own submission records -- recomputing the
    Lite medal rate from them reproduces the published 80.3% exactly -- so this is the
    number the comparison is actually about. It is a far better stopping target than the
    gold threshold on the tasks where MLEvolve itself does not medal: dog-breed's gold
    line is 0.00050 while MLEvolve managed 0.34346, and nyc-taxi's is 2.83377 while they
    managed 4.24585. Stopping at a medal line those tasks cannot reach means never
    stopping, and running the full wall for nothing.

    Caveat carried into claim_scope: their number is an official-test score, while the
    search compares against the carved split, and the two diverge in both directions
    across this campaign. Beating it on the carved split is therefore evidence, not
    proof, that the official grade will beat it too.
    """
    scores = []
    for path in sorted(reports_root.glob("mlevolve_group*/grading_report_group_*.json")):
        for row in json.loads(path.read_text()).get("competition_reports", []):
            if row.get("competition_id") == competition and row.get("score") is not None:
                scores.append(float(row["score"]))
    if not scores:
        return None
    return min(scores) if lower_is_better else max(scores)


def build_profile(donor: Path, *, competition: str, metadata: dict, model: str,
                  limits: dict, starter: str, budget_note: str, search_target, target_name: str,
                  reference_envelope: bool, envelope: dict | None = None,
                  forced_class: str | None = None, overrides: dict | None = None) -> dict:
    overrides = overrides or {}
    profile = json.loads(donor.read_text())
    # Every key but `tasks` is inherited. A hand-written profile lost sixteen of them
    # once -- including native_planner_tools, without which the planner sends no tool
    # schema and every proposal comes back as prose and is rejected -- so the panel
    # profile is the shape donor and only the task table is replaced.
    profile["tasks"] = {competition: {
        "starter": starter,
        "required_symbols": ["main"],
        "preserved_source_patterns": ["DATA_DIR", "OUTPUT_DIR"],
        # The MLE adapter refuses a profile carrying claude_code_score: the threshold
        # here is the competition's own leaderboard, not another agent's transcript.
        "search_metric": metadata["search_metric"],
        "search_target": search_target,
        "limits": limits,
    }}
    profile["model"] = model
    # The planner reasons less than the coder: measured at ~3900 output tokens for a
    # ~470 token proposal, which on this panel truncated aerial-cactus's batches until
    # the campaign gave up with planner_truncated_proposal_batches. Same settings the
    # NatureBench bundles ship.
    profile["planner_reasoning"] = {"effort": "medium", "exclude": True}
    screening = profile.setdefault("screening_execution_limits", {})
    # 10000 was still too small and it cost a whole run. nyc-taxi's batches 2 and 3 both
    # came back with provider_finish_reason "length" at exactly 10000 output tokens, and
    # two truncations in a row end the campaign: the escape hatch lowers the planner's
    # reasoning one step and gives up at the bottom, and from "medium" there is only one
    # step. The run stopped at 6658 of 40500 wall seconds with one scored candidate.
    # The line accepts far more than this -- 131072 was accepted on probe -- and tokens
    # are billed as generated, so a ceiling this far above the observed ~4000 costs
    # nothing until it is needed.
    screening["planner_output_tokens_per_batch"] = 24000
    if envelope is not None:
        ours = envelope["ours"]
        timeout = ours.get("candidate_execution_timeout_seconds")
        if timeout:
            screening["full_evaluation_timeout_seconds"] = timeout
            screening["execution_validation_timeout_seconds"] = timeout
        for key, value in (ours.get("per_action_floor") or {}).items():
            if key != "note":
                screening[key] = value
    if reference_envelope:
        # Required whenever the campaign leaves a count unlimited, and sized not to
        # bind: an action may spend what it needs, the wall decides when to stop.
        screening["max_output_tokens_per_action"] = 4_000_000
        screening["max_input_tokens_per_action_safety"] = 40_000_000
        # Two hours, against an evaluation timeout of 1800s, so the first action can run
        # several candidates before any measurement exists to size later ones. 3600 left
        # roughly 1770s of working time once one evaluation was reserved, which is what
        # ended both dog-breed actions with no scored program.
        screening["wall_time_seconds_per_action"] = 7200
    if envelope is not None:
        # Applied last: the envelope is the statement of record, and the reference-envelope
        # defaults above are only what applies when a dataset has not stated its own.
        ours = envelope["ours"]
        for key, value in (ours.get("per_action_counts") or {}).items():
            if key != "note":
                screening[key] = value
        per_action = ours.get("per_action_wall_seconds")
        if per_action is None:
            screening.pop("wall_time_seconds_per_action", None)
        else:
            screening["wall_time_seconds_per_action"] = per_action
        # A task's class, where the dataset declares one. Loosening the per-action budget
        # is justified by the tasks whose cells exhaust it; a task that never reaches it
        # gains nothing and loses the guard against a runaway cell, so the classes are
        # applied per task rather than across the dataset.
        classes = ours.get("per_task_classes") or {}
        for name, spec in classes.items():
            if not isinstance(spec, dict):
                continue
            chosen = forced_class or next(
                (n for n, sp in classes.items()
                 if isinstance(sp, dict) and competition in (sp.get("members") or [])), None)
            if name != chosen:
                continue
            for key, value in spec.items():
                if key in ("members", "evidence", "policy"):
                    continue
                if key == "candidate_execution_timeout_seconds":
                    screening["full_evaluation_timeout_seconds"] = value
                    screening["execution_validation_timeout_seconds"] = value
                else:
                    screening[key] = value
            print(f"task class: {competition} is {name}"
                  + (" (forced)" if forced_class else ""), file=sys.stderr)
            break
    # The command line is last, so a one-off tweak needs no edit to the envelope.
    for flag, key in (("model_calls_per_action", "model_calls_per_action"),
                      ("tool_calls_per_action", "tool_calls_per_action"),
                      ("max_repairs_per_action", "max_repairs_per_action"),
                      ("per_action_wall_seconds", "wall_time_seconds_per_action")):
        value = overrides.get(flag)
        if value is not None:
            screening[key] = value
    if overrides.get("eval_timeout_seconds") is not None:
        screening["full_evaluation_timeout_seconds"] = overrides["eval_timeout_seconds"]
        screening["execution_validation_timeout_seconds"] = overrides["eval_timeout_seconds"]
    if overrides:
        named = {k: v for k, v in overrides.items() if v is not None and k != "eval_timeout_seconds"}
        if named or overrides.get("eval_timeout_seconds") is not None:
            print(f"command-line overrides: {[k for k, v in overrides.items() if v is not None]}",
                  file=sys.stderr)
    # An execution timeout the action wall cannot honour is not a generous budget, it is
    # a silent tax. naturebench_session reserves min(full_evaluation_timeout + 30,
    # per_action_wall * 0.55) of each action for the scored evaluation and hands the
    # coder the rest, so once the timeout exceeds 0.55 of the wall the 0.55 term binds
    # permanently and the coder gets 45% of every action no matter how the timeout is
    # set. nyc-taxi ran its whole campaign that way -- 32400 against a 5400 wall -- and
    # its coder never saw more than 2430 seconds of an action.
    wall_per_action = screening.get("wall_time_seconds_per_action")
    timeout = screening.get("full_evaluation_timeout_seconds")
    if wall_per_action and timeout and timeout + 30 > wall_per_action * 0.55:
        raise SystemExit(
            f"candidate execution timeout {timeout}s cannot fit inside a "
            f"{wall_per_action}s action: the evaluation reserve would take "
            f"{wall_per_action * 0.55:.0f}s and cap the coder at 45% of every action. "
            f"Raise --per-action-wall-seconds above {(timeout + 30) / 0.55:.0f} or lower "
            f"--eval-timeout-seconds below {wall_per_action * 0.55 - 30:.0f}.")
    profile["schema_version"] = "swarmrsi-mlebench-brs-panel-v1"
    profile["selection_rule"] = (
        "MLE-Bench Lite split as published; siim-isic-melanoma-classification excluded "
        "for size. No score-based task selection.")
    profile["claim_scope"] = (
        f"backbone {model}; MLE-Bench Lite, seed {profile.get('seed')}. The stopping "
        f"criterion is {target_name} ({search_target}). Candidate resources match MLEvolve's "
        f"reference configuration. {budget_note} Scores reported against MLEvolve's "
        "published Lite medal rate are cross-configuration, not a same-route rerun.")
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("competition")
    parser.add_argument("--root", type=Path, required=True, help="campaign root on the host")
    # mlebench derives its data dir from the prepared tree as prepared.parent.parent,
    # so the archive has to be nested its way: <root>/<competition>/prepared/{public,
    # private}. A flat prepared/<competition> made every official grade fail with
    # "Dataset for competition ... is not prepared".
    parser.add_argument("--prepared-root", type=Path, default=Path("data/mlebench/data"))
    parser.add_argument("--mlebench-data-dir", type=Path, default=Path("data/mlebench/registry"),
                        help="data dir handed to the mlebench registry for leaderboards and graders")
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--donor", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_BACKBONE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--secret", default=".secrets/openrouter.json")
    parser.add_argument("--image", default="swarmrsi-mle:h200")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--holdout", type=float, default=0.2)
    # MLEvolve's reference configuration for a candidate trial, as the MLE-Bench
    # leaderboard's own runs/README.md records it: "Gemini-3-Pro-preview, 12 hours,
    # 21 vCPUs, 234GB of RAM, and 1 H200 GPU". The earlier 16 CPU / 64 GB here was
    # estimated from secondary material and understated their memory by 3.7x. The GPU
    # matches exactly -- this host is also an H200 -- so hardware is no longer a
    # difference between us and the reference.
    parser.add_argument("--cpus", type=int, default=21)
    parser.add_argument("--memory-mb", type=int, default=239616)
    parser.add_argument("--gpus", type=int, default=1)
    # Total wall = search envelope + grading reserve. Raising the reserve alone would
    # have shrunk the SEARCH envelope from 13500s to 11700s, which is the budget every
    # run in this campaign shares -- a rerun would then have silently been a different
    # configuration from the eleven runs it has to be comparable with. The reserve sits
    # outside the search budget instead, so 16200 - 2700 leaves the same 13500.
    parser.add_argument("--wall-seconds", type=int, default=16200)
    parser.add_argument("--terminal-reserve-seconds", type=int, default=2700,
                        help="wall reserved for the final re-fit and official grade, outside the search envelope")
    # Evaluations are the budget that should bind, because that is the budget the
    # reference states. MLEvolve's published configuration is a wall clock and a
    # candidate count, with no token ceiling; the 400k output cap inherited here from an
    # older single-task config bound first in every run and stopped the search after two
    # to six of sixteen evaluations, with a quarter of the wall and a third of the
    # tokens' worth of calls unused. Raised until evaluations or wall bind instead.
    parser.add_argument("--evaluations", type=int, default=16)
    parser.add_argument("--model-calls", type=int, default=600)
    # The PER-ACTION ceiling, not the campaign one. Measured over 143 cells of the first
    # pass: 18% ended call_or_tool_budget_exhausted having produced no runnable program
    # at all, concentrated exactly on the tasks with the largest data -- dog-breed 75%,
    # nyc-taxi 75%, tabular-may 50%, jigsaw 33% -- while every small task was at 0%.
    # Those tasks need longer programs than sixteen calls can write, and nyc-taxi's
    # winner fell back to hand-written numpy on a 55M-row regression because of it.
    # These default to None so that a flag actually given on the command line can be told
    # apart from one that merely has a default, and can therefore win over the envelope.
    # Precedence is: command line, then the task's class in the envelope, then the
    # envelope's dataset-wide values.
    parser.add_argument("--model-calls-per-action", type=int, default=None)
    parser.add_argument("--tool-calls-per-action", type=int, default=None)
    parser.add_argument("--max-repairs-per-action", type=int, default=None)
    parser.add_argument("--per-action-wall-seconds", type=int, default=None)
    parser.add_argument("--eval-timeout-seconds", type=int, default=None,
                        help="candidate execution timeout")
    parser.add_argument("--task-class", choices=("fast", "slow"), default=None,
                        help="force a class instead of the one the envelope records")
    parser.add_argument("--output-tokens", type=int, default=1200000)
    parser.add_argument("--input-tokens", type=int, default=20000000)
    parser.add_argument("--python", default="/opt/conda/envs/agent/bin/python")
    parser.add_argument("--max-payload-gb", type=float, default=2.0,
                        help="refuse a competition whose data copy would exceed this")
    parser.add_argument("--envelope", type=Path, default=None,
                        help="benchmark envelope file; defaults to deploy/envelopes/<benchmark>.json. "
                             "Supplying one overrides the budget flags below with its `ours` block.")
    parser.add_argument("--target", choices=("none", "gold", "mlevolve"), default="none",
                        help="optional public-search early stop; 'none' is the safe MLE default because "
                             "official-test thresholds are not calibrated to the carved public split")
    parser.add_argument("--mlevolve-reports", type=Path,
                        default=Path("data/mlebench/reference_reports"))
    parser.add_argument("--model-cache", type=Path, default=None,
                        help="read-only directory of pretrained weights to bind at /model_cache; "
                             "without one the sandbox has the libraries but no weights and cannot "
                             "fetch any, since it runs with no network")
    parser.add_argument("--reference-envelope", action="store_true",
                        help="budget the run the way the leaderboard states it: a wall clock and "
                             "nothing else, with per-action allowances derived from what actions "
                             "actually cost rather than fixed in advance")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--coder-finalization-reserve-seconds", type=float, default=None,
                        help="Opt-in: reserve action wall for controller validation, not extra compute")
    parser.add_argument("--internal-validation-policy", choices=("legacy", "controller_only"),
                        default="legacy", help="Opt-in qualification: candidate self-reports are diagnostic only")
    parser.add_argument("--submission-fit-policy", choices=("legacy", "full_refit"), default="legacy",
                        help="Opt-in: tell the coder the terminal re-execution must fit the submission "
                             "model on every training row, and flag that execution in its environment")
    parser.add_argument("--carve-folds", type=int, default=1,
                        help="Opt-in repeated carve: score every search candidate on K public hold-out "
                             "splits and select on the mean (table competitions only)")
    args = parser.parse_args()
    if args.carve_folds < 1:
        raise SystemExit("--carve-folds must be at least 1")

    envelope = None
    envelope_path = args.envelope or (Path(__file__).resolve().parents[1] / "envelopes/mlebench.json")
    if envelope_path.is_file():
        envelope = load_envelope(envelope_path)
        ours = envelope["ours"]
        args.wall_seconds = ours["wall_seconds"]
        args.terminal_reserve_seconds = ours["terminal_reserve_seconds"]
        hardware = ours.get("hardware") or {}
        args.cpus = hardware.get("cpus", args.cpus)
        args.memory_mb = hardware.get("memory_mb", args.memory_mb)
        args.gpus = hardware.get("gpus", args.gpus)
        args.target = ours.get("target_source", args.target)
        counts = ours.get("counts") or {}
        # An envelope that leaves the counts unlimited is the reference-envelope mode:
        # the wall decides when to stop and the allocator sizes actions from measurement.
        args.reference_envelope = all(counts.get(k) is None for k in
                                      ("model_calls", "tool_calls", "evaluations", "output_tokens"))
        print(f"envelope {envelope_path.name}: reference {envelope['reference']['name']}, "
              f"wall {args.wall_seconds}s, counts "
              f"{'unlimited' if args.reference_envelope else 'capped'}", file=sys.stderr)

    if args.competition in NO_TRAINING_TABLE:
        raise SystemExit(
            f"{args.competition} carries no training table: {NO_TRAINING_TABLE[args.competition]}. "
            "The carve splits rows of a table, so this competition needs its own package")

    expected_gb = LARGE_PAYLOAD_GB.get(args.competition, 0.0)
    if expected_gb > args.max_payload_gb:
        raise SystemExit(
            f"{args.competition} would copy about {expected_gb} GB into its package, over the "
            f"{args.max_payload_gb} GB budget; raise --max-payload-gb once there is room")

    donor = args.donor or (args.source / DONOR)
    root, tasks_root = args.root.resolve(), args.root.resolve() / "tasks"
    for name in ("configs", "runs", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)

    package = tasks_root / args.competition
    if not package.exists() or args.force:
        subprocess.run([sys.executable, str(Path(__file__).with_name("make_task_root.py")),
                        args.competition, "--cache", str(args.mlebench_data_dir),
                        "--prepared", str(args.prepared_root / args.competition / "prepared"),
                        "--tasks-root", str(tasks_root), "--official-source", str(args.official_source),
                        "--seed", str(args.seed), "--holdout", str(args.holdout),
                        "--python", args.python, *(["--force"] if args.force else []),
                        *(["--carve-folds", str(args.carve_folds)] if args.carve_folds > 1 else [])],
                       check=True)
    metadata = json.loads((package / "metadata.json").read_text())
    built_folds = int(metadata["public_split"].get("carve_folds", 1))
    if built_folds != args.carve_folds:
        raise SystemExit(f"{package} was built with {built_folds} carve fold(s), not {args.carve_folds}; "
                         "rebuild it with --force")
    gold = metadata["official_thresholds"]["gold_threshold"]
    lower = metadata["search_metric"]["direction"] == "minimize"
    target, target_name = None, "no early-stop threshold"
    if args.target == "gold":
        target, target_name = gold, "the competition's published gold threshold"
    elif args.target == "mlevolve":
        target = mlevolve_target(args.mlevolve_reports, args.competition, lower)
        if target is None:
            raise SystemExit(f"{args.competition} has no MLEvolve submission on record; "
                             "use --target gold or point --mlevolve-reports at the reports")
        target_name = "the best of MLEvolve's three submitted seeds"
    if target is None and args.target != "none":
        raise SystemExit(f"{args.competition} has no {args.target} target; it cannot stop early")

    # The terminal grade needs its own reservation inside the wall, and the inner
    # watchdog must fit within what is left: the constraint is
    # trial_timeout <= wall - terminal_reserve, and violating it is what killed r5.
    #
    # 900 seconds was not enough. The terminal step re-fits the winning program on the
    # FULL official training set, which is larger than the carved split the search
    # trained on, and then grades it. jigsaw's search ran to 12472s of its 13500s and
    # the re-fit on 159k rows overran what was left: the run ended
    # `terminal process exceeded shared deadline` with four scored candidates and no
    # official score at all. A run that cannot be graded is worth nothing, so the
    # reservation is now large enough to re-fit a big competition.
    terminal_reserve = args.terminal_reserve_seconds
    trial_timeout = args.wall_seconds - terminal_reserve

    if args.reference_envelope:
        # The leaderboard states MLEvolve's budget as "12 hours" and nothing else: no
        # count of model calls, tools, evaluations or tokens. Declaring those counts
        # anyway is what produced three rounds of whack-a-mole here -- output tokens
        # bound first, then model calls, then the per-action wall -- each raise merely
        # moving which arbitrary number stopped the search. Left null they reach the
        # allocator as infinity, and it sizes an action from what actions have actually
        # cost. A per-action wall ceiling stays, because before any action has run there
        # is no measurement to size the first one from.
        limits = {"backbone_message_ids": None, "tool_calls": None,
                  "score_bearing_evaluations": None,
                  "wall_time_seconds": float(trial_timeout),
                  "tokens": {"input": None, "cache_read_input": 0,
                             "context_read": None, "output": None}}
    else:
        limits = {"backbone_message_ids": args.model_calls, "tool_calls": args.model_calls * 2,
                  "score_bearing_evaluations": args.evaluations,
                  "wall_time_seconds": float(trial_timeout),
                  "tokens": {"input": args.input_tokens, "cache_read_input": 0,
                             "context_read": args.input_tokens, "output": args.output_tokens}}
    budget_note = (f"Wall budget is {args.wall_seconds / 3600:.1f} h per task, BELOW the "
                   "reference's 12 h; a matched or better medal rate under this budget is "
                   "a stronger claim, a worse one is not interpretable as a mechanism result.")
    profile = build_profile(donor, competition=args.competition, metadata=metadata,
                            model=args.model, limits=limits,
                            starter=os.path.relpath(package / "starter", root / "profiles"),
                            reference_envelope=args.reference_envelope,
                            forced_class=args.task_class,
                            overrides={k: getattr(args, k) for k in
                                       ("model_calls_per_action", "tool_calls_per_action",
                                        "max_repairs_per_action", "per_action_wall_seconds",
                                        "eval_timeout_seconds")},
                            envelope=envelope,
                            search_target=target, target_name=target_name,
                            budget_note=budget_note)
    profile["seed"] = args.seed
    profile.setdefault("screening_execution_limits", {})["internal_validation_policy"] = args.internal_validation_policy
    profile["screening_execution_limits"]["submission_fit_policy"] = args.submission_fit_policy
    if args.coder_finalization_reserve_seconds is not None:
        reserve = args.coder_finalization_reserve_seconds
        if not 0 < reserve < float("inf"):
            raise SystemExit("coder finalization reserve must be finite and positive")
        profile["screening_execution_limits"]["coder_finalization_reserve_seconds"] = reserve
    if args.carve_folds > 1:
        screening = profile["screening_execution_limits"]
        screening["carve_folds"] = args.carve_folds
        # Each extra fold re-executes the candidate under the same timeout, and the
        # session reserves carve_folds * full_evaluation_timeout of every action for it.
        wall_per_action, timeout = (screening.get("wall_time_seconds_per_action"),
                                    screening.get("full_evaluation_timeout_seconds"))
        if wall_per_action and timeout and args.carve_folds * timeout + 30 > wall_per_action * 0.55:
            print(f"warning: {args.carve_folds} carve folds x {timeout}s exceed 55% of the "
                  f"{wall_per_action}s action wall; the coder keeps 45% of each action and a slow "
                  "candidate can end repeated_carve_incomplete", file=sys.stderr)
    # One profile and one manifest PER COMPETITION. They were campaign-wide files at
    # first, so building the second task silently overwrote the first task's profile
    # and every earlier config then pointed at a profile naming a different task.
    profile_name = f"profiles/{args.competition}.json"
    (root / "profiles").mkdir(exist_ok=True)
    (root / profile_name).write_text(json.dumps(profile, indent=1, sort_keys=True) + "\n")

    assets = {"schema_version": "swarmrsi-mlebench-assets-v1", "profile": profile_name,
              "tasks_root": "tasks", "custody": os.path.relpath(package / "custody.json", root),
              "trial_timeout_seconds": trial_timeout, "terminal_timeout_seconds": terminal_reserve}
    cache = args.model_cache or ((envelope or {}).get("ours") or {}).get("model_cache")
    if cache:
        assets["model_cache"] = str(Path(cache).resolve())
    assets_path = root / f"assets-{args.competition}.json"
    assets_path.write_text(json.dumps(assets, indent=1, sort_keys=True) + "\n")

    config = {"schema_version": "swarmrsi-run-config-v1", "benchmark": "mlebench",
              "benchmark_config": str(assets_path), "method": "swarmrsi",
              "task": args.competition, "seed": args.seed,
              "search": {"name": "bidirectional_research", "parameters": {
                  "consolidation_enabled": True,
                  "consolidation_min_realized_families": 3,
                  "consolidation_max_scored_children": 2,
                  "consolidation_failure_patience": 2,
                  "consolidation_min_controller_delta": 0.0,
                  "consolidation_exploration_floor": 0.2,
                  "consolidation_require_internal_validation": False,
              }},
              "learning": {"rsi_enabled": False, "wisdom_enabled": False},
              "limits": ({"cost_usd": None, "evaluations": None, "input_tokens": None,
                          "model_calls": None, "output_tokens": None, "tool_calls": None,
                          "wall_seconds": args.wall_seconds}
                         if args.reference_envelope else
                         {"cost_usd": None, "evaluations": args.evaluations,
                          "input_tokens": args.input_tokens, "model_calls": args.model_calls,
                          "output_tokens": args.output_tokens, "tool_calls": args.model_calls * 2,
                          "wall_seconds": args.wall_seconds}),
              "model": {"api_key_env": None, "api_key_file": "/secrets/openrouter.json",
                        "base_url": args.base_url, "max_output_tokens": 32000,
                        "model": args.model, "temperature": 0},
              "output_root": str(root / "runs"),
              "runtime": {"kind": "local", "python": args.python, "cpus": args.cpus,
                          "memory_mb": args.memory_mb, "gpus": args.gpus}}
    config_path = root / "configs" / f"{args.competition}.json"
    config_path.write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")

    print(json.dumps({"competition": args.competition, "config": str(config_path),
                      "gold_threshold": gold, "search_target": target,
                      "target_source": args.target, "starter_baseline": metadata.get("starter_baseline"),
                      "metric": metadata["search_metric"], "split": metadata["public_split"],
                      "runtime": config["runtime"], "wall_seconds": args.wall_seconds,
                      "trial_timeout_seconds": trial_timeout}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
