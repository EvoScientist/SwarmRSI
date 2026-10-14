"""Full public search followed by a one-shot private MLE-Bench grade.

result.json is public selection evidence. Only benchmark_result.json delivers
an official metric. Interrupted private work is never implicitly retried.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .api import TaskSpec, RunResult
from .campaign import artifact_workspace_sha256
from .contracts import canonical_sha256
from .learning_evidence import source_ref, verify_ref
from .mlebench_tasks import verify_assets
from .naturebench_cli import run_lock, executable_path, _overlap
from .naturebench_evaluator import _candidate_command, _attest
from .naturebench_session import INTERNAL_VALIDATION_NAME, SUBMISSION_FIT_ENV
from .retained_results import read_json, write_immutable, require, verify_seal
from .candidate_execution import for_context, validate_binding, execute as execute_candidate, verify_receipt
from .container_runtime import RuntimeUnavailable


def manifest(config):
    require(config.benchmark_config is not None, 'MLE requires benchmark_config')
    raw = read_json(config.benchmark_config)
    allowed = {'schema_version', 'profile', 'tasks_root', 'custody',
               'trial_timeout_seconds', 'terminal_timeout_seconds', 'model_cache'}
    require(set(raw) <= allowed and allowed - {'model_cache'} <= set(raw),
            'invalid MLE assets manifest fields')
    require(raw['schema_version'] == 'swarmrsi-mlebench-assets-v1', 'unknown MLE assets schema')
    for field in ('trial_timeout_seconds', 'terminal_timeout_seconds'):
        require(type(raw[field]) is int and raw[field] > 0, 'invalid ' + field)
    # A read-only directory of pretrained weights for the candidate, or None.
    #
    # It used to be None unconditionally, and that silently decided what the search could
    # attempt. The sandbox has torch, transformers, timm and sentence-transformers
    # installed but runs --unshare-net, so with no cache a candidate cannot obtain a
    # single pretrained weight. On jigsaw every hypothesis the planner raised stayed
    # inside the TF-IDF family and the score stopped at 0.9856, while the reference --
    # whose config carries torch_hub_dir and pretrain_model_dir -- reached 0.98805.
    # Denying the weights is a decision about the method, so it is now stated in the
    # manifest rather than hard-coded.
    cache = raw.get('model_cache')
    result = {**raw, 'model_cache': Path(cache).resolve() if cache else None}
    for field in ('profile', 'tasks_root', 'custody'):
        require(isinstance(raw[field], str) and raw[field], 'invalid asset path')
        result[field] = (config.benchmark_config.parent / raw[field]).resolve()
    return result


class MLEBenchAdapter:
    def task(self, task_id, config):
        require(task_id == config.task, 'run config task differs from the requested task')
        assets = manifest(config)
        custody = verify_assets(assets['custody'])
        root = assets['tasks_root'] / task_id
        # The competition used to be a module constant, so only the one competition the
        # constant named could run. It now comes from the task package, which
        # deploy/mle/make_task_root.py generates per competition and which custody
        # covers, so the adapter is bound to the package rather than to a name in code.
        metadata = read_json(root / 'metadata.json')
        require(metadata['task_id'] == task_id, 'task package names a different competition')
        metric = metadata['search_metric']
        require(metric['direction'] in ('maximize', 'minimize'), 'task package metric has no usable direction')
        profile = read_json(assets['profile'])
        entry = profile['tasks'][task_id]
        require(source_ref(root / 'metadata.json') in custody['refs']
                and source_ref(root / 'evaluation/evaluator.py') in custody['refs'], 'task root differs from bound custody')
        require(entry['search_metric'] == metric and 'claude_code_score' not in entry,
                'MLE profile must use the task package metric, not a Claude/SOTA threshold')
        require(not config.learning.rsi_enabled and not config.learning.wisdom_enabled
                and config.winner_registry is None, 'MLE learning/reuse not yet qualified; use explicit search without RSI/Wisdom')
        public = root / 'problem/data'
        starter = (assets['profile'].parent / entry['starter']).resolve()
        protected = [Path(custody['prepared']), Path(custody['official_source']), root / 'evaluation', assets['custody']]
        mounts = [public, starter, executable_path(config.runtime.python).parent.parent, config.output_root,
                  Path('/usr'), Path('/etc'), Path('/bin'), Path('/lib'), Path('/lib64')]
        require(not any(_overlap(m, p) for m in mounts for p in protected), 'public/runtime/output mount overlaps full-data or private MLE assets')
        return TaskSpec('mlebench', task_id, (root / 'problem/README.md').read_text(),
                        starter, public, 'mlebench:' + custody['assets_sha256'],
                        metric['name'], metric['direction'], assets['custody'])


def _submission_fit_policy(assets):
    """The profile's opt-in policy, or legacy when the manifest carries no profile."""
    profile_path = assets.get('profile')
    if not profile_path:
        return 'legacy'
    limits = read_json(profile_path).get('screening_execution_limits') or {}
    return str(limits.get('submission_fit_policy', 'legacy'))


def _submission_fit_evidence(frozen, policy):
    """What the terminal program reported fitting its submission model on.

    Read from internal_validation.json, the one file a candidate may rewrite at terminal
    time without changing its source digest. Absent, malformed, or not requested: recorded
    as such. This is evidence for the ledger, not a gate -- a submission is graded whatever
    it says here.
    """
    if policy != 'full_refit':
        return {'policy': policy}
    # Raw, not read_internal_validation: that reader keeps only the gate's four fields
    # and returns None without a metric estimate, and a program may legitimately report
    # its fit without one.
    path = frozen / INTERNAL_VALIDATION_NAME
    try:
        raw = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None
    except (OSError, ValueError):
        raw = None
    reported = raw.get('submission_fit') if isinstance(raw, dict) else None
    if not isinstance(reported, dict):
        return {'policy': policy, 'reported': False}
    return {'policy': policy, 'reported': True,
            'mode': reported.get('mode'), 'rows_available': reported.get('rows_available'),
            'rows_fit': reported.get('rows_fit')}


def _source_digest(frozen):
    """Frozen-source digest for the post-execution check.

    The standing coder instruction asks run.py to write internal_validation.json
    at the workspace root, so a terminal run that obeys it rewrites that file;
    r7 (2026-09-17) lost its official grade to exactly that. The file is a
    candidate output, not source, and is excluded here only.
    """
    return artifact_workspace_sha256(frozen, exclude_names=(INTERNAL_VALIDATION_NAME,))


def _receipt(context, public):
    path = context.output_dir / 'benchmark_result.json'
    if not path.exists():
        return None
    record = read_json(path); verify_seal(record, 'receipt_sha256')
    require(record['public_result_ref'] == source_ref(context.output_dir / 'result.json'), 'public selection changed after terminal grade')
    for ref in record['refs']:
        verify_ref(ref)
    frozen = context.output_dir / 'terminal/program'
    terminal = frozen.parent
    actual_refs = [source_ref(p) for p in sorted(terminal.glob('*')) if p.is_file()]
    require(record['refs'] == actual_refs, 'terminal required reference set changed')
    required = {'selection.json'}
    if record['status'] == 'completed':
        required |= {'launch.json', 'execution.log', 'grade_launch.json', 'grader.log', 'official_report.json'}
    require(required.issubset({Path(ref['path']).name for ref in actual_refs}), 'missing required terminal evidence')
    selection = read_json(terminal / 'selection.json')
    require(selection['public_result'] == record['public_result_ref']
            and selection['selected_node'] == read_json(public.evidence)['best_node_id']
            and selection['workspace_sha256'] == record['workspace_sha256'] == artifact_workspace_sha256(public.artifact),
            'terminal program differs from public selection')
    require(selection['selected_executable'] == source_ref(frozen / 'run.py'), 'selected executable changed')
    # After a failed full-refit execution the graded submission comes from the second,
    # search-fit execution; its launch and receipt are the ones that must verify.
    suffix = '_search_fit' if (record.get('submission_fit') or {}).get('fallback') else ''
    if suffix:
        require({f'launch{suffix}.json', f'execution{suffix}.log'}.issubset({Path(ref['path']).name for ref in actual_refs}),
                'missing required terminal evidence')
    if (terminal / f'launch{suffix}.json').exists():
        launch = read_json(terminal / f'launch{suffix}.json')
        require(launch['selection'] == source_ref(terminal / 'selection.json') and launch['sandboxed'] is True,
                'launch differs from selected source or sandbox requirement')
        binding = for_context(context)
        require(launch.get('candidate_runtime') == binding, 'terminal backend differs from requested runtime')
        if binding is not None and record['status'] == 'completed':
            spec, _ = validate_binding(binding)
            custody = verify_assets(manifest(context.config)['custody'])
            receipt = verify_receipt(binding, read_json(terminal / f'candidate_runtime{suffix}.json'),
                                     argv=[spec.python, 'run.py'], workspace=frozen,
                                     public_data=Path(custody['prepared']) / 'public')
            require(receipt['status'] == 'candidate_exited' and receipt['returncode'] == 0
                    and launch['argv'] == [spec.python, 'run.py'], 'terminal container did not complete successfully')
    if (terminal / 'grade_launch.json').exists():
        custody = verify_assets(manifest(context.config)['custody'])
        grade_launch = read_json(terminal / 'grade_launch.json')
        require(grade_launch['argv'] == _grade_command(context, custody, terminal)
                and grade_launch['output_attestation'] == record['output_attestation'],
                'official grade invocation differs from selected submission/source/data')
    if 'source_sha256' in record:
        require(selection['source_sha256'] == record['source_sha256'] == _source_digest(frozen), 'terminal source changed')
    else:  # records sealed before the internal-validation exclusion existed
        require(artifact_workspace_sha256(frozen) == record['workspace_sha256'], 'terminal source changed')
    if record['status'] == 'completed':
        require(_attest(frozen / 'output') == record['output_attestation'], 'terminal output changed')
        report = read_json(context.output_dir / 'terminal/official_report.json')
        require(report['valid_submission'] is True and report['score'] == record['official_score']
                and report['competition_id'] == context.config.task
                and report['submission_path'] == str(frozen / 'output/submission.csv'), 'official result binding differs')
    details = {**public.details, 'public_auc': public.score, 'official_terminal_auc': record['official_score'],
               'metric_role': 'official_private_terminal', 'private_feedback_used_for_search': False,
               'new_scored_candidates': record['new_scored_candidates'], 'receipt_sha256': record['receipt_sha256'],
               'submission_fit': record.get('submission_fit')}
    return RunResult(context.run_id, context.config.task, context.config.method, record['status'],
                     frozen, record['official_score'], public.evaluator_id, path,
                     record['reason'], {**public.usage, 'terminal_evaluations': record['terminal_evaluations'],
                                       'total_elapsed_seconds': record['total_elapsed_seconds']}, details)


def workflow_status(adapter, task, context):
    inner = adapter._search_status(task, context)
    if (context.output_dir / 'result.json').exists():
        public = adapter._terminal(task, context)
        result = _receipt(context, public)
        if result is not None:
            from .unified_run import result_dict
            return {'status': result.status, 'result': result_dict(result)}
        return {'status': 'unfinished_terminal', 'search': inner, 'official_terminal_auc': None}
    return inner


def workflow_preflight(adapter, task, context):
    assets = manifest(context.config)
    custody = verify_assets(assets['custody'])
    binding = for_context(context)
    if binding is None:
        require(shutil.which('bwrap') is not None, 'MLE local runs require bubblewrap')
    else:
        from .container_runtime import preflight
        preflight(validate_binding(binding)[0])
    probe = subprocess.run([str(executable_path(context.config.runtime.python)), '-c',
        'from pathlib import Path;from swarmrsi.mlebench_tasks import load_official;'
        f'load_official(Path({custody["official_source"]!r}));'
        'from mlebench.grade import grade_csv;from mlebench.registry import registry;'
        f'registry.set_data_dir(Path({custody["prepared"]!r}).parent.parent).get_competition({task.task_id!r})'],
        capture_output=True, text=True, timeout=60)
    require(probe.returncode == 0, 'official grader runtime preflight failed: ' + probe.stderr[-3000:])
    result = adapter._operate(task, context, execute=False)
    return {**result, 'benchmark': 'mlebench', 'official_grader_importable': True,
            'terminal_evaluations_reserved': 1, 'terminal_seconds_reserved': assets['terminal_timeout_seconds']}


def _process(argv, *, cwd, env, deadline, log):
    timeout = deadline - time.time()
    require(timeout > 0, 'terminal budget exhausted before process launch')
    with log.open('x') as stream:
        child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise TimeoutError('terminal process exceeded shared deadline')


def _grade_command(context, custody, terminal):
    return [str(executable_path(context.config.runtime.python)), '-m', 'swarmrsi.mlebench_tasks', 'grade',
            '--source', custody['official_source'], '--prepared', custody['prepared'],
            '--submission', str(terminal / 'program/output/submission.csv'), '--result', str(terminal / 'official_report.json'),
            '--task', context.config.task]


def workflow_run(adapter, task, context):
    from .unified_run import local_admission
    assets = manifest(context.config)
    custody = verify_assets(assets['custody'])
    # Initialize/validate normal run ownership before adding the workflow lock.
    # This preflight never creates a model or runs a candidate.
    if not (context.output_dir / 'inputs.json').exists():
        ready = workflow_preflight(adapter, task, context)
        require(ready['ready'], 'public search preflight failed')
    with run_lock(context.output_dir / 'workflow'):
        marker = context.output_dir / 'workflow/start.json'
        if not marker.exists():
            write_immutable(marker, {'started_at': time.time(), 'config_sha256': canonical_sha256(context.config.to_dict())})
        start = read_json(marker)
        require(start['config_sha256'] == canonical_sha256(context.config.to_dict()), 'workflow configuration changed')
        public = adapter._operate(task, context, execute=True)
        if public.status != 'completed':
            return public
        receipt = _receipt(context, public)
        if receipt is not None:
            return receipt
        terminal = context.output_dir / 'terminal'
        if terminal.exists():
            return RunResult(context.run_id, task.task_id, context.config.method, 'pending',
                             reason='original terminal attempt is unfinished; no implicit retry')
        terminal.mkdir()
        frozen = terminal / 'program'
        ignore = shutil.ignore_patterns('output', '.swarmrsi', '__pycache__', '*.pyc', '.git', '.pytest_cache', '.mypy_cache', '.ruff_cache')
        require(not any(p.is_symlink() for p in public.artifact.rglob('*')), 'terminal artifact contains symlinks')
        shutil.copytree(public.artifact, frozen, ignore=ignore)
        digest = artifact_workspace_sha256(frozen)
        require(digest == artifact_workspace_sha256(public.artifact), 'final source copy differs')
        final = read_json(public.evidence)
        new_count = sum(n['id'] != 'root' and n['artifact_valid'] for n in read_json(public.evidence.parent / 'checkpoint.json')['nodes'])
        source_digest = _source_digest(frozen)
        selection = {'public_result': source_ref(context.output_dir / 'result.json'),
                     'selected_node': final['best_node_id'], 'workspace_sha256': digest, 'source_sha256': source_digest,
                     'selected_executable': source_ref(frozen / 'run.py'), 'selected_at': time.time()}
        write_immutable(terminal / 'selection.json', selection)
        (frozen / 'output').mkdir()
        deadline = time.time() + assets['terminal_timeout_seconds']
        if context.config.limits.wall_seconds is not None:
            deadline = min(deadline, start['started_at'] + context.config.limits.wall_seconds)
        score, status, reason, attestation, terminal_count = None, 'evaluation_invalid', None, None, 0
        submission_fit = {'policy': 'legacy'}
        try:
            env, _ = local_admission(context.config)
            # The official grade runs `-m swarmrsi.mlebench_tasks` in a subprocess. Without
            # PYTHONPATH it resolves to whatever copy of swarmrsi the image has installed,
            # which is not the controller's own source: every terminal grade failed on
            # `unrecognized arguments: --task`, an argument this source added. Controller and
            # grader must be the same code, so the path is carried through when one is set.
            if os.environ.get('PYTHONPATH'):
                env['PYTHONPATH'] = os.environ['PYTHONPATH']
            env.update({'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'PYTHONDONTWRITEBYTECODE': '1',
                        'DATA_DIR': str(Path(custody['prepared']) / 'public'), 'OUTPUT_DIR': str(frozen / 'output')})
            # Under the opt-in policy the terminal execution announces itself, so a program
            # written to the submission contract fits on every training row here and only
            # here. Absent the policy nothing changes: the flag is never set, and a program
            # that looks for it behaves as it did during search.
            submission_fit_policy = _submission_fit_policy(assets)
            if submission_fit_policy == 'full_refit':
                env[SUBMISSION_FIT_ENV] = 'full'
            binding = for_context(context)

            def execute_terminal(full_refit, suffix):
                """One terminal execution; True when the program exited cleanly."""
                nonlocal terminal_count
                if binding is not None:
                    spec, _ = validate_binding(binding)
                    argv = [spec.python, 'run.py']
                    require(deadline > time.time(), 'terminal budget exhausted before container launch')
                    write_immutable(terminal / f'launch{suffix}.json', {'argv': argv, 'candidate_runtime': binding,
                        'deadline': deadline, 'sandboxed': True, 'selection': source_ref(terminal / 'selection.json')})
                    terminal_count += 1
                    # Legacy keeps the exact historical call; only the opt-in policy adds a keyword.
                    flagged = {'environment': {SUBMISSION_FIT_ENV: 'full'}} if full_refit else {}
                    receipt = execute_candidate(binding, workspace=frozen,
                        public_data=Path(custody['prepared']) / 'public', argv=argv, timeout=deadline-time.time(), **flagged)
                    write_immutable(terminal / f'candidate_runtime{suffix}.json', receipt)
                    with (terminal / f'execution{suffix}.log').open('x') as stream:
                        stream.write(receipt.get('stdout', '') + receipt.get('stderr', ''))
                    require(receipt['cleanup_confirmed'] is True, 'terminal container cleanup unconfirmed')
                    return receipt['status'] == 'candidate_exited' and receipt['returncode'] == 0
                run_env = dict(env)
                if not full_refit:
                    run_env.pop(SUBMISSION_FIT_ENV, None)
                argv, cwd, candidate_env, sandboxed = _candidate_command(workspace=frozen,
                    data_dir=Path(custody['prepared']) / 'public', runtime_python=executable_path(context.config.runtime.python),
                    require_sandbox=True, environment=run_env,
                    inherit_model_cache=False, model_cache=assets.get('model_cache'))
                write_immutable(terminal / f'launch{suffix}.json', {'argv': argv, 'environment': candidate_env, 'deadline': deadline,
                                                                   'sandboxed': sandboxed, 'selection': source_ref(terminal / 'selection.json')})
                terminal_count += 1
                return _process(argv, cwd=cwd, env=candidate_env, deadline=deadline, log=terminal / f'execution{suffix}.log') == 0

            full_refit = submission_fit_policy == 'full_refit'
            fallback = False
            if not execute_terminal(full_refit, ''):
                # The all-rows branch is code the search never executed: search only ever ran
                # the program on the carved split. When that branch crashes, the program the
                # search actually validated is still available, so it runs once more without
                # the flag rather than losing the whole run (jigsaw-c1d-seed0: IndexError in
                # the refit branch after a 5 h search). Nothing has been graded at this point.
                require(full_refit and deadline > time.time(),
                        'terminal container execution failed' if binding is not None else 'terminal execution failed')
                shutil.rmtree(frozen / 'output')
                (frozen / 'output').mkdir()
                fallback = True
                require(execute_terminal(False, '_search_fit'), 'terminal execution failed, also without full refit')
            require(_source_digest(frozen) == source_digest, 'candidate modified frozen source')
            submission_fit = _submission_fit_evidence(frozen, submission_fit_policy)
            if fallback:
                submission_fit = {**submission_fit, 'fallback': 'search_fit_after_full_refit_failure'}
            attestation = _attest(frozen / 'output')
            verify_assets(assets['custody'])
            grade_argv = _grade_command(context, custody, terminal)
            write_immutable(terminal / 'grade_launch.json', {'argv': grade_argv, 'output_attestation': attestation})
            require(_process(grade_argv, cwd=terminal, env=env, deadline=deadline, log=terminal / 'grader.log') == 0, 'official grading failed')
            report = read_json(terminal / 'official_report.json')
            require(report['valid_submission'] is True, 'official grader rejected submission')
            require(_attest(frozen / 'output') == attestation, 'submission changed during grading')
            score, status = report['score'], 'completed'
        except (ValueError, OSError, TimeoutError, RuntimeUnavailable) as exc:
            reason = str(exc)
        refs = [source_ref(p) for p in sorted(terminal.glob('*')) if p.is_file()]
        body = {'schema_version': 'swarmrsi-mlebench-terminal-v1', 'status': status, 'official_score': score,
                'reason': reason, 'workspace_sha256': digest, 'source_sha256': source_digest, 'output_attestation': attestation,
                'new_scored_candidates': new_count, 'public_result_ref': source_ref(context.output_dir / 'result.json'),
                'refs': refs, 'terminal_evaluations': terminal_count, 'total_elapsed_seconds': time.time() - start['started_at'],
                'submission_fit': submission_fit}
        write_immutable(context.output_dir / 'benchmark_result.json', {**body, 'receipt_sha256': canonical_sha256(body)})
        return _receipt(context, public)
