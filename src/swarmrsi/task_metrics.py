"""Explicit public-search metric semantics; private test grades never enter here."""
import math


def target(task):
    if 'search_metric' in task:
        metric = task['search_metric']
        if (not isinstance(metric, dict) or set(metric) != {'name', 'direction', 'role'}
                or not isinstance(metric['name'], str) or not metric['name'].strip()
                or metric['direction'] not in {'maximize', 'minimize'} or metric['role'] != 'public_validation'):
            raise ValueError('invalid public search metric')
        value = task.get('search_target')
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('search_target must be finite or null')
        # The user specifies a native threshold; the search loop maximizes
        # signed utility, including for lower-is-better benchmark metrics.
        return None if value is None else value * (1 if metric['direction'] == 'maximize' else -1)
    value = task['claude_code_score']
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('claude_code_score must be a finite aggregate-improvement threshold')
    return value


def terminal_targets(task, overall, evolved):
    threshold = target(task)
    def beats(value):
        return value is not None and threshold is not None and value > threshold
    if 'search_metric' in task:
        return {'target_threshold': threshold, 'search_metric': task['search_metric'],
                'public_target_native_threshold': task.get('search_target'),
                'public_target_enabled': threshold is not None,
                'overall_public_target_reached': beats(overall) if threshold is not None else None,
                'evolved_public_target_reached': beats(evolved) if threshold is not None else None}
    return {'target_threshold': threshold, 'overall_beats_claude_code_same_backbone': beats(overall),
            'evolved_beats_claude_code_same_backbone': beats(evolved)}


def planner_metric_instruction(metric=None):
    if metric is None:
        return 'Scores are higher-is-better aggregate improvement. '
    target({'search_metric': metric})
    return (f"Scores are public validation {metric['name']} ({metric['direction']}), represented "
            'internally as higher-is-better signed search utility, not relative SOTA improvement. '
            'Official private terminal scores are unavailable during search. ')


def public_metric(raw, metadata):
    """Return native metric and maximization utility without SOTA normalization."""
    spec = metadata['search_metric']
    target({'search_metric': spec})
    value = raw[spec['name']]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('public metric must be finite')
    return float(value), float(value) * (1 if spec['direction'] == 'maximize' else -1)
