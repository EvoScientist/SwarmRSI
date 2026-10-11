"""extra_actions_after_target: keep searching N actions after the target is first reached."""
from evolvcoder.naturebench_session import target_stop_due


def run(metrics, minimum=3, threshold=0.1, extra=0):
    first = None
    for steps, metric in enumerate(metrics, start=1):
        stop, first = target_stop_due(steps, minimum, threshold, metric, extra, first)
        if stop:
            return steps
    return None


def test_default_stops_as_soon_as_the_target_counts():
    assert run([0.2, 0.21, 0.21, 0.3]) == 3  # reached at step 1, counts from the 3-action minimum


def test_two_extra_actions_after_the_target():
    assert run([-0.1, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2], extra=2) == 6


def test_extra_actions_count_from_the_first_time_the_target_counts():
    assert run([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], extra=2) == 5


def test_never_stops_below_the_target_or_without_one():
    assert run([0.05] * 10, extra=2) is None
    assert run([0.5] * 10, threshold=None) is None
