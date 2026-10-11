"""Re-sends after a transport failure wait long enough for a proxy outage to pass."""
from evolvcoder.naturebench_session import transport_backoff_seconds


def test_backoff_grows_and_caps_at_two_minutes():
    """The ceiling was a minute while proxy outages were believed to last seconds.

    One lasted about twenty minutes and ended ten paid cells at once, so eight
    re-sends now wait 705 s in total, against a four-hour cell.
    """
    assert [transport_backoff_seconds(n) for n in (1, 2, 3, 4, 9)] == [5.0, 20.0, 80.0, 120.0, 120.0]
    assert sum(transport_backoff_seconds(n) for n in range(1, 9)) == 705.0
