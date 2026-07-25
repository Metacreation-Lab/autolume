import pytest

from autolume.live.core.sources import MAX_SOURCES, SourceTable


def test_observe_returns_new_table_and_leaves_original_untouched():
    empty = SourceTable()
    table = empty.observe("/latent/x", 0.5, 1.0)
    assert table is not empty
    assert empty.get("/latent/x") is None
    assert table.get("/latent/x").value == 0.5
    assert table.get("/latent/x").timestamp == 1.0


def test_repeated_observation_updates_the_value_without_growing():
    table = SourceTable().observe("/latent/x", 0.5, 1.0)
    table = table.observe("/latent/x", 0.9, 2.0)
    assert len(table.entries) == 1
    assert table.get("/latent/x").value == 0.9
    assert table.get("/latent/x").timestamp == 2.0


def test_observing_leaves_a_captured_non_empty_snapshot_untouched():
    # What the copy on write exists for: a reader holds this snapshot while the
    # control thread keeps observing, including on an address it already has.
    captured = SourceTable().observe("/latent/x", 0.5, 1.0)
    table = captured.observe("/latent/x", 0.9, 2.0).observe("/latent/y", 0.1, 3.0)

    assert len(captured.entries) == 1
    assert captured.get("/latent/x").value == 0.5
    assert captured.get("/latent/x").timestamp == 1.0
    assert captured.get("/latent/y") is None
    assert table.get("/latent/x").value == 0.9


def test_get_returns_none_for_unknown_address():
    assert SourceTable().get("/nope") is None


def test_capacity_is_bounded_and_evicts_oldest_timestamp():
    table = SourceTable()
    for i in range(MAX_SOURCES):
        table = table.observe(f"/src/{i}", float(i), 100.0 + i)
    # Refresh the first address so it is no longer the oldest.
    table = table.observe("/src/0", 0.0, 500.0)
    table = table.observe("/newcomer", 1.0, 501.0)

    assert len(table.entries) == MAX_SOURCES
    assert table.get("/newcomer") is not None
    assert table.get("/src/1") is None
    assert table.get("/src/0") is not None
    assert table.get("/src/2") is not None


def test_recent_filters_by_window():
    table = SourceTable()
    table = table.observe("/fresh", 1.0, 90.0)
    table = table.observe("/stale", 1.0, 10.0)
    assert table.recent(now=100.0, window=60.0) == ["/fresh"]
    assert table.recent(now=100.0, window=200.0) == ["/fresh", "/stale"]


def test_recent_defaults_to_sixty_second_window():
    table = SourceTable().observe("/fresh", 1.0, 50.0).observe("/stale", 1.0, 5.0)
    assert table.recent(now=100.0) == ["/fresh"]


def test_recent_sorts_alphabetically_and_hides_underscore_segments():
    table = SourceTable()
    for address in ("/zeta", "/alpha", "/foo/_meta", "/_samplerate", "/foo/bar"):
        table = table.observe(address, 1.0, 10.0)
    assert table.recent(now=10.0) == ["/alpha", "/foo/bar", "/zeta"]


@pytest.mark.parametrize("value", ["0.5", b"0.5", [1.0], None, {"a": 1}, object()])
def test_non_numeric_values_are_ignored(value):
    empty = SourceTable()
    assert empty.observe("/latent/x", value, 1.0) is empty


@pytest.mark.parametrize("value, expected", [(1, 1.0), (True, 1.0), (2.5, 2.5)])
def test_numeric_values_are_coerced_to_float(value, expected):
    entry = SourceTable().observe("/latent/x", value, 1.0).get("/latent/x")
    assert isinstance(entry.value, float)
    assert entry.value == expected


def test_addresses_are_normalized_with_a_leading_slash():
    table = SourceTable().observe("latent/x", 0.5, 1.0)
    assert list(table.entries) == ["/latent/x"]
    table = table.observe("/latent/x", 0.9, 2.0)
    assert len(table.entries) == 1
    assert table.get("latent/x").value == 0.9
    assert table.recent(now=2.0) == ["/latent/x"]
