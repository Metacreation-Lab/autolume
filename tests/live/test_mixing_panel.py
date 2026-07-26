"""Pure helpers behind the Mixing panel.

Everything here is about the selection: how a per parameter list is grouped
into per resolution rows, what a row displays, and what the three editing
operations do to the list. The invariant behind all of it is that `"X"` may
only ever be a trailing run, because the merge reads the mixed model's output
resolution off the last layer still kept and refuses a gap. `cut`, `cascade`
and `recover` all have to preserve it, and the sequences below are what says
they do.
"""

import pytest

from autolume.live.core.mixing import (
    ORIGIN_A,
    ORIGIN_B,
    ORIGIN_X,
    conv_names,
    selection_length,
)
from autolume.live.core.params import SetCombinedLayers
from autolume.live.ui.panels.mixing import (
    ORIGIN_MIXED,
    MixRow,
    cascade,
    cut,
    default_selection,
    fits_pair,
    model_label,
    recover,
    resolution_rows,
    row_origin,
    save_note,
)


class Network:
    def __init__(self, names):
        self._names = names

    def named_parameters(self):
        return [(name, None) for name in self._names]


def block_names(resolutions, per_block=2):
    """Parameter names shaped like a StyleGAN generator's, plus the mapping.

    Built rather than hard coded so a pair of different depths is one argument
    apart, and run through the real `conv_names` below so the mapping filter is
    the shipped one rather than a restatement of it.
    """
    names = ["mapping.fc0.weight"]
    for resolution in resolutions:
        for index in range(per_block):
            names.append(f"synthesis.b{resolution}.conv{index}.weight")
        names.append(f"synthesis.b{resolution}.torgb.weight")
    return tuple(names)


def names_for(resolutions, per_block=2):
    return conv_names(Network(block_names(resolutions, per_block)))


NAMES_A = names_for((4, 8, 16))
NAMES_B = names_for((4, 8, 16, 32))


def is_trailing_run(entries):
    """Whether every removed layer sits at the end.

    The invariant, stated here rather than in the panel: the panel cannot
    produce anything else, and this is what proves that claim rather than
    something the panel itself has to check at runtime.
    """
    seen = False
    for entry in entries:
        if entry == ORIGIN_X:
            seen = True
        elif seen:
            return False
    return True


# --- the name lists these rows are built from -------------------------------


def test_the_mapping_network_is_not_part_of_the_selection():
    assert all("mapping" not in name for name in NAMES_A)


def test_a_three_block_generator_has_three_parameters_per_block():
    assert len(NAMES_A) == 9
    assert len(NAMES_B) == 12


# --- row derivation ---------------------------------------------------------


def test_one_row_per_resolution_covering_every_parameter_in_it():
    rows = resolution_rows(NAMES_A, NAMES_A)
    assert [row.resolution for row in rows] == [4, 8, 16]
    assert rows[0].indices == (0, 1, 2)
    assert rows[1].indices == (3, 4, 5)
    assert rows[2].indices == (6, 7, 8)


def test_the_rows_cover_every_index_the_selection_has():
    rows = resolution_rows(NAMES_A, NAMES_B)
    covered = [index for row in rows for index in row.indices]
    assert sorted(set(covered)) == list(range(selection_length(
        Network(block_names((4, 8, 16))), Network(block_names((4, 8, 16, 32)))
    )))


def test_a_deeper_second_model_gets_a_row_of_its_own():
    rows = resolution_rows(NAMES_A, NAMES_B)
    assert [row.resolution for row in rows] == [4, 8, 16, 32]
    assert rows[3].indices == (9, 10, 11)


def test_a_row_only_the_deeper_model_has_is_reachable_from_that_model_alone():
    rows = resolution_rows(NAMES_A, NAMES_B)
    assert rows[3].a_indices == ()
    assert rows[3].b_indices == (9, 10, 11)


def test_a_row_both_models_have_is_reachable_from_both():
    rows = resolution_rows(NAMES_A, NAMES_B)
    assert rows[0].a_indices == (0, 1, 2)
    assert rows[0].b_indices == (0, 1, 2)


def test_rows_follow_network_order_rather_than_numeric_order():
    # Not a doubling ladder. The row order is the order the parameters appear
    # in, which is the order the blocks actually run in.
    names = names_for((4, 64, 16))
    rows = resolution_rows(names, names)
    assert [row.resolution for row in rows] == [4, 64, 16]


def test_an_empty_pair_has_no_rows():
    assert resolution_rows((), ()) == ()


def test_a_row_compares_by_what_it_covers():
    assert MixRow(8, (1,), (1,), (1,)) == MixRow(8, (1,), (1,), (1,))
    assert MixRow(8, (1,), (1,), (1,)) != MixRow(8, (2,), (2,), (2,))


# --- the default selection --------------------------------------------------


def test_the_default_is_model_a_throughout():
    # All-A is bit identical to model A, so a fresh mix looks exactly like what
    # was already on screen and every click from there is a visible change.
    assert default_selection(NAMES_A, NAMES_A) == (ORIGIN_A,) * 9


def test_the_default_takes_the_deeper_models_tail_from_that_model():
    default = default_selection(NAMES_A, NAMES_B)
    assert default == (ORIGIN_A,) * 9 + (ORIGIN_B,) * 3


def test_the_default_is_the_length_the_merge_demands():
    default = default_selection(NAMES_A, NAMES_B)
    assert len(default) == max(len(NAMES_A), len(NAMES_B))
    assert fits_pair(default, NAMES_A, NAMES_B)


def test_the_default_never_removes_a_layer():
    assert ORIGIN_X not in default_selection(NAMES_A, NAMES_B)


def test_a_selection_of_the_wrong_length_does_not_fit_the_pair():
    assert not fits_pair((ORIGIN_A,) * 5, NAMES_A, NAMES_B)
    assert not fits_pair((), NAMES_A, NAMES_B)


def test_the_default_is_a_valid_event_payload():
    # `SetCombinedLayers` is the only way the selection reaches state, and the
    # control thread refuses anything that is not A, B or X.
    entries = default_selection(NAMES_A, NAMES_B)
    assert set(SetCombinedLayers(entries).entries) <= {ORIGIN_A, ORIGIN_B}


# --- what a row displays ----------------------------------------------------


ROWS = resolution_rows(NAMES_A, NAMES_B)


def test_a_row_whose_parameters_all_agree_shows_that_model():
    entries = default_selection(NAMES_A, NAMES_B)
    assert row_origin(entries, ROWS[0]) == ORIGIN_A
    assert row_origin(entries, ROWS[3]) == ORIGIN_B


def test_a_row_whose_parameters_disagree_shows_mixed():
    entries = list(default_selection(NAMES_A, NAMES_B))
    entries[4] = ORIGIN_B
    assert row_origin(tuple(entries), ROWS[1]) == ORIGIN_MIXED


def test_a_removed_row_shows_the_cut():
    entries = (ORIGIN_A,) * 3 + (ORIGIN_X,) * 9
    assert row_origin(entries, ROWS[1]) == ORIGIN_X


def test_a_row_with_no_entries_behind_it_shows_mixed_rather_than_guessing():
    assert row_origin((), ROWS[0]) == ORIGIN_MIXED


# --- the cascade ------------------------------------------------------------


def test_a_cascade_from_a_middle_row_sets_every_parameter_in_that_row():
    entries = default_selection(NAMES_A, NAMES_B)
    updated = cascade(entries, ROWS[1], ORIGIN_B)
    assert updated[3:6] == (ORIGIN_B,) * 3


def test_a_cascade_leaves_every_other_row_alone():
    entries = default_selection(NAMES_A, NAMES_B)
    updated = cascade(entries, ROWS[1], ORIGIN_B)
    assert updated[:3] == entries[:3]
    assert updated[6:] == entries[6:]


def test_a_cascade_makes_a_mixed_row_agree():
    entries = list(default_selection(NAMES_A, NAMES_B))
    entries[4] = ORIGIN_B
    updated = cascade(tuple(entries), ROWS[1], ORIGIN_A)
    assert row_origin(updated, ROWS[1]) == ORIGIN_A


def test_a_cascade_to_a_model_that_has_no_layer_there_changes_nothing():
    # Setting a parameter to a model that does not have it is rejected by the
    # merge, so the panel greys the box and the cascade writes nothing even if
    # it is called anyway.
    entries = default_selection(NAMES_A, NAMES_B)
    assert cascade(entries, ROWS[3], ORIGIN_A) == entries


def test_a_cascade_never_writes_a_removal():
    entries = default_selection(NAMES_A, NAMES_B)
    for row in ROWS:
        for origin in (ORIGIN_A, ORIGIN_B):
            assert ORIGIN_X not in cascade(entries, row, origin)


def test_a_cascade_keeps_the_selections_length():
    entries = default_selection(NAMES_A, NAMES_B)
    assert len(cascade(entries, ROWS[2], ORIGIN_B)) == len(entries)


# --- the cut ----------------------------------------------------------------


def test_a_cut_removes_the_row_and_everything_after_it():
    entries = default_selection(NAMES_A, NAMES_B)
    updated, _ = cut(entries, entries, ROWS[2])
    assert updated == (ORIGIN_A,) * 6 + (ORIGIN_X,) * 6


def test_a_cut_only_ever_produces_a_trailing_run():
    entries = default_selection(NAMES_A, NAMES_B)
    for row in ROWS:
        updated, _ = cut(entries, entries, row)
        assert is_trailing_run(updated), row


def test_a_cut_keeps_the_selections_length():
    entries = default_selection(NAMES_A, NAMES_B)
    updated, _ = cut(entries, entries, ROWS[1])
    assert len(updated) == len(entries)


def test_a_cut_holds_on_to_what_was_kept_before_it():
    entries = list(default_selection(NAMES_A, NAMES_B))
    entries[4] = ORIGIN_B
    _, cached = cut(tuple(entries), tuple(entries), ROWS[3])
    assert cached[4] == ORIGIN_B


def test_what_a_cut_holds_on_to_never_contains_a_removal():
    # Recover reads this back, so a removal in it would restore an X and could
    # break the trailing run invariant.
    entries = default_selection(NAMES_A, NAMES_B)
    cached = entries
    for row in reversed(ROWS):
        entries, cached = cut(entries, cached, row)
        assert ORIGIN_X not in cached, row


# --- recover ----------------------------------------------------------------


def test_recover_puts_back_the_resolution_it_was_pressed_on():
    entries = default_selection(NAMES_A, NAMES_B)
    cut_entries, cached = cut(entries, entries, ROWS[2])
    restored = recover(cut_entries, cached, ROWS, 2)
    assert restored[:9] == entries[:9]


def test_recover_leaves_the_deeper_resolutions_removed():
    entries = default_selection(NAMES_A, NAMES_B)
    cut_entries, cached = cut(entries, entries, ROWS[2])
    restored = recover(cut_entries, cached, ROWS, 2)
    assert restored[9:] == (ORIGIN_X,) * 3


def test_recover_on_the_last_row_restores_everything():
    entries = default_selection(NAMES_A, NAMES_B)
    cut_entries, cached = cut(entries, entries, ROWS[3])
    restored = recover(cut_entries, cached, ROWS, 3)
    assert restored == entries


def test_recover_preserves_the_trailing_run():
    entries = default_selection(NAMES_A, NAMES_B)
    for index in range(len(ROWS)):
        cut_entries, cached = cut(entries, entries, ROWS[index])
        assert is_trailing_run(recover(cut_entries, cached, ROWS, index))


def test_recover_restores_what_was_actually_held_rather_than_the_default():
    entries = list(default_selection(NAMES_A, NAMES_B))
    entries[4] = ORIGIN_B
    cut_entries, cached = cut(tuple(entries), tuple(entries), ROWS[2])
    restored = recover(cut_entries, cached, ROWS, 2)
    assert restored[4] == ORIGIN_B


def test_recover_on_a_row_that_no_longer_exists_changes_nothing():
    entries = (ORIGIN_X,) * 12
    assert recover(entries, entries, ROWS, 99) == entries
    assert recover(entries, entries, (), 0) == entries


def test_recovering_one_row_leaves_the_deeper_ones_for_their_own_press():
    # The deliberate shape of Recover, stated as a whole walk: a cut at
    # resolution 16 takes 16 and 32 with it, and getting both back is two
    # presses, not one. Walking up from the cut restores the whole selection.
    entries = default_selection(NAMES_A, NAMES_B)
    cut_entries, cached = cut(entries, entries, ROWS[2])
    walked = cut_entries
    for index in range(2, len(ROWS)):
        walked = recover(walked, cached, ROWS, index)
        assert is_trailing_run(walked)
    assert walked == entries


def test_cutting_and_recovering_the_deepest_row_comes_back_where_it_started():
    entries = default_selection(NAMES_A, NAMES_B)
    cut_entries, cached = cut(entries, entries, ROWS[-1])
    assert recover(cut_entries, cached, ROWS, len(ROWS) - 1) == entries


def test_a_long_sequence_of_edits_never_breaks_the_trailing_run():
    # The invariant under mixed use, not one operation at a time: the merge's
    # save and preview guarantees both rest on it.
    entries = default_selection(NAMES_A, NAMES_B)
    cached = entries
    script = [
        ("cascade", 1, ORIGIN_B),
        ("cut", 3, None),
        ("cascade", 0, ORIGIN_B),
        ("recover", 3, None),
        ("cut", 2, None),
        ("cascade", 1, ORIGIN_A),
        ("recover", 2, None),
        ("cut", 1, None),
        ("recover", 1, None),
    ]
    for action, index, origin in script:
        if action == "cascade":
            entries = cascade(entries, ROWS[index], origin)
        elif action == "cut":
            entries, cached = cut(entries, cached, ROWS[index])
        else:
            entries = recover(entries, cached, ROWS, index)
        assert is_trailing_run(entries), (action, index)
        assert len(entries) == 12
        assert set(entries) <= {ORIGIN_A, ORIGIN_B, ORIGIN_X}


# --- the save row and the headings ------------------------------------------


class Status:
    def __init__(self, path=None, error=None):
        self.path = path
        self.error = error


def test_a_saved_merge_names_the_file_it_wrote():
    assert save_note(Status(path="/models/merged.pkl")) == "Saved to /models/merged.pkl."


def test_a_failed_save_shows_the_failure_ahead_of_any_earlier_path():
    assert save_note(Status(path="/models/a.pkl", error="Disk full")) == "Disk full"


def test_nothing_saved_yet_says_nothing():
    assert save_note(Status()) is None


def test_a_model_heading_is_the_bare_file_name():
    assert model_label("/models/wikiart-1024.pkl") == "wikiart-1024.pkl"
    assert model_label("C:\\models\\faces.pkl") == "faces.pkl"


def test_an_empty_slot_is_named_as_empty_rather_than_blank():
    assert model_label(None) == "No second model loaded"
    assert model_label("") == "No second model loaded"


@pytest.mark.parametrize(
    "text",
    [
        save_note(Status(path="/a.pkl")),
        model_label("/models/a.pkl"),
        model_label(None),
        ORIGIN_MIXED,
    ],
)
def test_every_panel_sentence_is_drawable_by_the_bundled_font(text):
    assert text.isascii(), text
