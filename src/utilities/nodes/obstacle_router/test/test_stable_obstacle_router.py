import pytest

from obstacle_router.stable_classifier import (
    DYNAMIC,
    STATIC,
    UNKNOWN,
    StableObstacleClassifier,
    circular_std,
    output_membership,
)
from obstacle_router.opponent_selector import (
    initial_candidate_key,
    inside_forward_window,
    inside_opponent_corridor,
    unique_reidentification_candidate,
)
from types import SimpleNamespace


def make_classifier():
    return StableObstacleClassifier(track_length=22.0)


def feed(classifier, obstacle_id, s_values, d_values=None, start_time=0.0):
    if d_values is None:
        d_values = [0.0] * len(s_values)
    result = None
    for index, (s_value, d_value) in enumerate(zip(s_values, d_values)):
        result = classifier.update(
            obstacle_id, s_value, d_value, True, start_time + index * 0.05)
    return result


def test_noisy_stationary_obstacle_ignores_raw_style_flicker():
    classifier = make_classifier()
    positions = [3.01, 3.00, 3.02, 3.01, 2.99, 3.01] * 3
    result = feed(classifier, 1, positions)

    assert result.stable_class == STATIC
    assert result.std_s < classifier.min_std


def test_moving_opponent_becomes_dynamic():
    classifier = make_classifier()
    result = feed(classifier, 2, [3.0 + 0.03 * index for index in range(20)])

    assert result.stable_class == DYNAMIC
    assert result.std_s > classifier.max_std


def test_ambiguous_zone_keeps_previous_class():
    classifier = make_classifier()
    result = feed(classifier, 3, [4.0] * 12)
    assert result.stable_class == STATIC

    ambiguous = [4.03 if index % 2 else 3.97 for index in range(20)]
    result = feed(classifier, 3, ambiguous, start_time=1.0)

    assert classifier.min_std <= result.std_s <= classifier.max_std
    assert result.stable_class == STATIC


def test_start_finish_seam_has_small_circular_variation():
    samples = [21.98, 21.99, 0.01, 0.02]
    assert circular_std(samples, 22.0) == pytest.approx(0.015811, abs=1e-6)


def test_invisible_measurement_adds_no_evidence_or_sample():
    classifier = make_classifier()
    history = feed(classifier, 4, [5.0] * 7)
    assert history.stable_class == UNKNOWN
    before = (
        history.sample_count,
        history.static_evidence,
        history.dynamic_evidence,
    )

    history = classifier.update(4, 12.0, 2.0, False, 1.0)

    assert (
        history.sample_count,
        history.static_evidence,
        history.dynamic_evidence,
    ) == before


def test_histories_are_isolated_by_id():
    classifier = make_classifier()
    static_result = feed(classifier, 10, [2.0] * 14)
    dynamic_result = feed(
        classifier, 11, [8.0 + 0.04 * index for index in range(14)])

    assert static_result.stable_class == STATIC
    assert dynamic_result.stable_class == DYNAMIC
    assert (
        classifier.tracks[10].s_history
        is not classifier.tracks[11].s_history
    )


def test_dynamic_history_can_be_transferred_to_reacquired_id():
    classifier = make_classifier()
    previous = feed(
        classifier, 21, [6.0 + 0.04 * index for index in range(14)])
    assert previous.stable_class == DYNAMIC

    transferred = classifier.transfer_track(21, 37)
    result = classifier.update(37, 6.58, 0.0, True, 1.0)

    assert transferred is previous
    assert result.stable_class == DYNAMIC
    assert 21 not in classifier.tracks
    assert classifier.tracks[37] is previous


def test_output_membership_is_exclusive_and_unknown_is_stable_only():
    assert output_membership(STATIC) == (True, False)
    assert output_membership(DYNAMIC) == (False, True)
    assert output_membership(UNKNOWN) == (False, False)


def test_opponent_corridor_rejects_people_beyond_track_boundary():
    waypoints = [SimpleNamespace(s_m=0.0, d_left=0.5, d_right=0.5)]
    box = SimpleNamespace(s_center=0.0, d_center=0.20)
    outside_person = SimpleNamespace(s_center=0.0, d_center=0.36)

    assert inside_opponent_corridor(box, waypoints, 10.0, 0.28, 0.03)
    assert not inside_opponent_corridor(
        outside_person, waypoints, 10.0, 0.28, 0.03)


def test_opponent_candidate_must_be_in_forward_window():
    opponent = SimpleNamespace(s_center=1.0)

    assert inside_forward_window(opponent, 0.0, 10.0, 0.2, 8.0)
    assert not inside_forward_window(opponent, 1.1, 10.0, 0.2, 8.0)


def test_locked_opponent_survives_being_drawn_level_with():
    """A negative minimum keeps the target while the car passes it.

    Acquisition asks for 0.2 m ahead; the already-selected opponent is allowed
    behind, otherwise it drops out of the candidate list at the moment the two
    cars are side by side and the whole prediction/planning chain unwinds.
    """
    alongside = SimpleNamespace(s_center=0.9)

    assert not inside_forward_window(alongside, 1.0, 10.0, 0.2, 8.0)
    assert inside_forward_window(alongside, 1.0, 10.0, -1.5, 8.0)


def test_forward_window_rejects_a_target_beyond_the_rear_allowance():
    behind = SimpleNamespace(s_center=0.0)

    assert not inside_forward_window(behind, 2.0, 10.0, -1.5, 8.0)


def test_initial_candidate_prefers_box_nearer_raceline():
    box = SimpleNamespace(id=9, s_center=3.0, d_center=0.04)
    operator = SimpleNamespace(id=10, s_center=2.8, d_center=0.28)

    selected = min(
        (operator, box),
        key=lambda obstacle: initial_candidate_key(obstacle, 0.0, 10.0),
    )

    assert selected is box


def test_reidentification_refuses_two_similarly_plausible_objects():
    assert unique_reidentification_candidate(
        [(0.30, 11), (0.42, 12)],
        max_distance=1.2,
        ambiguity_margin=0.20,
    ) is None


def test_reidentification_accepts_one_clear_replacement():
    assert unique_reidentification_candidate(
        [(0.25, 11), (0.80, 12)],
        max_distance=1.2,
        ambiguity_margin=0.20,
    ) == (0.25, 11)
