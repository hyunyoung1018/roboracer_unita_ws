"""A prediction the state machine cannot recognise is a prediction it ignores.

One car, renamed once on the way through:

    tracking_node        calls it 83
    h2h_tracking_node    republishes it on /tracking/dynamic_obstacles as
                         logical_opponent_id (1000000)
    opp_prediction       reads that topic, so its PredictionArray says 1000000
    state_machine        reads /tracking/obstacles, so the obstacle is 83

_check_free_frenet matches prediction to obstacle by `prediction_id ==
obs.id`, so 1000000 == 83 is false on every tick and the dyn/nopred branch
runs instead - the opponent frozen where it stands. Measured on the car:

    obs 83 via dyn/nopred (id_mismatch or empty) free=-0.095 at 2.59m

Head to head only; state_machine_node.py is what time_trials runs.
"""

from types import SimpleNamespace

from state_machine.h2h_state_machine import H2HStateMachine

LOGICAL = 1000000


def obstacle(obstacle_id, s, d=0.0):
    return SimpleNamespace(id=obstacle_id, s_center=s, d_center=d)


def machine(raw_obstacles=(), opponents=(), stamp=0.0):
    m = H2HStateMachine.__new__(H2HStateMachine)
    m.obstacles_perception = list(raw_obstacles)
    m._opponent_obstacles = list(opponents)
    m._opponent_stamp = stamp
    m.opponent_stream_timeout_sec = 0.5
    m.now_sec = lambda: 0.0
    m.obstacles_prediction = []
    m.obstacles_prediction_id = None
    m.prediction_dt = 0.1
    return m


def prediction(prediction_id, count=3, dt=0.1):
    return SimpleNamespace(
        id=prediction_id, dt=dt,
        predictions=[SimpleNamespace(pred_s=float(i), pred_d=0.0)
                     for i in range(count)])


def deliver(m, data):
    H2HStateMachine.obstacle_prediction_cb(m, data)


def test_the_logical_id_is_translated_to_the_tracker_id():
    """The bug, stated as the fix."""
    m = machine(raw_obstacles=[obstacle(83, 10.65, -0.13)],
                opponents=[obstacle(LOGICAL, 10.65, -0.13)])
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == 83


def test_without_a_match_the_arriving_id_is_left_alone():
    """No opponent on the dynamic stream: nothing to translate against."""
    m = machine(raw_obstacles=[obstacle(83, 10.65)], opponents=[])
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == LOGICAL


def test_a_stale_dynamic_stream_does_not_translate():
    """Older than opponent_stream_timeout_sec is not a current position."""
    m = machine(raw_obstacles=[obstacle(83, 10.65)],
                opponents=[obstacle(LOGICAL, 10.65)], stamp=-5.0)
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == LOGICAL


def test_the_right_obstacle_is_picked_out_of_several():
    m = machine(
        raw_obstacles=[obstacle(80, 3.0, 0.4), obstacle(83, 10.65, -0.13),
                       obstacle(91, 15.0, 0.2)],
        opponents=[obstacle(LOGICAL, 10.65, -0.13)])
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == 83


def test_an_empty_prediction_does_not_relabel_the_previous_one():
    """The shared callback keeps the old array; relabelling it would lie."""
    m = machine(raw_obstacles=[obstacle(83, 10.65)],
                opponents=[obstacle(LOGICAL, 10.65)])
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == 83
    m.obstacles_perception = [obstacle(91, 10.65)]
    m._opponent_obstacles = [obstacle(LOGICAL, 10.65)]
    deliver(m, prediction(LOGICAL, count=0))
    assert m.obstacles_prediction_id == 83


def test_the_translation_follows_a_reidentification():
    """The tracker renames 83 to 91; the prediction must follow."""
    m = machine(raw_obstacles=[obstacle(83, 10.65)],
                opponents=[obstacle(LOGICAL, 10.65)])
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == 83
    m.obstacles_perception = [obstacle(91, 11.20)]
    m._opponent_obstacles = [obstacle(LOGICAL, 11.20)]
    deliver(m, prediction(LOGICAL))
    assert m.obstacles_prediction_id == 91


def test_the_shared_contract_still_holds():
    """Predictions and dt are stored exactly as the shared callback stores them."""
    m = machine(raw_obstacles=[obstacle(83, 10.65)],
                opponents=[obstacle(LOGICAL, 10.65)])
    data = prediction(LOGICAL, count=4, dt=0.25)
    deliver(m, data)
    assert len(m.obstacles_prediction) == 4
    assert m.prediction_dt == 0.25
