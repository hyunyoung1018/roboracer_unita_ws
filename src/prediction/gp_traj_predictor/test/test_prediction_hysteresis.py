from types import SimpleNamespace

from gp_traj_predictor.opp_prediction import OpponentPredictor


class FakePredictor:
    def __init__(self):
        self.track_length = 10.0
        self.trajectory = SimpleNamespace(
            oppwpnts=[
                SimpleNamespace(s_m=0.0, d_m=0.0),
                SimpleNamespace(s_m=5.0, d_m=0.0),
            ],
            lap_count=0.6,
            opp_is_on_trajectory=True,
        )
        self.trajectory_received_at = object()
        self._learned_gate_open = False
        self._learned_ready_count = 0
        self._learned_reject_count = 0
        self.params = {
            'trajectory_timeout': 2.0,
            'min_training_laps': 0.5,
            'learned_deviation_enter_threshold': 0.35,
            'learned_deviation_exit_threshold': 0.55,
            'learned_ready_confirm_frames': 3,
            'learned_reject_confirm_frames': 5,
        }

    def _age(self, _received_at):
        return 0.1

    def get_parameter(self, name):
        return SimpleNamespace(value=self.params[name])

    def _reset_learned_gate(self):
        OpponentPredictor._reset_learned_gate(self)

    def _validate_learned_profile(self, _s, _d):
        return True, None, None


def learned_status(predictor, deviation):
    obstacle = SimpleNamespace(id=7, s_center=2.0, d_center=deviation)
    return OpponentPredictor._learned_status(predictor, obstacle)


def test_gate_requires_three_good_frames_and_five_bad_frames():
    predictor = FakePredictor()

    assert learned_status(predictor, 0.34)[0] is False
    assert learned_status(predictor, 0.34)[0] is False
    assert learned_status(predictor, 0.34)[0] is True

    for _ in range(4):
        assert learned_status(predictor, 0.56)[0] is True
    ready, status, _ = learned_status(predictor, 0.56)
    assert ready is False
    assert status == 'DEVIATION_TOO_LARGE'


def test_open_gate_ignores_noise_between_enter_and_exit_thresholds():
    predictor = FakePredictor()
    for _ in range(3):
        learned_status(predictor, 0.30)

    ready, status, _ = learned_status(predictor, 0.43)

    assert ready is True
    assert status == 'LEARNED_READY'
