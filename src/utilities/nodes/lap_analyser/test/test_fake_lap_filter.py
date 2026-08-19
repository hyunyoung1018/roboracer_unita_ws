"""A car standing on the finish line was completing laps several times a second.

The wrap test fires on any forward crossing of the lap boundary, and a
stationary car's Frenet s wanders across it at the odometry rate. Straight off
the car: "completed lap #1 in 0.034", "#2 in 0.029", "#3 in 0.032".

relocalizer.py already refused to re-seed on those, but by then the count, the
log file and the statistics had been written. min_lap_time_s moves the same
judgement into lap_analyser, where the lap is decided.
"""

from lap_analyser.lap_analyser import LapAnalyser


TRACK = 21.9


class FakeClock:
    """Hand-wound clock, in seconds, exposing what rclpy's Time subtraction gives."""

    def __init__(self):
        self.seconds = 0.0

    def now(self):
        clock = self

        class _Stamp:
            def __sub__(self, other):
                return _Delta(clock.seconds - other.at)
        return _Stamp()

    def stamp(self):
        return _Mark(self.seconds)


class _Delta:
    def __init__(self, seconds):
        self.nanoseconds = seconds * 1e9


class _Mark:
    def __init__(self, at):
        self.at = at


class _Logger:
    def __init__(self):
        self.warnings = []

    def warn(self, message, **_):
        self.warnings.append(message)


def analyser(min_lap_time_s=2.0, lap_count=1):
    a = LapAnalyser.__new__(LapAnalyser)
    a.MIN_LAP_TIME_S = min_lap_time_s
    a.lap_count = lap_count
    a.track_length = TRACK
    a.s_finish = 0.0
    a.last_rel = 0.0
    a._clock = FakeClock()
    a.lap_start_time = a._clock.stamp()
    a._logger = _Logger()
    a.get_clock = lambda: a._clock
    a.get_logger = lambda: a._logger
    return a


def crossing(a, at_seconds):
    """Drive one wrap of the boundary at `at_seconds`, and say if it counted."""
    a._clock.seconds = at_seconds
    a.last_rel = TRACK - 0.05
    return LapAnalyser.check_for_finish_line_pass(a, 0.02)


# ------------------------------------------------------------ what was seen
def test_the_three_laps_in_a_tenth_of_a_second_are_all_refused():
    a = analyser()
    assert crossing(a, 0.034) is False
    assert crossing(a, 0.063) is False
    assert crossing(a, 0.095) is False


def test_a_refusal_says_so():
    a = analyser()
    crossing(a, 0.034)
    assert len(a._logger.warnings) == 1
    assert 'retriggering' in a._logger.warnings[0]


# ------------------------------------------------------------ a real lap
def test_a_real_lap_still_counts():
    assert crossing(analyser(), 4.4) is True


def test_the_boundary_case_counts():
    assert crossing(analyser(), 2.0) is True


def test_just_under_the_boundary_does_not():
    assert crossing(analyser(), 1.999) is False


# ------------------------------------ the guard must not eat the lap clock
def test_a_refused_crossing_leaves_lap_start_time_alone():
    a = analyser()
    before = a.lap_start_time.at
    crossing(a, 0.5)
    assert a.lap_start_time.at == before


def test_a_real_lap_after_refused_crossings_is_timed_from_the_real_start():
    # Three retriggers at the start of the lap, then the lap itself. If a
    # refusal had reset the clock the lap would time as 4.0 s, not 4.5.
    a = analyser()
    for t in (0.10, 0.15, 0.20):
        assert crossing(a, t) is False
    a._clock.seconds = 4.5
    assert a._seconds_since_lap_start() == 4.5


# ------------------------------------------------------------ untouched paths
def test_no_wrap_is_still_no_lap():
    a = analyser()
    a._clock.seconds = 100.0
    a.last_rel = 5.0
    assert LapAnalyser.check_for_finish_line_pass(a, 5.1) is False


def test_the_first_crossing_of_the_run_is_never_too_soon():
    # lap_count -1 starts the clock rather than ending a lap.
    a = analyser(lap_count=-1)
    assert crossing(a, 0.01) is True


def test_the_guard_is_tunable():
    assert crossing(analyser(min_lap_time_s=0.5), 1.0) is True
    assert crossing(analyser(min_lap_time_s=6.0), 4.4) is False


# --------------------------------- what the time guard does NOT catch
def test_a_long_stand_on_the_line_still_reports_a_lap():
    """Documented, not asserted as desirable.

    The guard is on elapsed time alone, so a car parked on the boundary for
    longer than min_lap_time_s and then nudged across it produces a lap that
    was never driven. Catching that needs a progress test - "did the car get
    round" - which this does not do.
    """
    assert crossing(analyser(), 2.5) is True
