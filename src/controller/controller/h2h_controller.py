#!/usr/bin/env python3
"""Head-to-head controller: the shared one, told that OVERTAKE is not free.

This exists as a subclass rather than as edits to
:mod:`controller.combined.src.Controller` because time_trials.launch.xml runs
that class and must keep behaving exactly as it does today. Both overrides here
are inert unless the state machine reports OVERTAKE with an opponent, and time
trials has no opponent to report.

TWO THINGS CHANGE, and they are the two ways entering OVERTAKE used to be a
discontinuity.

ONE: the trailing controller keeps running.

The shared controller gates trailing on `state == "TRAILING"`, so the instant
the state machine arms an evasion the gap to the opponent stops being held and
the car takes the path's planned speed instead. Measured on 2026-08-21 over
314 s: 50 entries and 135 exits, one source change every 1.7 s, and each entry
released the car from about 2.5 m/s onto a 4 m/s path with the opponent still
two metres ahead. From outside the car that is "it was trailing, then it
suddenly accelerated into the back of the other car and stopped" - the stop
being trailing_emergency_gap once the drop put it back in TRAILING.

Lateral and longitudinal are separate questions and the stack already answers
them separately: the path says WHERE to drive, the trailing PID says HOW CLOSE
to get. Entering OVERTAKE should change the first and not the second.

TWO: the heading gain no longer steps.

The shared controller multiplies the heading gain by 0.65 while OVERTAKE is the
state. With a source change every 1.7 s that is a 1.54x step in the gain,
landing on the same tick as the path itself jumping about 0.3 m sideways - the
two compound, and on a straight, where the raceline asks for almost no steering
at all, the pair IS the steering input. Ramping between the two values over
overtake_gain_ramp_sec removes the step without changing either end.

WHAT NEITHER DOES: make the car overtake more or less readily. Both gates are
the state machine's, and neither is touched here.
"""

from controller.combined.src.Controller import Controller


class H2HController(Controller):
    """Hold the opponent's gap through an evasion, and ramp the gain into it."""

    # [s] Time constant for the heading-gain ramp between 1.0 and the
    # overtaking 0.65. About a third of the measured 1.7 s between source
    # changes, so a manoeuvre that lasts gets the full reduction while one that
    # is abandoned in 0.41 s - the measured median - barely moves the gain.
    # Zero restores the shared step exactly.
    GAIN_RAMP_SEC = 0.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start where the shared class starts: no overtake, no reduction.
        self._overtake_gain = 1.0
        # Off by default so an instance of this class behaves like its parent
        # until the manager turns it on from h2h_controller.yaml.
        self.trail_while_overtaking = False
        self.overtake_gain_ramp_sec = self.GAIN_RAMP_SEC

    def trailing_active(self):
        """Trail in OVERTAKE too, when the state machine names an opponent.

        `self.opponent` is whatever arrived on BehaviorStrategy.trailing_targets
        this tick, so this cannot invent a target: with the list empty the
        condition is false and the shared behaviour is what runs. During a
        static evasion the state machine puts the SELECTED OPPONENT there
        rather than the obstacle the path is for - see
        H2HStateMachine.get_traling_target - which is the half of this that
        decides the gap is held to the right thing.
        """
        if self.opponent is None:
            return False
        if self.state == "TRAILING":
            return True
        return bool(self.trail_while_overtaking and self.state == "OVERTAKE")

    def overtake_gain_scale(self, dt):
        """The same 0.65, reached over a ramp instead of in one tick.

        A first-order lag towards the shared value. dt comes from the control
        loop, so the ramp is in seconds rather than ticks and does not change
        with loop_rate.

        Guarded on a non-positive or non-finite dt because the caller derives
        it from message stamps, and a repeated stamp would otherwise make alpha
        zero and freeze the gain at whatever it happened to hold.
        """
        target = super().overtake_gain_scale(dt)
        ramp = float(self.overtake_gain_ramp_sec)
        if ramp <= 0.0:
            self._overtake_gain = target
            return target
        try:
            step = float(dt)
        except (TypeError, ValueError):
            step = 0.0
        if not (step > 0.0) or step != step:  # non-positive or NaN
            return self._overtake_gain
        alpha = min(1.0, step / ramp)
        self._overtake_gain += alpha * (target - self._overtake_gain)
        return self._overtake_gain
