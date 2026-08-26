#!/usr/bin/env python3
"""Head-to-head controller node: the shared manager, building H2HController.

Separate from controller_manager.py because time_trials.launch.xml runs that
node and must keep behaving exactly as it does today. Everything the manager
does - parameters, callbacks, the drive command, the live-tuning hooks - is
inherited unchanged; only the class it instantiates and two parameters change.

See h2h_controller.py for what those two parameters buy.
"""

import rclpy

from controller.controller_manager import ControllerManager
from controller.h2h_controller import H2HController


class H2HControllerManager(ControllerManager):
    """The shared manager with the head-to-head controller in it."""

    CONTROLLER_CLASS = H2HController

    def __init__(self):
        super().__init__()
        # [bool] Keep holding the opponent's gap while OVERTAKE is the state.
        #
        # Declared here rather than in the shared manager so time trials never
        # sees the key. h2h_controller.yaml sets it true; setting it false
        # is the A/B for the whole behaviour and works while driving.
        self.trail_while_overtaking = bool(
            self._get_param('trail_while_overtaking', True))
        # [s] Ramp for the heading-gain reduction that OVERTAKE applies. 0
        # restores the shared node's single-tick step.
        self.overtake_gain_ramp_sec = float(
            self._get_param('overtake_gain_ramp_sec',
                            H2HController.GAIN_RAMP_SEC))

    def global_wpnts_cb(self, data):
        """Push the two settings in once the controller has been built.

        The controller is constructed lazily, on the first global waypoints
        message, so there is nothing to configure before this runs.
        """
        super().global_wpnts_cb(data)
        self._push_settings()

    def dyn_param_cb(self, params):
        """Let both settings be changed while driving, then defer to the base.

        The shared callback dispatches on name and silently ignores anything it
        does not know, so passing the whole list on after handling these two is
        safe and keeps every existing live parameter working.
        """
        for param in params:
            if param.name == 'trail_while_overtaking':
                self.trail_while_overtaking = bool(param.value)
                self._push_settings()
            elif param.name == 'overtake_gain_ramp_sec':
                self.overtake_gain_ramp_sec = float(param.value)
                self._push_settings()
        return super().dyn_param_cb(params)

    def _push_settings(self):
        controller = getattr(self, 'controller', None)
        if isinstance(controller, H2HController):
            controller.trail_while_overtaking = self.trail_while_overtaking
            controller.overtake_gain_ramp_sec = self.overtake_gain_ramp_sec


def main(args=None):
    rclpy.init(args=args)
    node = H2HControllerManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
