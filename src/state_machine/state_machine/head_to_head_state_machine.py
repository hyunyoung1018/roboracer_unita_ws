#!/usr/bin/env python3
"""Head-to-head wrapper for the legacy UNITA state machine."""

import rclpy

from .state_machine_node import StateMachine


class HeadToHeadStateMachine(StateMachine):
    """Correct only the startup semantics of ``use_force_trailing``.

    The shared state machine negates this parameter during construction, while
    its runtime callback assigns it directly. Keeping the correction here
    preserves time-trials behavior and scopes it to head-to-head launches.
    """

    def __init__(self):
        super().__init__()
        self.use_force_trailing = bool(self.params.use_force_trailing)


def main(args=None):
    rclpy.init(args=args)
    state_machine = HeadToHeadStateMachine()
    try:
        rclpy.spin(state_machine)
    except KeyboardInterrupt:
        pass
    finally:
        state_machine.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
