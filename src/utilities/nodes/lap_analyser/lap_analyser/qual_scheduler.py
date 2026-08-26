#!/usr/bin/env python3

"""Race-day qualifying schedule: drive conservatively, then commit.

Qualifying is scored on the best lap, not the first, and the risk is not
uniform across a run. The first laps are the ones where the localisation has
not settled, the tyres are cold and nobody has seen the car on this track yet;
the later ones are where a time is worth taking. So the run is split in two:

    laps 1..switch_lap      phase 1 - slower, longer L1 lookahead
    laps switch_lap+1..     phase 2 - faster, shorter lookahead

Both knobs live on `sector_tuner`, which owns the speed scaling and the
per-sector L1 lookahead floor. It re-reads every parameter of its own on any
parameter event and republishes /global_waypoints_scaled and
/sector_t_clip_min, so setting them is all this node has to do - no message,
no restart, and the same values a human would drag in rqt.

The lap boundary comes from lap_analyser's `lap_data`, which is published once
per completed lap, so this node does not re-implement lap counting and can be
killed mid-run without affecting anything else. lap_analyser now counts from
the first pose estimate, so "lap 10" means ten laps of the actual run.

Everything is a parameter and every parameter is live:

    ros2 param set /qual_scheduler switch_lap 12
    ros2 param set /qual_scheduler phase2_scaling 0.65

A change to a phase already applied does not take effect on its own - re-apply
it with the service below, or wait for the next phase.

    ros2 service call /qual_scheduler/reapply std_srvs/srv/Trigger
"""

from rcl_interfaces.msg import Parameter as ParameterMsg
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Trigger

from f110_msgs.msg import LapData


class QualScheduler(Node):
    def __init__(self):
        super().__init__('qual_scheduler')

        self.declare_parameter('switch_lap', 10)
        self.declare_parameter('phase1_scaling', 0.5)
        self.declare_parameter('phase1_t_clip_min', 1.05)
        self.declare_parameter('phase2_scaling', 0.6)
        self.declare_parameter('phase2_t_clip_min', 1.10)
        # 0 or less leaves the map's own t_clip_min alone. A map that defines
        # none is a map whose controller.yaml value is the tuned one, and
        # sector_tuner only publishes the topic when every sector carries the
        # field - inventing values here would silently take that over.
        self.declare_parameter('apply_t_clip_min', True)
        self.declare_parameter('sector_tuner_node', 'sector_tuner')

        node = self.get_parameter('sector_tuner_node').value
        self.set_cli = self.create_client(SetParameters, f'/{node}/set_parameters')
        self.get_cli = self.create_client(GetParameters, f'/{node}/get_parameters')

        self.n_sectors = None
        self.phase = 0

        self.create_subscription(LapData, 'lap_data', self.lap_data_cb, 10)
        self.create_service(Trigger, '~/reapply', self.reapply_cb)

        # What is actually in force, for anything that wants to display it.
        # Latched, because the phase changes twice in a run and a viewer
        # started in between would otherwise see nothing until the next one.
        self.status_pub = self.create_publisher(
            String, 'qual_status',
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))

        # sector_tuner comes up in the same launch, so the first attempt is
        # usually too early. Retry until it answers rather than failing the run
        # on a startup race.
        self.startup_timer = self.create_timer(1.0, self._startup)

        self.get_logger().info(
            f"QualScheduler started: laps 1-{self.get_parameter('switch_lap').value} at "
            f"scaling {self.get_parameter('phase1_scaling').value} / t_clip_min "
            f"{self.get_parameter('phase1_t_clip_min').value}, then "
            f"{self.get_parameter('phase2_scaling').value} / "
            f"{self.get_parameter('phase2_t_clip_min').value}")

    # ------------------------------------------------------------------ #
    # startup                                                            #
    # ------------------------------------------------------------------ #
    def _startup(self):
        if not self.set_cli.service_is_ready() or not self.get_cli.service_is_ready():
            self.get_logger().info(
                f"waiting for {self.get_parameter('sector_tuner_node').value}...",
                throttle_duration_sec=5.0)
            return
        self.startup_timer.cancel()
        request = GetParameters.Request(names=['n_sectors'])
        self.get_cli.call_async(request).add_done_callback(self._n_sectors_cb)

    def _n_sectors_cb(self, future):
        try:
            values = future.result().values
        except Exception as exc:                       # noqa: BLE001 - report and stop
            self.get_logger().error(f"could not read n_sectors: {exc}")
            return
        if not values or values[0].type != ParameterType.PARAMETER_INTEGER:
            self.get_logger().error(
                "sector_tuner has no integer n_sectors - not touching its parameters")
            return
        self.n_sectors = int(values[0].integer_value)
        self.get_logger().info(f"sector_tuner reports {self.n_sectors} sector(s)")
        self.apply_phase(1)

    # ------------------------------------------------------------------ #
    # phases                                                             #
    # ------------------------------------------------------------------ #
    def apply_phase(self, phase):
        if self.n_sectors is None:
            self.get_logger().warn(
                f"phase {phase} is due but sector_tuner has not answered yet - "
                "retrying on the next lap")
            return False

        scaling = float(self.get_parameter(f'phase{phase}_scaling').value)
        t_clip_min = float(self.get_parameter(f'phase{phase}_t_clip_min').value)
        with_lookahead = bool(self.get_parameter('apply_t_clip_min').value)

        params = []
        for i in range(self.n_sectors):
            params.append(ParameterMsg(
                name=f'Sector{i}.scaling',
                value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                     double_value=scaling)))
            if with_lookahead and t_clip_min > 0.0:
                params.append(ParameterMsg(
                    name=f'Sector{i}.t_clip_min',
                    value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                         double_value=t_clip_min)))

        self.phase = phase
        future = self.set_cli.call_async(SetParameters.Request(parameters=params))
        future.add_done_callback(
            lambda f, p=phase, s=scaling, t=t_clip_min: self._set_done(f, p, s, t))
        return True

    def _set_done(self, future, phase, scaling, t_clip_min):
        try:
            results = future.result().results
        except Exception as exc:                       # noqa: BLE001 - report and stop
            self.get_logger().error(f"phase {phase} not applied: {exc}")
            return
        refused = [r.reason for r in results if not r.successful]
        if refused:
            self.get_logger().error(
                f"phase {phase} partly refused by sector_tuner: {refused}")
            return
        self.get_logger().warn(
            f"=== QUAL PHASE {phase}: scaling {scaling:.2f}, "
            f"t_clip_min {t_clip_min:.2f} on all {self.n_sectors} sector(s) ===")
        # Only after sector_tuner has accepted it - the overlay should show what
        # the car is driving on, not what was asked for.
        self.status_pub.publish(String(
            data=f"PHASE {phase}  x{scaling:.2f}  t{t_clip_min:.2f}"))

    # ------------------------------------------------------------------ #
    # triggers                                                           #
    # ------------------------------------------------------------------ #
    def lap_data_cb(self, msg: LapData):
        switch_lap = int(self.get_parameter('switch_lap').value)
        if self.phase >= 2:
            return
        if msg.lap_count >= switch_lap:
            self.get_logger().warn(
                f"lap {msg.lap_count} completed (switch_lap {switch_lap}) - going to phase 2")
            self.apply_phase(2)
        elif self.phase == 0:
            # sector_tuner answered late; phase 1 never landed.
            self.apply_phase(1)

    def reapply_cb(self, _request, response):
        phase = self.phase if self.phase else 1
        response.success = self.apply_phase(phase)
        response.message = (f"re-applied phase {phase}" if response.success
                            else "sector_tuner has not answered yet")
        return response


def main():
    rclpy.init()
    node = QualScheduler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
