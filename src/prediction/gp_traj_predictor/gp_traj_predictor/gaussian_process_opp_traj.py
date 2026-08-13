#!/usr/bin/env python3
"""Estimate periodic opponent lateral and velocity profiles with Gaussian processes."""

import numpy as np
import rclpy
from f110_msgs.msg import OpponentTrajectory, OppWpnt, ProjOppTraj, WpntArray
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from frenet_conversion.frenet_converter import FrenetConverter
from .prediction_node import ccma_smooth, trajectory_markers


def _periodic_distance(a, b, length):
    difference = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    return np.minimum(difference, length - np.minimum(difference, length))


def _kernel(distance, length_scale, matern=False):
    scaled = distance / max(float(length_scale), 1e-3)
    if matern:
        root3 = np.sqrt(3.0)
        return (1.0 + root3 * scaled) * np.exp(-root3 * scaled)
    return np.exp(-0.5 * scaled * scaled)


def _gp_predict(train_s, train_y, query_s, track_length, length_scale,
                noise, jitter, matern=False):
    train_s = np.asarray(train_s, dtype=float) % track_length
    train_y = np.asarray(train_y, dtype=float)
    query_s = np.asarray(query_s, dtype=float) % track_length
    mean = float(np.mean(train_y))
    centered = train_y - mean
    k_train = _kernel(
        _periodic_distance(train_s, train_s, track_length), length_scale, matern)
    k_train += np.eye(len(train_s)) * (float(noise) ** 2 + float(jitter))
    k_query = _kernel(
        _periodic_distance(query_s, train_s, track_length), length_scale, matern)
    try:
        chol = np.linalg.cholesky(k_train)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, centered))
        prediction = mean + k_query @ alpha
        solved = np.linalg.solve(chol, k_query.T)
        variance = np.maximum(0.0, 1.0 - np.sum(solved * solved, axis=0))
    except np.linalg.LinAlgError:
        prediction = np.interp(query_s, np.sort(train_s), train_y[np.argsort(train_s)])
        variance = np.ones(len(query_s))
    return prediction, variance


def aggregate_training_samples(detections, track_length, bin_size):
    """Median-collapse conflicting measurements that occupy the same s bin."""
    groups = {}
    width = max(1e-3, float(bin_size))
    for point in detections:
        values = (point.s, point.d, point.vs, point.vd)
        if not np.all(np.isfinite(values)):
            continue
        s = float(point.s) % float(track_length)
        key = int(np.floor(s / width))
        groups.setdefault(key, []).append((s, *map(float, values[1:])))

    rows = []
    for values in groups.values():
        values = np.asarray(values, dtype=float)
        rows.append(tuple(np.median(values, axis=0)))
    rows.sort(key=lambda row: row[0])
    if not rows:
        return tuple(np.asarray([], dtype=float) for _ in range(4))
    columns = np.asarray(rows, dtype=float).T
    return tuple(columns[index] for index in range(4))


def nearest_observation_distance(query_s, train_s, track_length):
    """Circular distance from every query point to a real training bin."""
    if len(train_s) == 0:
        return np.full(len(query_s), np.inf)
    return np.min(_periodic_distance(query_s, train_s, track_length), axis=1)


class GaussianProcessOpponentTrajectory(Node):
    """Fit fixed-kernel periodic GPs without a scikit-learn runtime dependency."""

    def __init__(self):
        super().__init__('gaussian_process_opp_traj')
        defaults = {
            'min_training_points': 8,
            'max_training_points': 200,
            'training_s_bin_size': 0.15,
            'max_extrapolation_distance': 1.0,
            'd_length_scale': 1.0,
            'vs_length_scale': 1.5,
            'd_noise': 0.04,
            'vs_noise': 0.15,
            'gp_jitter': 1e-6,
            'min_speed': 0.0,
            'max_speed': 12.0,
            'opponent_width': 0.28,
            'boundary_margin': 0.03,
            'boundary_clipped_variance': 1.0,
            # [Hz] How often the GP is refitted. See _fit_timer.
            'fit_rate_hz': 3.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.global_msg = None
        self.converter = None
        self.track_length = None
        self.latest_projected = None
        self._pending_projected = None

        self.trajectory_pub = self.create_publisher(
            OpponentTrajectory, '/opponent_trajectory', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/opponent_traj_markerarray', 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(ProjOppTraj, '/proj_opponent_trajectory', self._projected_cb, 10)
        fit_rate = float(self.get_parameter('fit_rate_hz').value)
        self.create_timer(1.0 / max(fit_rate, 0.1), self._fit_timer)
        self.get_logger().info(
            f'GP refit rate {fit_rate:.1f} Hz, '
            f'{int(self.get_parameter("max_training_points").value)} training points max')

    def _global_cb(self, msg):
        if not msg.wpnts:
            return
        self.global_msg = msg
        self.track_length = float(msg.wpnts[-1].s_m)
        x = np.asarray([w.x_m for w in msg.wpnts], dtype=float)
        y = np.asarray([w.y_m for w in msg.wpnts], dtype=float)
        psi = np.asarray([w.psi_rad for w in msg.wpnts], dtype=float)
        self.converter = FrenetConverter(x, y, psi)

    def _projected_cb(self, msg):
        """Store the newest observation set. Fitting happens on the timer."""
        self.latest_projected = msg
        self._pending_projected = msg

    def _fit_timer(self):
        """Refit at a fixed, low rate instead of once per measurement.

        This used to be the body of _projected_cb, so a GP fit ran for every
        message opponent_trajectory published - and that node publishes on
        every accepted sample, up to its 25 Hz loop rate. Each fit is two
        Cholesky factorisations of a max_training_points square matrix plus
        two dense cross-covariance blocks against every queried waypoint, so
        the cost is cubic in a number that was set to 200.

        Measured on the car: the particle filter reported
        `iters per sec: 39, possible: 65` and was starved on 310 of 310
        samples, with all six cores at 100%. A localisation filter dropping
        40% of its scans is the failure that matters - it feeds cur_s, which
        feeds the local path the controller follows.

        Nothing downstream needs 25 Hz here. opp_prediction accepts a learned
        trajectory up to trajectory_timeout (2.0 s) old, and the opponent's
        line is a lap-scale quantity - it does not change between two
        consecutive lidar frames. 3 Hz leaves an order of magnitude of margin
        against that timeout.

        Only the newest message is kept, so a burst of measurements collapses
        into one fit rather than queueing.
        """
        msg = self._pending_projected
        self._pending_projected = None
        if msg is None:
            return
        if self.converter is None or self.global_msg is None:
            return
        minimum = int(self.get_parameter('min_training_points').value)
        if len(msg.detections) < minimum:
            return
        self._fit_publish(msg)

    def _fit_publish(self, msg):
        detections = list(msg.detections)
        maximum = int(self.get_parameter('max_training_points').value)
        if len(detections) > maximum:
            indices = np.linspace(0, len(detections) - 1, maximum).astype(int)
            detections = [detections[index] for index in indices]
        train_s, train_d, train_vs, train_vd = aggregate_training_samples(
            detections,
            self.track_length,
            self.get_parameter('training_s_bin_size').value,
        )
        if len(train_s) < int(self.get_parameter('min_training_points').value):
            return

        waypoints = self.global_msg.wpnts[:-1] or self.global_msg.wpnts
        all_query_s = np.asarray([w.s_m for w in waypoints], dtype=float)
        observed_distance = nearest_observation_distance(
            all_query_s, train_s, self.track_length)
        observed = observed_distance <= float(
            self.get_parameter('max_extrapolation_distance').value)
        full_coverage = bool(np.all(observed))
        waypoints = [
            waypoint for waypoint, keep in zip(waypoints, observed) if keep
        ]
        query_s = all_query_s[observed]
        if len(query_s) < int(self.get_parameter('min_training_points').value):
            return
        d_pred, d_var = _gp_predict(
            train_s, train_d, query_s, self.track_length,
            self.get_parameter('d_length_scale').value,
            self.get_parameter('d_noise').value,
            self.get_parameter('gp_jitter').value,
            matern=True,
        )
        vs_pred, vs_var = _gp_predict(
            train_s, train_vs, query_s, self.track_length,
            self.get_parameter('vs_length_scale').value,
            self.get_parameter('vs_noise').value,
            self.get_parameter('gp_jitter').value,
            matern=False,
        )
        vs_pred = np.clip(
            vs_pred,
            float(self.get_parameter('min_speed').value),
            float(self.get_parameter('max_speed').value),
        )
        vd_pred, _ = _gp_predict(
            train_s, train_vd, query_s, self.track_length,
            self.get_parameter('vs_length_scale').value,
            self.get_parameter('vs_noise').value,
            self.get_parameter('gp_jitter').value,
            matern=False,
        )

        clearance = (
            0.5 * float(self.get_parameter('opponent_width').value)
            + float(self.get_parameter('boundary_margin').value)
        )
        lower = np.asarray([-float(w.d_right) + clearance for w in waypoints])
        upper = np.asarray([float(w.d_left) - clearance for w in waypoints])
        valid_bounds = lower < upper
        if not np.any(valid_bounds):
            return
        waypoints = [
            waypoint for waypoint, keep in zip(waypoints, valid_bounds) if keep
        ]
        query_s = query_s[valid_bounds]
        d_pred = d_pred[valid_bounds]
        d_var = d_var[valid_bounds]
        vs_pred = vs_pred[valid_bounds]
        vs_var = vs_var[valid_bounds]
        vd_pred = vd_pred[valid_bounds]
        lower = lower[valid_bounds]
        upper = upper[valid_bounds]
        if len(query_s) < int(self.get_parameter('min_training_points').value):
            return
        raw_d_pred = d_pred.copy()
        d_pred = np.clip(d_pred, lower, upper)
        clipped = np.abs(raw_d_pred - d_pred) > 1e-6
        d_var[clipped] = np.maximum(
            d_var[clipped],
            float(self.get_parameter('boundary_clipped_variance').value),
        )

        xy = self.converter.get_cartesian(query_s, d_pred).T
        # Once a full lap exists, apply circular CCMA smoothing and convert the
        # result back through the shared UNITA converter.
        if msg.lapcount >= 1.0 and full_coverage and len(xy) >= 5:
            xy = ccma_smooth(xy, window=5)
            sd = self.converter.get_frenet(xy[:, 0], xy[:, 1])
            d_pred = np.asarray(sd[1], dtype=float)
            smoothed_raw = d_pred.copy()
            d_pred = np.clip(d_pred, lower, upper)
            smoothed_clipped = np.abs(smoothed_raw - d_pred) > 1e-6
            d_var[smoothed_clipped] = np.maximum(
                d_var[smoothed_clipped],
                float(self.get_parameter('boundary_clipped_variance').value),
            )
            xy = self.converter.get_cartesian(query_s, d_pred).T

        output = OpponentTrajectory()
        output.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        output.lap_count = float(msg.lapcount)
        output.opp_is_on_trajectory = bool(msg.opp_is_on_trajectory)
        for index, (s, d, point, vs, vd, dv, vv) in enumerate(zip(
                query_s, d_pred, xy, vs_pred, vd_pred, d_var, vs_var)):
            output.oppwpnts.append(OppWpnt(
                id=index,
                s_m=float(s),
                d_m=float(d),
                x_m=float(point[0]),
                y_m=float(point[1]),
                proj_vs_mps=float(vs),
                vd_mps=float(vd),
                d_var=float(dv),
                vs_var=float(vv),
            ))
        self.trajectory_pub.publish(output)
        max_speed = max(0.1, float(np.max(vs_pred)))
        self.marker_pub.publish(trajectory_markers(
            output.header,
            [(x, y, max(0.05, v / max_speed)) for (x, y), v in zip(xy, vs_pred)],
            'gp_opponent_trajectory',
            (1.0, 1.0, 0.0),
        ))


def main(args=None):
    rclpy.init(args=args)
    node = GaussianProcessOpponentTrajectory()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
