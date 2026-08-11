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


class GaussianProcessOpponentTrajectory(Node):
    """Fit fixed-kernel periodic GPs without a scikit-learn runtime dependency."""

    def __init__(self):
        super().__init__('gaussian_process_opp_traj')
        defaults = {
            'min_training_points': 8,
            'max_training_points': 200,
            'd_length_scale': 1.0,
            'vs_length_scale': 1.5,
            'd_noise': 0.04,
            'vs_noise': 0.15,
            'gp_jitter': 1e-6,
            'min_speed': 0.0,
            'max_speed': 12.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.global_msg = None
        self.converter = None
        self.track_length = None
        self.latest_projected = None

        self.trajectory_pub = self.create_publisher(
            OpponentTrajectory, '/opponent_trajectory', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, '/opponent_traj_markerarray', 10)
        self.create_subscription(WpntArray, '/global_waypoints', self._global_cb, 10)
        self.create_subscription(ProjOppTraj, '/proj_opponent_trajectory', self._projected_cb, 10)

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
        self.latest_projected = msg
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
        train_s = np.asarray([p.s for p in detections], dtype=float)
        train_d = np.asarray([p.d for p in detections], dtype=float)
        train_vs = np.asarray([p.vs for p in detections], dtype=float)
        train_vd = np.asarray([p.vd for p in detections], dtype=float)
        finite = np.isfinite(train_s + train_d + train_vs + train_vd)
        train_s, train_d = train_s[finite], train_d[finite]
        train_vs, train_vd = train_vs[finite], train_vd[finite]
        if len(train_s) < int(self.get_parameter('min_training_points').value):
            return

        waypoints = self.global_msg.wpnts[:-1] or self.global_msg.wpnts
        query_s = np.asarray([w.s_m for w in waypoints], dtype=float)
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
        vd_pred = np.interp(
            query_s,
            np.sort(train_s),
            train_vd[np.argsort(train_s)],
        )

        xy = self.converter.get_cartesian(query_s, d_pred).T
        # Once a full lap exists, apply circular CCMA smoothing and convert the
        # result back through the shared UNITA converter.
        if msg.lapcount >= 1.0 and len(xy) >= 5:
            xy = ccma_smooth(xy, window=5)
            sd = self.converter.get_frenet(xy[:, 0], xy[:, 1])
            d_pred = np.asarray(sd[1], dtype=float)

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
