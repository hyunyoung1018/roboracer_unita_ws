#!/usr/bin/env python3
"""
UNITA racing state machine.

Ported from the ROS1 (catkin/rospy) `state_machine` package. This is the racing
"brain": it subscribes to perception / planning / localization topics, computes a
set of boolean conditions, runs the state-transition graph and publishes the chosen
driving behaviour (local waypoints + BehaviorStrategy).

The full feature set is present (RECOVERY / START / multi-planner
sustainability / prediction-aware free checks / velocity replanning / BehaviorStrategy
trailing & overtaking targets). The race_stack ROS2 template was used only for the
ament/rclpy structural idioms.
"""
import os
import time
import json
import configparser

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

import transforms3d
from ament_index_python.packages import get_package_share_directory

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from std_msgs.msg import String, Float32, Float32MultiArray, Bool
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from f110_msgs.msg import (
    ObstacleArray,
    OTWpntArray,
    WpntArray,
    BehaviorStrategy,
    PredictionArray,
)

import trajectory_planning_helpers as tph

from vel_planner.vel_planner import calc_vel_profile
from state_machine.states_types import StateType
from state_machine import states
from state_machine import state_transitions
from state_machine.state_machine_params import StateMachineParams


try:
    # if we are in the car, vesc msgs are built and we read them
    from vesc_msgs.msg import VescStateStamped
except Exception:
    pass


class WaypointData:
    """Holds the latest waypoints of a given planner together with its (dynamic)
    parameters. In ROS1 these parameters were served by a per-planner
    `dynamic_reconfigure` server (dyn_planner_tuner.cfg). In ROS2 they are declared on
    the state-machine node as nested parameters `<planner_name>.<param>` (loaded from
    the planner yaml in this package's config/planners directory).
    """

    def __init__(self, node: "StateMachine", planner_name: str, is_closed: bool):
        self.node = node
        self.name = planner_name
        self.list = []
        self.array = None
        self.stamp = None
        self.is_init = False
        # Debug: wall-clock of the last initialize_traj (cache replacement) and
        # how many times it has happened. None/0 until the first real update.
        self.last_init_sec = None
        self.init_count = 0
        self.is_gb_track_wpnts = False
        self.is_ot_wpnts = False
        self.closest_target = None
        self.closest_gap = None
        # Debug: last _check_free_frenet decision detail (per-obstacle branch/free_dist).
        self.free_dbg = None
        self.is_closed = is_closed
        self.vel_planner_safety_factor = 1.0
        # When frozen, the cache is NOT re-initialized from fresh planner output; the
        # path captured on entry is kept and only sliced (tail trimmed) as the car
        # advances. Used to hold one blended-recovery path while trailing on it, so the
        # controller target stops jumping every frame (see _hold_recovery_freeze).
        self.frozen = False
        self.max_speed_mps = None
        self.update_param()

    def update_param(self):
        get = self.node.get_planner_param
        self.min_horizon = get(self.name, "min_horizon")
        # A planner nothing publishes is not a planner that is failing. Set
        # `enabled: false` in its yaml and the state machine stops offering
        # the state at all, rather than asking every tick and logging that
        # the topic is still empty.
        enabled = get(self.name, "enabled")
        self.enabled = True if enabled is None else bool(enabled)

        self.max_horizon = get(self.name, "max_horizon")
        self.lateral_width_m = get(self.name, "lateral_width_m")
        self.free_scaling_reference_distance_m = get(self.name, "free_scaling_reference_distance_m")
        self.latest_threshold = get(self.name, "latest_threshold")
        self.on_spline_front_horizon_thres_m = get(self.name, "on_spline_front_horizon_thres_m")
        self.on_spline_min_dist_thres_m = get(self.name, "on_spline_min_dist_thres_m")
        self.hyst_timer_sec = get(self.name, "hyst_timer_sec")
        self.killing_timer_sec = get(self.name, "killing_timer_sec")

        # Speed limits for this planner's path. Both optional: only overridden
        # when the planner's yaml carries the key, so recovery keeps the 0.5
        # safety factor it is given in code and every other planner keeps 1.0.
        sf = get(self.name, "vel_planner_safety_factor")
        if sf is not None:
            self.vel_planner_safety_factor = float(sf)
        ms = get(self.name, "max_speed_mps")
        self.max_speed_mps = float(ms) if ms is not None and float(ms) > 0.0 else None

    def initialize_traj(self, wpnt):
        if len(wpnt.wpnts) != 0:
            self.stamp = wpnt.header.stamp
            self.list = wpnt.wpnts
            self.array = np.array([[w.x_m, w.y_m, w.s_m, w.d_m] for w in wpnt.wpnts])
            self.is_init = True
            # Debug: when this cache was last replaced with fresh planner output.
            # Lets the loop snapshot report cache staleness (wall-clock since the
            # last real re-init) independently of the message header stamp.
            self.last_init_sec = self.node.now_sec()
            self.init_count += 1


def time_to_float(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class StateMachine(Node):
    """
    This state machine subscribes to topics and calculates flags/conditions.
    State transitions and state behaviors are described in `transitions.py` and `states.py`
    """

    def __init__(self) -> None:
        super().__init__(
            "state_machine",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = "state_machine"

        self.main_loop = None  # set later, referenced by params callback

        # Load planner configs (planner_name -> {param: value}) before declaring params
        self._planner_param_cache = {}
        self._load_planner_configs()

        # PARAMETER DECLARATION (replaces rospy.get_param + dyn_reconfigure)
        self.params = StateMachineParams(self)
        self.add_on_set_parameters_callback(self.params.parameters_callback)

        # Convenience aliases (kept as attributes for parity with the ROS1 code which
        # read these directly off `self`). They mirror self.params.* values.
        self.rate_hz = self.params.rate_hz
        self.n_loc_wpnts = self.params.n_loc_wpnts
        self.timetrials_only = self.params.timetrials_only
        self.config_dir = self.params.config_dir
        self.map_dir = self.params.map_dir
        if not self.config_dir:
            self.get_logger().warn(
                "config_dir is unset: no vehicle params, velocity replanning degraded")
        self.ot_planner = self.params.ot_planner
        self.track_length = self.params.track_length
        self.volt_threshold = self.params.volt_threshold

        self.local_wpnts = WpntArray()
        self.waypoints_dist = 0.1  # [m]
        self.measuring = self.params.measuring

        # sectors: read the map yamls at startup, live-update from the sector tuner nodes
        # (ROS1: /map_params + /ot_map_params and the dyn_sector_* servers)
        self.map_name = self._get_str_param("map", "")
        self.sectors_params = {}
        self.ot_sectors_params = {}
        self.only_ftg_zones = []
        self.ftg_counter = 0

        self.cur_s = 0.0
        self.cur_d = 0.0
        self.cur_vs = 0.0

        # Velocity Planning - load racecar config from stack_master
        self._load_vehicle_dynamics()

        # overtaking variables
        self.n_ot_sectors = 0
        self.overtake_wpnts = None
        self.overtake_zones = []

        # read the map sector yamls, then build only_ftg_zones / overtake_zones
        self._load_sector_yamls()
        self._load_sector_params()
        self.cur_volt = 11.69  # default value for sim
        self.static_overtaking_mode = False
        # Per-loop slice diagnostics, set by get_splini_wpts / get_recovery_wpts,
        # reset at the top of loop() so a snapshot only shows the source actually used.
        self._splini_dbg = None
        self._recovery_dbg = None
        # Previous loop's source cache, for rule 2 (drop the cache on a real src change).
        self._prev_src_cache = None

        # waypoint variables
        self.cur_id_ot = 1
        self.max_speed = -1
        self.max_s = 0
        self.current_position = None
        self.gb_wpnts = None
        self.recovery_wpnts = None
        self._recovery_plain = None
        self._recovery_blended = None
        self.gb_max_idx = None
        self.wpnt_dist = self.waypoints_dist
        self.num_glb_wpnts = 0
        self.num_ot_points = 0
        self.previous_index = 0

        # dynamic-parameter-backed attributes (aliases onto params)
        self.gb_ego_width_m = self.params.gb_ego_width_m
        self.gb_horizon_m = self.params.gb_horizon_m
        self.interest_horizon_m = self.params.interest_horizon_m
        self.static_overtake_max_speed_mps = self.params.static_overtake_max_speed_mps
        self.static_overtake_better_by_m = self.params.static_overtake_better_by_m
        self.static_overtake_min_clearance_m = (
            self.params.static_overtake_min_clearance_m)
        self.overtake_min_closing_mps = self.params.overtake_min_closing_mps

        self.last_recovery_update_time = None
        self.cur_gb_wpnts = WaypointData(self, "global_tracking", True)
        self.cur_recovery_wpnts = WaypointData(self, "recovery_planner", False)
        self.cur_avoidance_wpnts = WaypointData(self, "dynamic_avoidance_planner", False)
        self.cur_static_avoidance_wpnts = WaypointData(self, "static_avoidance_planner", False)
        self.cur_start_wpnts = WaypointData(self, "start_planner", False)

        self.cur_avoidance_wpnts.is_ot_wpnts = True
        self.cur_static_avoidance_wpnts.is_ot_wpnts = True
        self.cur_gb_wpnts.is_gb_track_wpnts = True
        self.cur_recovery_wpnts.vel_planner_safety_factor = 0.5

        self.gb_closest_target = None
        self.gb_closest_gap = None
        self.recovery_closest_target = None
        self.recovery_closest_gap = None
        self.ot_closest_target = None
        self.ot_closest_gap = None

        self.behavior_strategy = BehaviorStrategy()

        # mincurv spline
        self.mincurv_spline_x = None
        self.mincurv_spline_y = None
        # ot spline
        self.ot_spline_x = None
        self.ot_spline_y = None
        self.ot_spline_d = None
        self.recompute_ot_spline = True
        # live sector retune from the sector tuner nodes (after recompute_ot_spline exists)
        self._setup_sector_live_update()

        # obstacle avoidance variables
        self.obstacles = []
        self.obstacles_in_interest = []
        self.cur_obstacles_in_interest = []
        self.obstacles_perception = []
        self.obstacles_prediction_id = None
        self.obstacles_prediction = []
        self.prediction_dt = 0.02  # updated from PredictionArray.dt; matches predictor
        self.ego_prediction = []
        self.obstacle_was_here = True
        self.side_by_side_threshold = 0.6
        self.merger = None
        self.force_trailing = False
        self.use_force_trailing = not self.params.use_force_trailing

        # spliner variables
        self.splini_ttl = self.params.splini_ttl
        self.splini_ttl_counter = int(self.splini_ttl * self.rate_hz)
        self.avoidance_wpnts = None
        self.static_avoidance_wpnts = None
        self.start_wpnts = None
        self.start_wpnts_array = None
        self.last_valid_avoidance_wpnts = None
        self.last_valid_avoidance_array = None
        self.last_valid_static_avoidance_wpnts = None

        self.overtaking_horizon_m = self.params.overtaking_horizon_m
        self.lateral_width_ot_m = self.params.lateral_width_ot_m
        self.splini_hyst_timer_sec = self.params.splini_hyst_timer_sec
        self.emergency_break_horizon = self.params.emergency_break_horizon
        self.emergency_break_d = 0.12  # [m]

        # Graph based variables
        self.graph_based_wpts = None
        self.gb_wpnts_arr = None
        # Frenet variables
        self.frenet_wpnts = WpntArray()

        # FTG params
        self.ftg_speed_mps = self.params.ftg_speed_mps
        self.ftg_timer_sec = self.params.ftg_timer_sec
        self.ftg_disabled = not self.params.ftg_active

        # Force GBTRACK state
        self.force_raceline_state = self.params.force_RACELINE

        self.overtaking_ttl_sec = self.params.overtaking_ttl_sec
        self.overtaking_ttl_count = 0
        self.overtaking_ttl_count_threshold = int(self.overtaking_ttl_sec * self.rate_hz)
        # Grace window (in loops) during which the OT-blended recovery path is allowed
        # as the recovery source. The blended path (OT heading -> GB) only makes sense
        # when leaving OVERTAKE; outside that window plain recovery is used, so a car
        # that never overtook (OT sector off) never trails on the blended OT line.
        # Set to a positive count while in OVERTAKE and decremented each loop after.
        self.blended_recovery_grace_loops = int(0.5 * self.rate_hz)
        self._blended_grace_count = 0

        self.save_start_traj = False
        self.cur_start_wpnts_candidate = OTWpntArray()
        self.need_start_traj = False
        # visualization variables
        self.first_visualization = True
        self.x_viz = 0
        self.y_viz = 0

        # STATES
        self.cur_state = StateType.RACELINE
        self.local_wpnts_src = StateType.RACELINE
        self.static_avoid = False
        self.fail_trailing = False

        self.states = {
            StateType.RACELINE: states.RacelineTracking,
            StateType.OVERTAKE: states.Overtaking,
            StateType.FTGONLY: states.FTGOnly,
            StateType.RECOVERY: states.RECOVERY,
            StateType.START: states.START,
        }
        self.state_transitions = {
            StateType.RACELINE: state_transitions.RacelineTrackingTransition,
            StateType.RECOVERY: state_transitions.RecoveryTransition,
            StateType.TRAILING: state_transitions.TrailingTransition,
            StateType.ATTACK: state_transitions.TrailingTransition,
            StateType.OVERTAKE: state_transitions.OvertakingTransition,
            StateType.FTGONLY: state_transitions.FTGOnlyTransition,
            StateType.START: state_transitions.StartTransition,
        }

        self.opponent = ObstacleArray()

        qos = QoSProfile(depth=10)

        # SUBSCRIPTIONS
        self.create_subscription(Odometry, "/car_state/odom", self.odom_cb, qos)
        self._wait_for_attr("current_position", "/car_state/odom")

        self.create_subscription(WpntArray, "/global_waypoints_scaled", self.glb_wpnts_cb, qos)
        self.create_subscription(WpntArray, "/planner/recovery/wpnts", self.recovery_wpnts_cb, qos)
        self.create_subscription(
            WpntArray, "/planner/ot_blended_recovery/wpnts", self.ot_blended_recovery_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints/overtaking", self.overtake_cb, qos)
        self._wait_for_attr("gb_wpnts", "/global_waypoints_scaled")
        self._wait_for_attr("overtake_wpnts", "/global_waypoints/overtaking")

        self.create_subscription(Odometry, "/car_state/odom_frenet", self.frenet_pose_cb, qos)
        self.create_subscription(WpntArray, "/global_waypoints", self.glb_wpnts_og_cb, qos)

        self.create_subscription(ObstacleArray, "/tracking/obstacles", self.obstacle_perception_cb, qos)
        self.create_subscription(
            PredictionArray, "/opponent_prediction/obstacles_pred", self.obstacle_prediction_cb, qos
        )
        self.create_subscription(PredictionArray, "/mpc_controller/ego_prediction", self.ego_prediction_cb, qos)

        if self.ot_planner == "spliner" or self.ot_planner == "sqp" or self.ot_planner == "lane_change":
            self.create_subscription(OTWpntArray, "/planner/avoidance/otwpnts", self.avoidance_cb, qos)
            if self.ot_planner == "sqp" or self.ot_planner == "lane_change":
                self.create_subscription(
                    OTWpntArray, "/planner/avoidance/static_otwpnts", self.static_avoidance_cb, qos
                )
        if self.ot_planner == "sqp" or self.ot_planner == "lane_change":
            self.create_subscription(Float32MultiArray, "/planner/avoidance/merger", self.merger_cb, qos)
            self.create_subscription(Bool, "/opponent_prediction/force_trailing", self.force_trailing_cb, qos)
            self.create_subscription(Bool, "planner/avoidance/fail_trailing", self.fail_trailing_cb, qos)

        if not self.params.sim:
            self.create_subscription(VescStateStamped, "/vesc/sensors/core", self.vesc_state_cb, qos)

        self.create_subscription(OTWpntArray, "/planner/start_wpnts", self.start_wpnts_cb, qos)
        self.create_subscription(Bool, "/save_start_traj", self.save_start_traj_cb, qos)

        # PUBLICATIONS
        self.behavior_strategy_pub = self.create_publisher(BehaviorStrategy, "behavior_strategy", 1)
        self.trailing_marker_pub = self.create_publisher(Marker, "/state_machine/trailing_target", 10)
        self.overtaking_marker_pub = self.create_publisher(Marker, "/state_machine/overtaking_target", 10)
        self.loc_wpnt_pub = self.create_publisher(WpntArray, "local_waypoints", 1)
        self.vis_loc_wpnt_pub = self.create_publisher(MarkerArray, "local_waypoints/markers", 10)
        self.state_pub = self.create_publisher(String, "state_machine", 1)
        # Per-loop diagnostic snapshot (JSON) for offline/live debugging of the
        # local_wpnts source selection and stale-cache leaks.
        self.debug_pub = self.create_publisher(String, "/state_machine/debug", 10)
        self.state_mrk = self.create_publisher(Marker, "/state_marker", 10)
        self.emergency_pub = self.create_publisher(Marker, "/emergency_marker", 5)
        self.ot_section_check_pub = self.create_publisher(Bool, "/ot_section_check", 1)
        # "left" / "right" / "auto" for the car's current s. Transient-local so a
        # planner that starts late still gets the current answer.
        self.ot_preferred_side_pub = self.create_publisher(
            String, "/ot_preferred_side",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        # ROS1 published this from dynamic_statemachine_server when the save_start_traj
        # rqt button was pressed; re-homed here as a momentary param (see loop()).
        self.save_start_traj_pub = self.create_publisher(Bool, "/save_start_traj", 1)
        self._save_start_traj_requested = False
        self._save_params_requested = False
        if self.measuring:
            self.latency_pub = self.create_publisher(Float32, "/state_machine/latency", 10)

        # Markers are for people, and people do not need 50 Hz. Profiling put
        # 81% of loop() inside _pub_local_wpnts, almost all of it constructing
        # and serialising visualization_msgs/Marker - a Marker in Humble carries
        # a CompressedImage, a MeshFile and three arrays, so building one costs
        # about 100 us before anything is drawn. Everything the car needs
        # (behavior_strategy, local_waypoints, the state string) still goes out
        # every tick; only the drawing is throttled.
        # has_parameter first: the node is built with
        # automatically_declare_parameters_from_overrides, so anything in the
        # yaml is already declared and declaring it again raises.
        if not self.has_parameter("viz_rate_hz"):
            self.declare_parameter("viz_rate_hz", 10.0)
        self.viz_rate_hz = float(self.get_parameter("viz_rate_hz").value)
        self._last_viz_sec = 0.0
        self._throttle_state = {}

        # MAIN LOOP at fixed rate
        self.main_loop = self.create_timer(1.0 / self.rate_hz, self.loop)

    def _throttled_info(self, key, msg, period=1.0):
        """Rate-limit a message per KEY rather than per call site.

        rclpy's throttle_duration_sec is keyed on where the call is written, so
        one line of code shared by several planners throttles across all of
        them: the raceline check and the static-avoidance check are the same
        statement with a different wpnts_data, and for a whole second only
        whichever ran first is printed. The other is dropped silently.

        That is precisely the message that explains a refusal - the log said
        "[global_tracking] blocked by obs 1" and "path_free=False" with no line
        naming what blocked the avoidance path, because the raceline check had
        already spent the second.
        """
        now = self.now_sec()
        if now - self._throttle_state.get(key, -1e9) < period:
            return
        self._throttle_state[key] = now
        self.get_logger().info(msg)

    def _viz_due(self):
        """True when this tick should draw. Rate-limited, and skipped entirely
        when nothing is subscribed - with RViz closed the markers cost nothing
        at all, which is how the car should normally race."""
        if self.viz_rate_hz <= 0.0:
            return False
        if (self.vis_loc_wpnt_pub.get_subscription_count() == 0 and
                self.state_mrk.get_subscription_count() == 0 and
                self.overtaking_marker_pub.get_subscription_count() == 0 and
                self.trailing_marker_pub.get_subscription_count() == 0):
            return False
        now = self.now_sec()
        if now - self._last_viz_sec < 1.0 / self.viz_rate_hz:
            return False
        self._last_viz_sec = now
        return True

    # ---------------------------------------------------------------------- #
    # SETUP HELPERS                                                           #
    # ---------------------------------------------------------------------- #
    def _wait_for_attr(self, attr, topic):
        """rclpy equivalent of rospy.wait_for_message."""
        while rclpy.ok() and getattr(self, attr, None) is None:
            self.get_logger().info(f"Waiting for message on {topic}", throttle_duration_sec=1.0)
            rclpy.spin_once(self, timeout_sec=0.1)

    def _load_planner_configs(self):
        """Load the per-planner yaml files shipped in this package's config/planners dir
        and declare them as nested ROS2 parameters (<planner>.<key>)."""
        import yaml

        try:
            share = get_package_share_directory("state_machine")
        except Exception:
            share = None

        planner_names = [
            "global_tracking",
            "recovery_planner",
            "dynamic_avoidance_planner",
            "static_avoidance_planner",
            "start_planner",
        ]
        for pname in planner_names:
            data = {}
            if share is not None:
                cfg = os.path.join(share, "config", "planners", pname + ".yaml")
                if os.path.exists(cfg):
                    with open(cfg, "r") as f:
                        data = yaml.safe_load(f) or {}
            self._planner_param_cache[pname] = data
            for key, val in data.items():
                pname_param = f"{pname}.{key}"
                try:
                    self.declare_parameter(pname_param, val)
                except Exception:
                    pass

    def _get_str_param(self, name, default=""):
        try:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            v = self.get_parameter(name).value
            return v if v is not None else default
        except Exception:
            return default

    def _load_sector_yamls(self):
        # read the map sector yamls into sectors_params / ot_sectors_params (ROS1 /map_params, /ot_map_params)
        import yaml
        maps_dir = self.map_dir
        if not maps_dir:
            self.get_logger().warn(f"[{self.name}] map_dir is unset; no sectors loaded")
            return
        sp = os.path.join(maps_dir, "speed_scaling.yaml")
        if os.path.exists(sp):
            with open(sp, "r") as f:
                d = yaml.safe_load(f) or {}
            # Current ROS 2 maps use the launched node name (sector_tuner) as
            # their YAML root.  Keep accepting the legacy ROS1-port name so
            # older map directories remain usable.
            block = d.get("sector_tuner") or d.get("speed_sector_tuner") or {}
            self.sectors_params = block.get("ros__parameters", {}) or {}
        else:
            self.get_logger().warn(f"[{self.name}] {sp} not found; no FTG-only zones")
        # The map's overtaking sectors unless something asked for another
        # file. See state_machine_params.ot_sectors_file - the split exists so
        # ot_flag / preferred_side / yeet_factor stay head-to-head's.
        op = getattr(self.params, "ot_sectors_file", "") or os.path.join(
            maps_dir, "ot_sectors.yaml")
        if os.path.exists(op):
            with open(op, "r") as f:
                d = yaml.safe_load(f) or {}
            # ot_sectors.yaml is loaded by the /ot_interpolator ROS 2 node.
            # Accept the old name as a fallback, but prefer the actual node
            # name used by every map currently shipped in this workspace.
            block = d.get("ot_interpolator") or d.get("ot_sector_tuner") or {}
            self.ot_sectors_params = block.get("ros__parameters", {}) or {}
        else:
            self.get_logger().warn(f"[{self.name}] {op} not found; no overtake zones")

    def _load_sector_params(self):
        # build zones from the sector dicts (ROS1 sector_dyn_param_cb / ot_dyn_param_cb)
        self.only_ftg_zones = []
        self.n_sectors = int(self.sectors_params.get("n_sectors", 0))
        for i in range(self.n_sectors):
            sec = self.sectors_params.get(f"Sector{i}", {}) or {}
            if sec.get("only_FTG", False):
                # end+1 == next sector's start: close the 1-index gap so adjacent FTG
                # sectors don't briefly drop to RACELINE (ROS1 used [start, end]).
                self.only_ftg_zones.append([sec.get("start", 0), sec.get("end", 0) + 1])

        self.overtake_zones = []
        # [[start, end, side], ...] over the SAME index range as overtake_zones.
        # side is "left", "right" or "auto"; auto means the planner decides, which
        # is what every map does until one says otherwise.
        self.overtake_sides = []
        self.n_ot_sectors = int(self.ot_sectors_params.get("n_sectors", 0))
        for i in range(self.n_ot_sectors):
            sec = self.ot_sectors_params.get(f"Overtaking_sector{i}", {}) or {}
            if sec.get("ot_flag", False):
                bounds = [sec.get("start", 0), sec.get("end", 0) + 1]
                self.overtake_zones.append(bounds)
                side = str(sec.get("preferred_side", "auto")).strip().lower()
                if side not in ("left", "right", "auto"):
                    self.get_logger().warn(
                        f"[{self.name}] Overtaking_sector{i}: preferred_side "
                        f"'{side}' is not left/right/auto - using auto")
                    side = "auto"
                self.overtake_sides.append(bounds + [side])

    def _setup_sector_live_update(self):
        # ROS2 replacement of ROS1 /dyn_sector_speed & /dyn_sector_overtake subscriptions
        try:
            from rclpy.parameter_event_handler import ParameterEventHandler
        except ImportError:
            # Humble does not ship it; utilities/libraries carries the backport.
            from parameter_event_handler.parameter_event_handler import ParameterEventHandler
        self._sector_evt_handler = ParameterEventHandler(self)
        self._sector_evt_cb_handle = self._sector_evt_handler.add_parameter_event_callback(
            self._sector_param_event_cb)

    @staticmethod
    def _param_msg_value(p):
        # rcl_interfaces/Parameter -> python value (bool/int/double only needed here)
        t = p.value.type
        if t == 1:
            return p.value.bool_value
        if t == 2:
            return p.value.integer_value
        if t == 3:
            return p.value.double_value
        return None

    def _sector_param_event_cb(self, event):
        node = event.node.lstrip("/")
        if node in ("sector_tuner", "speed_sector_tuner"):
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Sector") and p.name.endswith(".only_FTG"):
                    key = p.name.split(".")[0]
                    self.sectors_params.setdefault(key, {})["only_FTG"] = bool(self._param_msg_value(p))
            self._load_sector_params()
        elif node in ("ot_interpolator", "ot_sector_tuner"):
            for p in list(event.new_parameters) + list(event.changed_parameters):
                if p.name.startswith("Overtaking_sector") and p.name.endswith(".ot_flag"):
                    key = p.name.split(".")[0]
                    self.ot_sectors_params.setdefault(key, {})["ot_flag"] = bool(self._param_msg_value(p))
                    self.recompute_ot_spline = True
                elif (p.name.startswith("Overtaking_sector")
                        and p.name.endswith(".preferred_side")):
                    # Live, because which way to go is the kind of thing that
                    # is decided by watching the car try it.
                    key = p.name.split(".")[0]
                    self.ot_sectors_params.setdefault(key, {})["preferred_side"] = \
                        str(p.value.string_value)
                self._sector_param_extra(p)
            self._load_sector_params()

    def _sector_param_extra(self, p):
        """Seam for a subclass that owns more of ot_sectors.yaml than this one.

        Called once per changed /ot_interpolator parameter, after the ot_flag
        and preferred_side branches above have had their turn. yeet_factor is
        head-to-head's and lives in h2h_state_machine; nothing here reads it.
        """
        return

    def get_planner_param(self, planner_name, key):
        """Read a planner parameter; falls back to cached yaml value."""
        full = f"{planner_name}.{key}"
        if self.has_parameter(full):
            return self.get_parameter(full).value
        return self._planner_param_cache.get(planner_name, {}).get(key)

    def _load_vehicle_dynamics(self):
        """Load veh params + ggv / ax_max machine info from stack_master config."""
        self.pars = {}
        parser = configparser.ConfigParser()
        ini_ok = False
        if self.config_dir:
            ini_ok = bool(parser.read(os.path.join(self.config_dir, "racecar_f110.ini")))

        if not ini_ok:
            # Sim / missing config fallback: provide sane defaults so the node still runs.
            self.get_logger().warn(
                "racecar_f110.ini not found; using default vehicle params (velocity replanning degraded)"
            )
            self.pars["veh_params"] = {
                "v_max": 7.0, "length": 0.535, "width": 0.3,
                "mass": 3.5, "dragcoeff": 0.0136, "g": 9.81,
            }
            self.pars["vel_calc_opts"] = {"dyn_model_exp": 1.0, "vel_profile_conv_filt_window": None}
            self.ggv = None
            self.ax_max_machines = None
            self.b_ax_max_machines = None
            return

        self.pars["veh_params"] = json.loads(parser.get("GENERAL_OPTIONS", "veh_params"))
        self.pars["vel_calc_opts"] = json.loads(parser.get("GENERAL_OPTIONS", "vel_calc_opts"))
        vdyn = os.path.join(self.config_dir, "veh_dyn_info")
        ggv_path = os.path.join(vdyn, "ggv.csv")
        ax_max_path = os.path.join(vdyn, "ax_max_machines.csv")
        b_ax_max_path = os.path.join(vdyn, "b_ax_max_machines.csv")
        self.ggv, self.ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
            ggv_import_path=ggv_path, ax_max_machines_import_path=ax_max_path
        )
        # Braking limits are optional. Without this guard a missing csv raises
        # FileNotFoundError out of the constructor, and this node dying takes
        # the whole drive chain with it: no /behavior_strategy means the
        # controller never publishes and the car cannot be driven at all. Fall
        # back to the acceleration table, which is the conservative direction -
        # this car brakes at least as hard as it accelerates.
        try:
            _, self.b_ax_max_machines = tph.import_veh_dyn_info.import_veh_dyn_info(
                ggv_import_path=ggv_path, ax_max_machines_import_path=b_ax_max_path
            )
        except Exception as exc:
            self.get_logger().warn(
                f"{b_ax_max_path} unreadable ({exc}); using the acceleration "
                "limits for braking too")
            self.b_ax_max_machines = self.ax_max_machines

    def now_sec(self) -> float:
        return time_to_float(self.get_clock().now().to_msg())

    #############
    # CALLBACKS #
    #############
    def save_start_traj_cb(self, msg):
        if len(self.cur_start_wpnts_candidate.wpnts) != 0:
            self.update_velocity(self.cur_start_wpnts_candidate, self.cur_start_wpnts.vel_planner_safety_factor)
            self.cur_start_wpnts.initialize_traj(self.cur_start_wpnts_candidate)
            self.cur_state = StateType.START

    def vesc_state_cb(self, data):
        self.cur_volt = data.state.voltage_input

    def frenet_planner_cb(self, data: WpntArray):
        self.frenet_wpnts = data

    def recovery_wpnts_cb(self, data: WpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor)
        self._recovery_plain = data
        self._select_recovery_source()

    def ot_blended_recovery_cb(self, data: WpntArray):
        # OT-blended recovery: OT heading for the first ~1 m then splined to GB.
        # Published every loop by recovery_spliner (empty when no OT path). When it
        # carries a valid path we prefer it over plain recovery so the RECOVERY src
        # keeps the overtake line instead of snapping straight to GB.
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_recovery_wpnts.vel_planner_safety_factor)
        self._recovery_blended = data
        self._select_recovery_source()

    def _select_recovery_source(self):
        # recovery_wpnts feeds _check_latest_wpnts (freshness + on-spline). Prefer the
        # blended path when it is fresh and non-empty, else fall back to plain recovery.
        # While the recovery cache is frozen (held path in use), don't swap the source
        # from under it -- the freeze in _check_latest_wpnts keeps the captured cache.
        if self.cur_recovery_wpnts.frozen:
            return
        # The blended path is only meaningful when leaving OVERTAKE: it keeps the OT
        # heading for ~1 m then splines back to GB. Outside the post-OVERTAKE grace
        # window it must NOT stand in as the recovery source, otherwise a car that is
        # merely trailing (OT sector off, OVERTAKE never entered) would follow the OT
        # line whenever it drifts off the raceline. Fall back to plain recovery then.
        allow_blended = self.cur_state == StateType.OVERTAKE or self._blended_grace_count > 0
        blended = self._recovery_blended
        if allow_blended and blended is not None and len(blended.wpnts) != 0 and (
            self.now_sec() - time_to_float(blended.header.stamp)
        ) <= self.cur_recovery_wpnts.latest_threshold:
            self.recovery_wpnts = blended
        else:
            self.recovery_wpnts = self._recovery_plain

    def _hold_recovery_freeze(self):
        # Called once per loop after the src is decided. Freeze the recovery cache while
        # RECOVERY is the active source; release it the moment the src leaves RECOVERY so
        # the next entry captures a fresh path.
        if self.local_wpnts_src == StateType.RECOVERY:
            # On entry the cache was just re-inited with fresh output (frozen was False
            # during this loop's transition); now latch it so later loops hold it.
            self.cur_recovery_wpnts.frozen = True
        else:
            self.cur_recovery_wpnts.frozen = False

    def avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            # No lift here: the raceline is a hard ceiling. h2h_state_machine
            # overrides this to spend yeet_factor on the lane change, which is
            # the only path that needs a speed advantage over the line it
            # departs from - and the only mode that has one.
            self.update_velocity(data, self.cur_avoidance_wpnts.vel_planner_safety_factor,
                                 self.cur_avoidance_wpnts.max_speed_mps)
        self.avoidance_wpnts = data

    def static_avoidance_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.update_velocity(data, self.cur_static_avoidance_wpnts.vel_planner_safety_factor,
                                 self.cur_static_avoidance_wpnts.max_speed_mps)
        self.static_avoidance_wpnts = data

    def start_wpnts_cb(self, data: OTWpntArray):
        if len(data.wpnts) != 0:
            self.cur_start_wpnts_candidate = data

    def overtake_cb(self, data):
        self.overtake_wpnts = data.wpnts
        self.num_ot_points = len(self.overtake_wpnts)
        if self.recompute_ot_spline and self.num_ot_points != 0:
            self.ot_splinification()
            self.recompute_ot_spline = False

    def glb_wpnts_cb(self, data: WpntArray):
        # last point's s == loop length (ROS1 read this from /global_republisher/track_length)
        track_len = data.wpnts[-1].s_m
        data.wpnts = data.wpnts[:-1]  # exclude last point (== first)
        self.gb_wpnts = data
        self.num_glb_wpnts = len(data.wpnts)
        self.n_loc_wpnts = min(self.n_loc_wpnts, int(self.num_glb_wpnts / 2))
        self.max_s = data.wpnts[-1].s_m
        if track_len > 1.0:
            self.track_length = track_len
        self.wpnt_dist = data.wpnts[1].s_m - data.wpnts[0].s_m
        self.gb_max_idx = data.wpnts[-1].id
        if self.ot_planner == "graph_based":
            self.gb_wpnts_arr = np.array([
                [w.s_m, w.d_m, w.x_m, w.y_m, w.d_right, w.d_left, w.psi_rad,
                 w.kappa_radpm, w.vx_mps, w.ax_mps2] for w in data.wpnts
            ])

    def glb_wpnts_og_cb(self, data):
        if self.max_speed == -1:
            self.max_speed = max([wpnt.vx_mps for wpnt in data.wpnts])

    def graphbased_wpts_cb(self, data):
        arr = np.asarray(data.data)
        self.graph_based_wpts = arr.reshape(data.layout.dim[0].size, data.layout.dim[1].size)
        self.graph_based_action = data.layout.dim[0].label

    def obstacle_perception_cb(self, data):
        if not self.timetrials_only:
            self.obstacles_perception = data.obstacles[:]
            self.obstacles = data.obstacles
            obstacles_in_interest = []
            for obs in data.obstacles:
                gap = (obs.s_start - self.cur_s) % self.track_length
                if gap < self.interest_horizon_m:
                    obstacles_in_interest.append(obs)
            self.obstacles_in_interest = obstacles_in_interest

    def ego_prediction_cb(self, data):
        self.ego_prediction = data.predictions if len(data.predictions) != 0 else []

    def obstacle_prediction_cb(self, data):
        if len(data.predictions) != 0:
            self.obstacles_prediction_id = data.id
            self.obstacles_prediction = data.predictions
            # Time step between consecutive predictions, carried on the message so the
            # ttc->prediction-index conversion in _check_free_frenet stays in sync with
            # the predictor's dt (falls back to the last known dt if a msg omits it).
            if data.dt > 0.0:
                self.prediction_dt = data.dt
        else:
            self.obstacles_prediction = []

    def frenet_pose_cb(self, data: Odometry):
        self.cur_s = data.pose.pose.position.x
        self.cur_d = data.pose.pose.position.y
        self.cur_vs = data.twist.twist.linear.x
        if self.num_ot_points != 0:
            self.cur_id_ot = int(self._find_nearest_ot_s())

    def odom_cb(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        q = data.pose.pose.orientation
        # transforms3d uses [w, x, y, z]
        _, _, theta = transforms3d.euler.quat2euler([q.w, q.x, q.y, q.z])
        self.current_position = [x, y, theta]

    def merger_cb(self, data):
        self.merger = data.data

    def force_trailing_cb(self, data):
        self.force_trailing = data.data if self.use_force_trailing else False

    def fail_trailing_cb(self, data):
        self.fail_trailing = data.data

    ######################################
    # ATTRIBUTES/CONDITIONS CALCULATIONS #
    ######################################
    def _check_only_ftg_zone(self) -> bool:
        ftg_only = False
        if len(self.only_ftg_zones) != 0:
            for sector in self.only_ftg_zones:
                if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                    ftg_only = True
                    break
        return ftg_only

    def _check_close_to_raceline(self, threshold_m=None) -> bool:
        if threshold_m is None:
            return np.abs(self.cur_d) < self.gb_ego_width_m
        else:
            return np.abs(self.cur_d) < threshold_m

    def _check_close_to_raceline_heading(self, threshold_deg=None) -> bool:
        cloest_wpnt_idx = int(self.cur_s / self.waypoints_dist) % self.num_glb_wpnts
        cloest_wpnt_psi = self.cur_gb_wpnts.list[cloest_wpnt_idx].psi_rad
        if threshold_deg is None:
            return np.abs(self.current_position[2] - cloest_wpnt_psi) < np.deg2rad(20)
        else:
            return np.abs(self.cur_d) < np.deg2rad(threshold_deg)

    def _check_ot_sector(self) -> bool:
        # ROS1: no overtake zone matching cur_s -> not in an OT sector (return False).
        # (An empty overtake_zones means overtaking is suppressed, as in ROS1.)
        for sector in self.overtake_zones:
            if sector[0] <= self.cur_s / self.waypoints_dist <= sector[1]:
                self.ot_section_check_pub.publish(Bool(data=True))
                self._publish_preferred_side()
                return True
        self.ot_section_check_pub.publish(Bool(data=False))
        self._publish_preferred_side()
        return False

    def _preferred_side_here(self) -> str:
        """Which way this map wants an overtake taken at the car's s.

        "auto" - the planner's own room comparison - unless the sector says
        otherwise. Outside every overtaking sector it is also "auto": the
        sector table has nothing to say there, and an overtake is not armed
        anyway.
        """
        if self.waypoints_dist <= 0.0:
            return "auto"
        index = self.cur_s / self.waypoints_dist
        for start, end, side in getattr(self, "overtake_sides", []):
            if start <= index <= end:
                return side
        return "auto"

    def _publish_preferred_side(self) -> None:
        """Tell the planners, rather than making them read the sector table.

        They do not know where the sector boundaries are and have no reason to:
        this node already walks that table every tick for _check_ot_sector, so
        the answer is free here and one string on a latched topic downstream.
        """
        side = self._preferred_side_here()
        if side != getattr(self, "_last_preferred_side", None):
            self._last_preferred_side = side
            self.get_logger().info(
                f"[{self.name}] overtaking side policy here: {side}")
        self.ot_preferred_side_pub.publish(String(data=side))

    def refresh_planner_params(self, planner_name: str) -> None:
        """Re-read one planner's parameter block after a live `ros2 param set`.

        WaypointData reads its block once, when it is constructed. Without
        this, `ros2 param set /state_machine static_avoidance_planner.
        max_speed_mps 1.5` reports success and changes nothing until the next
        launch - which is a bad way to spend a tuning session.
        """
        for data in (self.cur_gb_wpnts, self.cur_recovery_wpnts,
                     self.cur_avoidance_wpnts, self.cur_static_avoidance_wpnts):
            if data is not None and data.name == planner_name:
                data.update_param()
                self.get_logger().info(f"{planner_name}: parameters re-read")
                return

    def _check_getting_closer(self, threshold_m=3.0) -> bool:
        if (
            len(self.obstacles_in_interest) != 0
            and self.cur_vs - self.obstacles_in_interest[0].vs > -0.5
        ):
            return True
        else:
            return False

    def _check_enemy_in_front(self) -> bool:
        horizon = self.gb_horizon_m
        for obs in self.obstacles:
            gap = (obs.s_start - self.cur_s) % self.track_length
            if gap < horizon:
                return True
        return False

    def _check_latest_wpnts(self, src_wpnts, wpnts_data: WaypointData):
        # Frozen cache: keep the captured path, do NOT replace it with fresh output.
        # Stay "available" as long as we are still on the held path (on_spline); once
        # the car runs off its tail the freeze naturally lapses to unavailable.
        if wpnts_data.frozen:
            return bool(wpnts_data.is_init and self._check_on_spline(wpnts_data))

        # Four ways to say no, and they mean completely different things: the
        # planner never published, it published an empty path, its path is too
        # old, or the car is not near the path it drew. Say which.
        if not wpnts_data.enabled:
            return False

        def no(reason):
            self.get_logger().info(
                f"{wpnts_data.name}: not usable - {reason}", throttle_duration_sec=2.0)
            return False

        if src_wpnts is None:
            # Once, not every two seconds. A topic with no publisher stays
            # that way, and repeating it buries the refusals that change.
            self.get_logger().info(
                f"{wpnts_data.name}: nothing has ever been received on its topic",
                once=True)
            return False
        if len(src_wpnts.wpnts) == 0:
            return no("published an empty path")
        age = self.now_sec() - time_to_float(src_wpnts.header.stamp)
        if age > wpnts_data.latest_threshold:
            return no(f"path is {age:.3f} s old, threshold {wpnts_data.latest_threshold}")
        wpnts_data.initialize_traj(src_wpnts)
        if not self._check_on_spline(wpnts_data):
            gap = (wpnts_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = float(np.min(np.linalg.norm(
                wpnts_data.array[:, 0:2] - self.current_position[:2], axis=1)))
            return no(
                f"car is not on it: nearest point {min_dist:.2f} m away "
                f"(need < {wpnts_data.on_spline_min_dist_thres_m}), "
                f"path ends {gap:.2f} m ahead "
                f"(need > {wpnts_data.on_spline_front_horizon_thres_m})")
        return True

    def _check_ftg(self) -> bool:
        threshold = self.ftg_timer_sec * self.rate_hz
        if self.ftg_disabled:
            return False
        else:
            if (self.cur_state == StateType.TRAILING or self.cur_state == StateType.ATTACK) and \
                    self.cur_vs < self.ftg_speed_mps:
                self.ftg_counter += 1
                # Only the crossing, not the count. The count ticked twice a
                # second for the whole three seconds and then reset, so the
                # common case - trailing briefly below ftg_speed_mps and
                # recovering - printed six lines to say nothing happened.
                if self.ftg_counter == 1:
                    self.get_logger().info(
                        f"[{self.name}] stopped behind something; FTG in "
                        f"{self.ftg_timer_sec:.1f} s unless this clears")
                elif self.ftg_counter == int(threshold):
                    self.get_logger().warn(
                        f"[{self.name}] stuck for {self.ftg_timer_sec:.1f} s - "
                        f"falling back to FTG")
            else:
                self.ftg_counter = 0
            return self.ftg_counter > threshold

    def _check_on_spline(self, wpnt_data) -> bool:
        if wpnt_data.is_init:
            gap = (wpnt_data.list[-1].s_m - self.cur_s) % self.track_length
            min_dist = np.min(np.linalg.norm(wpnt_data.array[:, 0:2] - self.current_position[:2], axis=1))
            if gap > wpnt_data.on_spline_front_horizon_thres_m and min_dist < wpnt_data.on_spline_min_dist_thres_m:
                return True
        return False

    @staticmethod
    def _lateral_half_width(obs) -> float:
        """Half the obstacle's LATERAL extent, in metres.

        Not size/2. A detector that models every obstacle as a square of side
        `size` and fills d_left/d_right from that same number, so there size/2
        is the lateral half-width. Ours measures the real Frenet extents per
        axis and reports `size` as the diameter of the circle that bounds the
        cluster - the longest span in any direction. For anything elongated
        along the track that is far wider than the obstacle actually is
        sideways, and this is the clearance test, so the path gets judged
        blocked when it is not. Measured on the car: size ran about 0.58 m
        larger than the lateral extent.

        Falls back to size/2 for a producer that leaves the d bounds at zero.
        """
        lateral = float(obs.d_left) - float(obs.d_right)
        return lateral / 2.0 if lateral > 1e-6 else float(obs.size) / 2.0

    @staticmethod
    def _edge_gap(path_d, obs) -> float:
        """Signed gap from a path at `path_d` to the obstacle's lateral band.

        Negative when the path is inside it. This is the same geometry the
        spline planner uses to place its apex, and the two have to agree or
        one of them publishes a path the other will not drive.

        The old form was |path_d - d_center| minus the half width, which is a
        SYMMETRIC box around d_center. The tracker does not produce one: it
        holds the left and right half-extents separately and writes
        d_right = d_center - right, d_left = d_center + left. When those two
        differ, d_center is not the midpoint, and rebuilding the band as
        d_center +/- (d_left - d_right)/2 slides it sideways - dragging the
        NEAR edge closer than the obstacle actually is by half the asymmetry.

        Measured: an obstacle reported d_center=+0.30 with a half width of
        0.25, so this modelled it as +0.05..+0.55, while the planner's own
        apex puts its near edge at +0.11. Six centimetres of obstacle that is
        not there, and it is why the refusals in that log land one and two
        millimetres under the threshold over and over - "free +0.015, needs
        0.016", "free +0.048, needs 0.050". At 0.63 m and 3.7 m/s the car had
        nowhere to go by then.

        Falls back to the symmetric box for a producer that leaves the bounds
        at zero, exactly as _lateral_half_width does.
        """
        d_right = float(obs.d_right)
        d_left = float(obs.d_left)
        if d_left - d_right <= 1e-6:
            half = float(obs.size) / 2.0
            d_right, d_left = float(obs.d_center) - half, float(obs.d_center) + half
        if path_d >= d_left:
            return path_d - d_left
        if path_d <= d_right:
            return d_right - path_d
        return -min(path_d - d_right, d_left - path_d)

    def _check_free_frenet(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = 2.0
        max_horizon = wpnts_data.max_horizon
        is_gb_track_wpnts = wpnts_data.is_gb_track_wpnts
        is_ot_wpnts = wpnts_data.is_ot_wpnts
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        obstacle_predictions = self.obstacles_prediction

        # Debug: per-obstacle record of which branch decided free/blocked, so a
        # "GB judged free while an obstacle is right ahead" can be explained
        # (empty obstacle list vs prediction branch vs static/dynamic geom).
        dbg = {"is_init": bool(wpnts_data.is_init), "n_obs": len(obstacles), "obs": []}

        if wpnts_data.is_init:
            max_gap = (wpnts_data.array[-1, 2] - self.cur_s) % self.max_s
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                relative_vs = self.cur_vs - obs.vs
                clip_vs = max(relative_vs, self.overtake_min_closing_mps)
                ttc = (gap - self.pars["veh_params"]["length"]) / clip_vs
                tt0 = (gap + 0.3 * self.pars["veh_params"]["length"]) / clip_vs

                rec = {"id": int(obs.id), "static": bool(obs.is_static),
                       "gap": round(float(gap), 2), "d": round(float(obs.d_center), 3),
                       "branch": None, "free_dist": None, "blocked": False}

                if obs.is_static:
                    if not wpnts_data.is_closed and gap > max_gap:
                        # An obstacle PAST THE END of this path is not on it.
                        #
                        # This branch used to set is_free = False, silently -
                        # it is the one condition in this function with no log
                        # line, which is why the refusals only ever named
                        # global_tracking while path_free stayed False on the
                        # avoidance cache with nothing to explain it.
                        #
                        # An avoidance path is about nine metres long on a
                        # twenty metre loop, so on any lap there is always
                        # something beyond its end - a wall false positive is
                        # enough. Treating that as "the path is blocked" made
                        # path_free permanently false and no static avoidance
                        # could ever arm.
                        #
                        # The car does not follow this path forever. It runs
                        # to the end and the state machine decides again, with
                        # that obstacle then inside the horizon. Skip it here.
                        rec["branch"] = "static/beyond_path (ignored)"
                        self._throttled_info(
                            f"beyond/{wpnts_data.name}",
                            f"[{wpnts_data.name}] obs {int(obs.id)} is "
                            f"{gap:.2f} m ahead, past the end of this path "
                            f"at {max_gap:.2f} m - not this path's problem",
                            5.0)
                    elif gap < max_horizon:
                        ot_d = 0
                        if not is_gb_track_wpnts:
                            avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_s))
                            ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                        free_dist = self._edge_gap(ot_d, obs) - self.gb_ego_width_m / 2
                        scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                        rec["branch"] = "static/geom"
                        rec["free_dist"] = round(float(free_dist), 3)
                        if free_dist < lateral_width_m * scaling_factor:
                            is_free = False
                            rec["blocked"] = True
                            # Which path, which obstacle, and what it needed.
                            # Without the name this fires for the raceline
                            # check and the avoidance check alike, and those
                            # use different margins, so the same number can be a
                            # correct "raceline is blocked, go around" or the
                            # refusal that stops the car going around.
                            self._throttled_info(
                                f"blocked/{wpnts_data.name}",
                                f"[{wpnts_data.name}] blocked by obs {int(obs.id)} "
                                f"spanning d={obs.d_right:+.2f}..{obs.d_left:+.2f} "
                                f"({gap:.2f} m ahead): "
                                f"path is at d={ot_d:+.3f}, free {free_dist:+.3f} m, "
                                f"needs {lateral_width_m * scaling_factor:.3f}",
                                1.0,
                            )
                            if closest_obs is None or min_gap > gap:
                                closest_obs = obs
                                min_gap = gap
                    else:
                        rec["branch"] = "static/gap>=max_horizon"
                else:
                    if len(obstacle_predictions) != 0 and self.obstacles_prediction_id == obs.id:
                        rec["branch"] = "dyn/pred"
                        start_idx = 0
                        end_idx = len(obstacle_predictions)
                        if is_ot_wpnts:
                            if ttc > 0:
                                start_idx = min(int(ttc / self.prediction_dt), len(obstacle_predictions))
                            if tt0 > 0:
                                end_idx = min(int(tt0 / self.prediction_dt), len(obstacle_predictions))
                        worst_fd = None
                        for obs_pred in obstacle_predictions[start_idx:end_idx]:
                            wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs_pred.pred_s))
                            wpnt_d = wpnts_data.list[wpnt_idx].d_m
                            min_dist = abs(wpnt_d - obs_pred.pred_d)
                            free_dist = min_dist - self._lateral_half_width(obs) - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            if worst_fd is None or free_dist < worst_fd:
                                worst_fd = free_dist
                            # Was a raw float dump of every prediction pose,
                            # twice a second, at full precision, whether or not
                            # anything was wrong. The refusal it belongs to is
                            # already reported once per path by the
                            # "blocked over the full prediction" line below,
                            # with the worst clearance and the pose count. Kept
                            # at debug for when this branch itself is suspect.
                            if is_ot_wpnts:
                                self.get_logger().debug(
                                    f"pred pose: free {free_dist:+.3f} m, "
                                    f"needs {lateral_width_m * scaling_factor:.3f}, "
                                    f"path d={wpnt_d:+.3f}, opp d={obs_pred.pred_d:+.3f}")
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                rec["blocked"] = True
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
                        rec["free_dist"] = None if worst_fd is None else round(float(worst_fd), 3)
                        rec["pred_n"] = int(end_idx - start_idx)
                    else:
                        rec["branch"] = "dyn/nopred (id_mismatch or empty)"
                        rec["pred_id"] = int(self.obstacles_prediction_id) if self.obstacles_prediction_id is not None else None
                        rec["pred_len"] = len(obstacle_predictions)
                        if not wpnts_data.is_closed and gap > max_gap:
                            # Same reasoning as the static branch above: an
                            # obstacle PAST THE END of a non-closed path is not
                            # on it. Without a prediction this obstacle is just
                            # a frozen box at its current position, and that box
                            # is beyond where this path stops - the car drives
                            # to the end and the state machine decides again,
                            # with the obstacle then inside the horizon.
                            #
                            # This used to set is_free = False, silently - the
                            # one dynamic condition with no log line. In time
                            # trials nothing publishes predictions, so a track
                            # the tracker had not yet classified static (young,
                            # re-created after an occlusion, or a wall ghost)
                            # landed here on every tick and vetoed a static
                            # avoidance the car had room to drive, with nothing
                            # in the log to say so: three boxes in an S, path
                            # correctly ended before the second, car stopped
                            # behind the first with path_free=False and no
                            # "blocked by" line. The head-to-head wrapper
                            # already excuses exactly this case after the fact
                            # (_blocked_only_beyond_path); deciding it here
                            # covers time trials too, and leaves that excusal a
                            # no-op rather than wrong.
                            rec["branch"] = "dyn/nopred/beyond_path (ignored)"
                            self._throttled_info(
                                f"beyond/{wpnts_data.name}",
                                f"[{wpnts_data.name}] non-static obs {int(obs.id)} "
                                f"is {gap:.2f} m ahead, past the end of this "
                                f"path at {max_gap:.2f} m - not this path's "
                                f"problem",
                                5.0)
                        elif gap < max_horizon:
                            ot_d = 0
                            if not is_gb_track_wpnts:
                                avoid_wpnt_idx = np.argmin(abs(wpnts_data.array[:, 2] - obs.s_center))
                                ot_d = wpnts_data.list[avoid_wpnt_idx].d_m
                            free_dist = self._edge_gap(ot_d, obs) - self.gb_ego_width_m / 2
                            scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                            rec["free_dist"] = round(float(free_dist), 3)
                            if free_dist < lateral_width_m * scaling_factor:
                                is_free = False
                                rec["blocked"] = True
                                # Say so, like the static branch does. A
                                # refusal with no log line is what made the
                                # beyond_path case above take a day to find.
                                self._throttled_info(
                                    f"blocked/{wpnts_data.name}",
                                    f"[{wpnts_data.name}] blocked by non-static "
                                    f"obs {int(obs.id)} (no prediction) "
                                    f"spanning d={obs.d_right:+.2f}..{obs.d_left:+.2f} "
                                    f"({gap:.2f} m ahead): "
                                    f"path is at d={ot_d:+.3f}, "
                                    f"free {free_dist:+.3f} m, "
                                    f"needs {lateral_width_m * scaling_factor:.3f}",
                                    1.0,
                                )
                                if closest_obs is None or min_gap > gap:
                                    closest_obs = obs
                                    min_gap = gap
                        else:
                            rec["branch"] = "dyn/nopred/gap>=max_horizon"
                dbg["obs"].append(rec)
        else:
            # An OT/recovery cache with no valid path (is_init False, e.g. expired by
            # _expire_stale_cache) must NOT read as "free": treating a missing avoidance
            # path as clear keeps OVERTAKE alive on an empty cache, which then emits an
            # empty local_wpnts. Report blocked so the source is dropped instead.
            is_free = not (wpnts_data.is_ot_wpnts and not wpnts_data.is_init)

        dbg["is_free"] = bool(is_free)
        wpnts_data.free_dbg = dbg
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _check_free_cartesian(self, wpnts_data) -> bool:
        is_free = True
        closest_obs = None
        min_gap = None
        min_horizon = wpnts_data.min_horizon
        max_horizon = wpnts_data.max_horizon
        free_scaling_reference_distance_m = wpnts_data.free_scaling_reference_distance_m
        lateral_width_m = wpnts_data.lateral_width_m

        obstacles = self.cur_obstacles_in_interest
        if wpnts_data.is_init:
            for obs in obstacles:
                obs_s = obs.s_center
                gap = (obs_s - self.cur_s) % self.max_s
                if gap < max_horizon or min_horizon < (gap - self.max_s):
                    dists = np.linalg.norm(wpnts_data.array[:, 0:2] - np.array([obs.x_m, obs.y_m]), axis=1)
                    min_dist = np.min(dists)
                    free_dist = min_dist - self._lateral_half_width(obs) - self.gb_ego_width_m / 2
                    scaling_factor = np.clip(gap / free_scaling_reference_distance_m, 0.0, 1.0)
                    if free_dist < lateral_width_m * scaling_factor:
                        is_free = False
                        if closest_obs is None or min_gap > gap:
                            closest_obs = obs
                            min_gap = gap
                        self.get_logger().info(
                            f"[{self.name}] RECOVERY_FREE False, obs dist to recovery lane: {min_dist} m",
                            throttle_duration_sec=1.0,
                        )
        else:
            is_free = True
        wpnts_data.closest_target = closest_obs
        wpnts_data.closest_gap = min_gap
        return is_free

    def _src_cache(self, src):
        # The planner-output cache a given local_wpnts_src slices from (None if the
        # source is not backed by an OT/recovery cache, e.g. RACELINE).
        if src == StateType.OVERTAKE:
            return self.cur_static_avoidance_wpnts if self.static_overtaking_mode else self.cur_avoidance_wpnts
        if src == StateType.RECOVERY:
            return self.cur_recovery_wpnts
        return None

    def _expire_stale_cache(self, wpnts_data, ttl_sec):
        # Drop a stale planner-output cache: one that is NOT the current source and
        # whose planner stopped emitting for > ttl_sec (a ghost / old frozen path).
        # The cache actively driven as local_wpnts_src is exempt -- kept alive by the
        # on_spline/hyst/killing hysteresis in _check_availability so the car keeps
        # following it through a few skipped solver frames. A cache that stays alive
        # (planner keeps publishing) but is not the current source is left intact so
        # it can be re-selected instantly with fresh data.
        if not wpnts_data.is_init:
            return
        if wpnts_data is self._src_cache(self.local_wpnts_src):
            return
        if wpnts_data.last_init_sec is None:
            return
        if self.now_sec() - wpnts_data.last_init_sec > ttl_sec:
            wpnts_data.is_init = False
            wpnts_data.closest_target = None

    def _check_availability(self, wpnts, wpnts_data) -> bool:
        """Is there a path to drive, and refresh the cache to the newest one.

        A newer path always wins. The hysteresis is for the case where there
        is NO usable new path - a skipped solver frame - and it holds the
        cached one while the car is still on it.

        It used to be the other way round: while the car was on the cached
        path and the cache was under killing_timer_sec old, this returned True
        without asking the planner anything. _check_latest_wpnts is the only
        thing that calls initialize_traj, so the cache was the only thing that
        ever got driven, and it was never replaced.

        That is what made the third of three obstacles unavoidable. Outside
        OVERTAKE the cache refreshes every tick, because
        _check_static_overtaking_mode calls _check_latest_wpnts itself; inside
        OVERTAKE only this function runs, so the car committed to whatever
        path it entered on. The path for the second obstacle is cut short
        before the third and has already returned to d=0 by its end, and the
        third reads as "past the end of this path - not this path's problem",
        so the path stayed free, OVERTAKE stayed on, and the planner's path
        for the third obstacle - published, correct, visible in RViz - was
        never read. The cache only turned over when the car ran off the end of
        the old path, half a metre before it, which at 3 m/s is a third of a
        second before the obstacle.

        Two and one obstacle worked because the last path in the chain is not
        cut short: with nothing behind it, it does the avoiding itself.
        """
        age = self.now_sec() - time_to_float(wpnts_data.stamp)

        if age > wpnts_data.killing_timer_sec:
            wpnts_data.is_init = False
            return bool(self._check_latest_wpnts(wpnts, wpnts_data))

        # A fresh path beats the cached one, always. This is the line the bug
        # was hiding behind.
        if self._check_latest_wpnts(wpnts, wpnts_data):
            return True

        # No usable fresh path. Keep driving the cached one while the car is
        # still on it and it has not gone stale.
        if age <= wpnts_data.hyst_timer_sec:
            return bool(self._check_on_spline(wpnts_data))

        return False

    def _check_sustainability(self, src_wpnts, wpnts_data) -> bool:
        if self._check_availability(src_wpnts, wpnts_data) and self._check_free_frenet(wpnts_data):
            return True
        return False

    def _check_overtaking_mode(self) -> bool:
        if (
            self._check_ot_sector()
            and self._check_getting_closer(threshold_m=10.0)
            and self._check_latest_wpnts(self.avoidance_wpnts, self.cur_avoidance_wpnts)
            and self._check_free_frenet(self.cur_avoidance_wpnts)
        ):
            self.static_overtaking_mode = False
            return True
        else:
            return False

    @staticmethod
    def _worst_free(wpnts_data):
        """Tightest clearance _check_free_frenet found on this path, or None.

        Read back out of the debug record it already fills in, so comparing
        two paths costs nothing extra.
        """
        dbg = wpnts_data.free_dbg
        if not dbg:
            return None
        room = [rec["free_dist"] for rec in dbg.get("obs", ())
                if rec.get("free_dist") is not None]
        return min(room) if room else None

    def _worth_driving(self, wpnts_data, path_free) -> bool:
        """Should this avoidance path be driven?

        Free is enough on its own. When it is not free, the question is not
        "is this path safe" but "is it better than where refusing it puts the
        car" - and refusing it puts the car on the raceline, which is the
        thing that was declared blocked in the first place.

        path_free is a yes/no against a fixed margin, so a refusal by
        millimetres sent the car down a line that goes straight through the
        obstacle. Measured twice in one run: the avoidance cleared by 0.022 m
        and 0.024 m against margins of 0.034 and 0.031, so it was refused, and
        the raceline it fell back to was 0.203 m and 0.222 m INSIDE the same
        obstacle.

        When the raceline is free this returns False - a marginal avoidance is
        not worth leaving a clear line for - and the frame in that same run
        where the avoidance really was the worse of the two (free -0.055
        against the raceline's +0.070) stays on the line.

        Used by both the entry and the sustainability check, which have to
        agree: with the comparison only on entry, the car dropped out of
        OVERTAKE the moment the path went marginal and had to re-earn it.
        Head to head overrides both of those and calls this from each, and it
        can be holding either the static or the dynamic cache, which is why
        the path being judged is an argument rather than assumed.
        """
        if path_free:
            return True
        if self._check_free_frenet(self.cur_gb_wpnts):
            return False
        avoid_room = self._worst_free(wpnts_data)
        line_room = self._worst_free(self.cur_gb_wpnts)
        # A floor the comparison cannot argue past, the same lesson
        # min_path_clearance_m taught the planner: better than the raceline is
        # not the same as passable.
        #
        # free_dist is already the gap between the car's SIDE and the
        # obstacle's edge, so a negative value means the body overlaps it.
        # Without this the comparison happily drove that, because the raceline
        # overlapped further. Measured over one run, four of the twelve
        # acceptances were negative - the worst took a path 0.117 m inside an
        # obstacle because the raceline was 0.284 m inside. Both are collisions;
        # the avoidance was simply a slower one.
        #
        # When neither clears, the honest answer is to trail and stop, which is
        # what refusing here produces.
        if avoid_room is not None and avoid_room < self.static_overtake_min_clearance_m:
            self.get_logger().warn(
                f"[{wpnts_data.name}] leaves only {avoid_room:+.3f} m at its "
                f"tightest, under the {self.static_overtake_min_clearance_m:.3f} m "
                f"floor - refusing even though the raceline is worse "
                f"({line_room if line_room is None else f'{line_room:+.3f}'} m)",
                throttle_duration_sec=1.0)
            return False
        if (avoid_room is None or line_room is None
                or avoid_room <= line_room + self.static_overtake_better_by_m):
            return False
        self.get_logger().warn(
            f"taking [{wpnts_data.name}] even though it is not clear: it leaves "
            f"{avoid_room:+.3f} m at its tightest and the raceline leaves "
            f"{line_room:+.3f} m, and the raceline is where trailing would put "
            f"the car",
            throttle_duration_sec=1.0)
        return True

    def _check_static_overtaking_mode(self) -> bool:
        # Evaluated separately rather than short-circuited so a refusal can say
        # which condition refused. All four have to hold, and from outside the
        # car they look identical: it trails the obstacle and never pulls out.
        #
        # The speed gate was a hardcoded 3.0, below the raceline's own top
        # speed once speed_scaling reached 0.5 - so for 30% of the lap, the
        # fast 30%, avoidance could not arm at all. That is precisely where an
        # obstacle appears late: coming out of a corner the car is at its
        # quickest and the obstacle was occluded until a metre or two ago. It
        # would trail, brake, and by the time it was under 3.0 it was too close
        # to swerve - and a stopped car in front of a stationary obstacle is a
        # fixed point, so it simply sat there.
        slow_enough = self.cur_vs < self.static_overtake_max_speed_mps
        closing = self._check_getting_closer(threshold_m=7.0)
        have_path = self._check_latest_wpnts(
            self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
        path_free = self._check_free_frenet(self.cur_static_avoidance_wpnts)

        if slow_enough and closing and have_path and self._worth_driving(
                self.cur_static_avoidance_wpnts, path_free):
            self.static_overtaking_mode = True
            return True

        if len(self.cur_obstacles_in_interest) != 0:
            self.get_logger().info(
                "static avoidance refused: "
                f"slow_enough={slow_enough} "
                f"(vs {self.cur_vs:.2f} < {self.static_overtake_max_speed_mps:.2f}), "
                f"closing={closing}, have_path={have_path}, path_free={path_free}",
                throttle_duration_sec=2.0)
        return False

    def _check_overtaking_mode_sustainability(self) -> bool:
        if self.static_overtaking_mode:
            # The same test as the entry, through _worth_driving. They used to
            # differ: entry would accept a path that beat the raceline, and
            # this would then throw it away on the next tick for not being
            # free, so the car flicked out of OVERTAKE and had to earn it back.
            if (
                self._check_availability(self.static_avoidance_wpnts, self.cur_static_avoidance_wpnts)
                and self._worth_driving(
                    self.cur_static_avoidance_wpnts,
                    self._check_free_frenet(self.cur_static_avoidance_wpnts))
            ):
                return True
        else:
            if self._check_availability(self.avoidance_wpnts, self.cur_avoidance_wpnts):
                self.get_logger().debug("AVAILABLE")
                if self._check_free_frenet(self.cur_avoidance_wpnts):
                    return True
        return False

    ################
    # HELPER FUNCS #
    ################
    def update_velocity(self, wpnts_msg, safety_factor=1.0, max_speed=None,
                        yeet_factor=1.0):
        if self.ggv is None or self.gb_wpnts is None:
            return  # velocity replanning unavailable (no veh dyn info / no gb wpnts yet)
        wpnts = wpnts_msg.wpnts
        if len(wpnts) < 3:
            return
        kappa = np.array([wp.kappa_radpm for wp in wpnts])
        el_lengths = np.array([
            np.linalg.norm([
                wpnts[i + 1].x_m - wpnts[i].x_m,
                wpnts[i + 1].y_m - wpnts[i].y_m,
            ])
            for i in range(len(wpnts) - 1)
        ])
        # Bail if the path is degenerate: a zero-length segment or any non-finite input makes
        # calc_vel_profile divide by zero -> NaN velocities that propagate into the local path
        # and eventually the base_link TF. Leaving the original vx_mps untouched is the safe path.
        if (el_lengths <= 1e-6).any() or not np.all(np.isfinite(el_lengths)) \
                or not np.all(np.isfinite(kappa)):
            self.get_logger().warn(
                f"[{self.name}] degenerate path in update_velocity; keeping planner velocities",
                throttle_duration_sec=1.0,
            )
            return

        glb_start_idx = int(wpnts_msg.wpnts[-1].s_m / self.wpnt_dist)
        v_end = self.gb_wpnts.wpnts[glb_start_idx % len(self.gb_wpnts.wpnts)].vx_mps

        ax_max_machines_sf = self.ax_max_machines.copy()
        b_ax_max_machines_sf = self.b_ax_max_machines.copy()
        ax_max_machines_sf[:, 1] *= safety_factor
        b_ax_max_machines_sf[:, 1] *= safety_factor

        vx_profile = calc_vel_profile(
            ax_max_machines=ax_max_machines_sf,
            kappa=kappa,
            el_lengths=el_lengths,
            closed=False,
            drag_coeff=self.pars["veh_params"]["dragcoeff"],
            m_veh=self.pars["veh_params"]["mass"],
            b_ax_max_machines=b_ax_max_machines_sf,
            ggv=self.ggv,
            v_max=self.pars["veh_params"]["v_max"],
            filt_window=self.pars["vel_calc_opts"]["vel_profile_conv_filt_window"],
            dyn_model_exp=self.pars["vel_calc_opts"]["dyn_model_exp"],
            v_start=self.cur_vs,
            v_end=v_end,
        )

        # Cap before ax is derived from the profile, so the accelerations stay
        # consistent with the speeds actually commanded. safety_factor only
        # scales what the motor may do; it does not bound cornering speed,
        # which comes from ggv and the path's curvature. On an evasion swerve
        # through a metre-wide corridor that limit is still far too quick, so
        # the planner gets to state a ceiling outright.
        #
        # First: never faster than the raceline is at the same place.
        #
        # calc_vel_profile above works from ggv, and ggv.csv is flat 12 m/s^2
        # for ax AND ay at every speed - it is not a measured friction ellipse,
        # it is a placeholder. What actually keeps the car on the road is the
        # sector scaling in the map's speed_scaling.yaml, 0.6 on map_yes, and
        # /global_waypoints_scaled already has it baked in. The avoidance
        # profile never saw it: only its END point was tied to a scaled speed,
        # everything between came straight from the optimistic ggv.
        #
        # So on a corner exit the path was planned most of a factor of two
        # quicker than the line it departs from, the car accelerated into a
        # bend with no grip budget left for it, ran wide and hit the wall. It
        # only showed up once the flat max_speed cap stopped hiding it.
        #
        # The raceline speed is the right bound because the avoidance path is
        # never straighter than the line - it is the line plus a swerve - and
        # because it carries whatever the sectors were tuned to, which is where
        # the real knowledge about this track lives.
        raceline_kappa = None
        if self.gb_wpnts is not None and self.wpnt_dist > 0.0:
            n_gb = len(self.gb_wpnts.wpnts)
            idx = (np.asarray([wp.s_m for wp in wpnts]) / self.wpnt_dist)
            idx = np.clip(idx, 0, None).astype(int) % n_gb
            raceline_v = np.asarray([self.gb_wpnts.wpnts[i].vx_mps for i in idx])
            raceline_kappa = np.abs(np.asarray(
                [self.gb_wpnts.wpnts[i].kappa_radpm for i in idx]))

            # yeet_factor lifts that ceiling, and ONLY where the path is no
            # sharper than the line beneath it.
            #
            # Passing a moving car needs a speed advantage; a ceiling at
            # exactly the raceline speed forbids one, which is why the map has
            # carried a yeet_factor since ForzaETH. But the paragraph above is
            # the reason this cap exists at all: on a corner exit the path was
            # planned near twice the speed of the line it departs from, the car
            # accelerated into a bend with no grip budget left, ran wide and hit
            # the wall. ggv.csv is still the flat 12.0 placeholder it was then,
            # so nothing downstream would catch that.
            #
            # The two are reconciled by WHERE it applies. Lifting is allowed
            # exactly where the path bends no more than the raceline - the
            # straight the pass is made on - and tapers to nothing at the
            # sharpest bend the path makes, which is the corner exit the crash
            # happened on. Same shape as the max_speed blend below, with the
            # weight the other way up, so the two cannot disagree about which
            # part of the path is the tight one.
            lift = float(yeet_factor)
            if lift > 1.0:
                extra = np.abs(kappa)
                if raceline_kappa is not None:
                    extra = np.maximum(0.0, extra - raceline_kappa)
                worst = extra.max()
                # 1 where the path is no sharper than the line, 0 at its own
                # worst bend.
                straightness = (1.0 - extra / worst) if worst > 1e-6 \
                    else np.ones_like(extra)
                raceline_v = raceline_v * (1.0 + (lift - 1.0) * straightness)
            vx_profile = np.minimum(vx_profile, raceline_v)

        # The ceiling follows where the path BENDS, not where it sits.
        #
        # It was flat over the whole path once, and the whole path is about six
        # metres of approach - so spotting an obstacle early meant crawling the
        # entire way to it. Scaling it by displacement fixed that, and then the
        # corridor planner broke it a second way: a corridor holds one offset
        # across every obstacle it covers, so |d| is at its maximum for the
        # whole hold and the car crawled the length of it. Which is backwards.
        # A held corridor is the one part of the manoeuvre that IS straight -
        # running parallel to the raceline, past two obstacles, while everyone
        # else swings back to the line between them. That stretch is where the
        # time is, and it was the slowest part of the path.
        #
        # What is actually tight is where the path bends MORE than the line
        # underneath it. That is the swerve in, the swerve out, and the apex of
        # a single-obstacle dodge - and it is nothing at all in the middle of a
        # corridor, where the path is the raceline shifted sideways.
        #
        # Normalised against this path's own worst point, so it needs no
        # parameter and adapts to the shape: 1.0 at the sharpest bend the path
        # makes, 0 where it is no sharper than the line. A single obstacle is
        # therefore completely unchanged - its apex IS its sharpest point.
        if max_speed is not None:
            max_speed = float(max_speed)
            extra = np.abs(kappa)
            if raceline_kappa is not None:
                extra = np.maximum(0.0, extra - raceline_kappa)
            worst = extra.max()
            if worst > 1e-6:
                vx_profile = np.minimum(
                    vx_profile,
                    vx_profile - (vx_profile - max_speed) * (extra / worst))
            else:
                vx_profile = np.minimum(vx_profile, max_speed)

            # Then make the deceleration into the apex something the car can
            # actually do. calc_vel_profile has a backward pass for exactly
            # this, but capping afterwards happens behind its back, and a
            # ceiling imposed after the fact is a speed cliff the controller
            # is asked to track and cannot. Walk back from the end braking at
            # the machine limit and take whichever is lower.
            #
            # b_ax_max_machines is the braking table, in m/s^2 against speed,
            # already scaled by safety_factor above.
            for i in range(len(vx_profile) - 2, -1, -1):
                a_brake = abs(np.interp(
                    vx_profile[i + 1], b_ax_max_machines_sf[:, 0], b_ax_max_machines_sf[:, 1]))
                reachable = np.sqrt(
                    vx_profile[i + 1] ** 2 + 2.0 * a_brake * el_lengths[i])
                vx_profile[i] = min(vx_profile[i], reachable)

        for i in range(len(vx_profile)):
            wpnts_msg.wpnts[i].vx_mps = vx_profile[i]

        ax_profile = tph.calc_ax_profile.calc_ax_profile(
            vx_profile=vx_profile, el_lengths=el_lengths, eq_length_output=False
        )
        for i in range(len(ax_profile)):
            wpnts_msg.wpnts[i].ax_mps2 = ax_profile[i]
        wpnts[len(ax_profile)].ax_mps2 = ax_profile[-1]

    def mincurv_splinification(self):
        coords = np.empty((len(self.cur_gb_wpnts.list), 4))
        for i, wpnt in enumerate(self.cur_gb_wpnts.list):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.vx_mps
        self.mincurv_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.mincurv_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.mincurv_spline_v = Spline(coords[:, 0], coords[:, 3])
        self.get_logger().info(f"[{self.name}] Splinified Min Curve")

    def ot_splinification(self):
        coords = np.empty((len(self.overtake_wpnts), 5))
        for i, wpnt in enumerate(self.overtake_wpnts):
            coords[i, 0] = wpnt.s_m
            coords[i, 1] = wpnt.x_m
            coords[i, 2] = wpnt.y_m
            coords[i, 3] = wpnt.d_m
            coords[i, 4] = wpnt.vx_mps
        coords = coords[coords[:, 0].argsort()]
        # Drop non-finite rows and duplicate/non-increasing s: scipy Spline requires a
        # strictly increasing x or it raises / returns NaN. A reversed or seam-jumped
        # overtake path would otherwise poison every downstream spline eval with NaN.
        coords = coords[np.isfinite(coords).all(axis=1)]
        if len(coords) >= 2:
            keep = np.concatenate([[True], np.diff(coords[:, 0]) > 1e-6])
            coords = coords[keep]
        if len(coords) < 4:
            self.get_logger().warn(
                f"[{self.name}] overtake wpnts degenerate ({len(coords)} usable); skipping splinification",
                throttle_duration_sec=1.0,
            )
            return
        self.ot_spline_x = Spline(coords[:, 0], coords[:, 1])
        self.ot_spline_y = Spline(coords[:, 0], coords[:, 2])
        self.ot_spline_d = Spline(coords[:, 0], coords[:, 3])
        self.ot_spline_v = Spline(coords[:, 0], coords[:, 4])
        self.get_logger().info(f"[{self.name}] Splinified Overtaking Curve")

    def _find_nearest_ot_s(self) -> float:
        half_search_dim = 5
        idxs = [
            i % self.num_ot_points
            for i in range(self.cur_id_ot - half_search_dim, self.cur_id_ot + half_search_dim)
        ]
        ses = np.array([self.overtake_wpnts[i].s_m for i in idxs])
        dists = np.abs(self.cur_s - ses)
        chose_id = np.argmin(dists)
        s_ot = idxs[chose_id]
        s_ot %= self.num_ot_points
        return s_ot

    def get_splini_wpts(self) -> WpntArray:
        if self.static_overtaking_mode:
            wpnts = self.cur_static_avoidance_wpnts
        else:
            wpnts = self.cur_avoidance_wpnts

        # Never slice an invalidated cache: once _expire_stale_cache drops a path
        # (planner stopped emitting), is_init is False and its array is a frozen
        # old trajectory. Returning [] here makes the caller fall back to RACELINE
        # instead of emitting a stale/behind-the-car local path.
        if not wpnts.is_init:
            self._splini_dbg = {"static": bool(self.static_overtaking_mode), "invalid_cache": True}
            return []

        diff = np.linalg.norm(wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
        min_idx = np.argmin(diff)
        avoidance_wpnts = wpnts.list[min_idx:min_idx + self.n_loc_wpnts]

        n_from_avoid = len(avoidance_wpnts)
        glb_extended = 0
        if len(avoidance_wpnts) < self.n_loc_wpnts:
            glb_start_idx = int(wpnts.list[-1].s_m / self.wpnt_dist) + 1
            extra_wpnts = [
                self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                for i in range(self.n_loc_wpnts - len(avoidance_wpnts))
            ]
            avoidance_wpnts.extend(extra_wpnts)
            glb_extended = len(extra_wpnts)

        # Record exactly how this OVERTAKE local path was assembled so the debug
        # snapshot can explain a frozen/behind-the-car first_s (argmin pick vs
        # cache extent vs global-fill), instead of guessing from the raw topic.
        self._splini_dbg = {
            "static": bool(self.static_overtaking_mode),
            "min_idx": int(min_idx),
            "min_dist": round(float(diff[min_idx]), 3),
            "pick_s": round(float(wpnts.array[min_idx, 2]), 3),
            "cache_n": int(len(wpnts.list)),
            "cache_s0": round(float(wpnts.array[0, 2]), 3),
            "cache_slast": round(float(wpnts.array[-1, 2]), 3),
            "n_from_avoid": int(n_from_avoid),
            "glb_extended": int(glb_extended),
        }
        return avoidance_wpnts

    def get_recovery_wpts(self) -> WpntArray:
        if self.cur_recovery_wpnts.is_init:
            diff = np.linalg.norm(self.cur_recovery_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            wpnts = self.cur_recovery_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            n_from_rec = len(wpnts)
            glb_extended = 0
            if len(wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_recovery_wpnts.list[-1].s_m / self.wpnt_dist)
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(wpnts))
                ]
                wpnts.extend(extra_wpnts)
                glb_extended = len(extra_wpnts)
            self._recovery_dbg = {
                "min_idx": int(min_idx),
                "min_dist": round(float(diff[min_idx]), 3),
                "pick_s": round(float(self.cur_recovery_wpnts.array[min_idx, 2]), 3),
                "cache_n": int(len(self.cur_recovery_wpnts.list)),
                "cache_s0": round(float(self.cur_recovery_wpnts.array[0, 2]), 3),
                "cache_slast": round(float(self.cur_recovery_wpnts.array[-1, 2]), 3),
                "n_from_rec": int(n_from_rec),
                "glb_extended": int(glb_extended),
            }
            return wpnts

    def get_start_wpts(self) -> WpntArray:
        if self.cur_start_wpnts.is_init:
            diff = np.linalg.norm(self.cur_start_wpnts.array[:, 0:2] - self.current_position[:2], axis=1)
            min_idx = np.argmin(diff)
            start_wpnts = self.cur_start_wpnts.list[min_idx:min_idx + self.n_loc_wpnts]
            if len(start_wpnts) < self.n_loc_wpnts:
                glb_start_idx = int(self.cur_start_wpnts.list[-1].s_m / self.wpnt_dist) + 1
                extra_wpnts = [
                    self.cur_gb_wpnts.list[(glb_start_idx + i) % len(self.cur_gb_wpnts.list)]
                    for i in range(self.n_loc_wpnts - len(start_wpnts))
                ]
                start_wpnts.extend(extra_wpnts)
            return start_wpnts
        else:
            self.get_logger().debug(f"[{self.name}] No valid avoidance waypoints, passing global waypoints")

    #######
    # VIZ #
    #######
    def _publish_debug(self, local_wpnts):
        # Emit a per-loop JSON snapshot on /state_machine/debug capturing the
        # full local_wpnts source-selection state. Purpose: catch a stale source
        # cache (e.g. cur_recovery_wpnts frozen at an old snapshot while the raw
        # recovery topic keeps advancing) leaking into local_wpnts, and the
        # controller-poisoning "car ran off the end of a frozen local path"
        # condition (idx near the tail -> empty curvature slice -> NaN).
        # Building the snapshot runs two real checks and then json.dumps, every
        # tick, for a topic nobody reads unless they are debugging. Skip it when
        # nothing is listening.
        if self.debug_pub.get_subscription_count() == 0:
            return

        def s0(wpnts):
            return round(wpnts[0].s_m, 3) if wpnts else None

        first_s = local_wpnts[0].s_m if local_wpnts else None
        last_s = local_wpnts[-1].s_m if local_wpnts else None
        # frenet gap between car and where the emitted local path starts (wrap-safe)
        gap = None
        if first_s is not None and self.track_length:
            ds = (first_s - self.cur_s) % self.track_length
            gap = round(min(ds, self.track_length - ds), 3)

        rec = self.cur_recovery_wpnts
        avoid = self.cur_avoidance_wpnts
        snap = {
            "t": round(self.now_sec(), 3),
            "src": self.local_wpnts_src.name,
            "state": self.cur_state.name,
            "cur_s": round(self.cur_s, 3),
            "cur_d": round(self.cur_d, 4),
            "cur_vs": round(self.cur_vs, 3),
            "close_to_raceline": bool(self._check_close_to_raceline(0.05)
                                      * self._check_close_to_raceline_heading(20)),
            "n_obs": len(self.cur_obstacles_in_interest),
            "local_first_s": None if first_s is None else round(first_s, 3),
            "local_last_s": None if last_s is None else round(last_s, 3),
            "local_n": len(local_wpnts) if local_wpnts else 0,
            "start_gap_m": gap,  # >~2m means a stale cache leaked in
            "recovery": {
                "topic_s": s0(self.recovery_wpnts.wpnts) if self.recovery_wpnts is not None else None,
                "cache_s": s0(rec.list),
                "cache_last_s": round(rec.list[-1].s_m, 3) if rec.list else None,
                "cache_age": (None if rec.stamp is None
                              else round(self.now_sec() - time_to_float(rec.stamp), 3)),
                # Wall-clock since the cache was last actually re-initialized with
                # fresh planner output, and total re-init count. If reinit_age keeps
                # growing while the topic advances, the cache is stale (never re-slotted).
                "reinit_age": (None if rec.last_init_sec is None
                               else round(self.now_sec() - rec.last_init_sec, 3)),
                "reinit_count": rec.init_count,
                "is_init": rec.is_init,
            },
            "avoidance": {
                "topic_s": s0(self.avoidance_wpnts.wpnts) if self.avoidance_wpnts is not None else None,
                "cache_s": s0(avoid.list),
                "cache_last_s": round(avoid.list[-1].s_m, 3) if avoid.list else None,
                "cache_age": (None if avoid.stamp is None
                              else round(self.now_sec() - time_to_float(avoid.stamp), 3)),
                "reinit_age": (None if avoid.last_init_sec is None
                               else round(self.now_sec() - avoid.last_init_sec, 3)),
                "reinit_count": avoid.init_count,
                "is_init": avoid.is_init,
            },
            # Internal slice detail from get_splini_wpts / get_recovery_wpts for
            # THIS loop (None if that source was not used). Shows the exact
            # min_idx, the s it picked, cache extent, and how many points came
            # from the avoidance/recovery cache vs global-fill -> pinpoints why
            # local_first_s sits where it does (argmin pick vs glb-extend).
            "splini_slice": self._splini_dbg,
            "recovery_slice": self._recovery_dbg,
            # Last free-check decisions this loop (why GB/recovery was judged free
            # or blocked). gb_free explains a "drove into an obstacle ahead" event.
            "gb_free": self.cur_gb_wpnts.free_dbg,
            "recovery_free": self.cur_recovery_wpnts.free_dbg,
        }
        self.debug_pub.publish(String(data=json.dumps(snap)))

    def _pub_local_wpnts(self, wpts, draw=True):
        loc_wpnts = WpntArray()
        loc_wpnts.wpnts = wpts if wpts is not None else []
        loc_wpnts.header.stamp = self.get_clock().now().to_msg()
        loc_wpnts.header.frame_id = "map"

        # The path itself, every tick. This is what the controller drives.
        self.loc_wpnt_pub.publish(loc_wpnts)

        if not draw:
            return

        # ONE marker, not one per waypoint. SPHERE_LIST draws a sphere at every
        # entry of `points`, so the picture is identical while the message is a
        # single Marker plus N Points instead of N Markers - and a Point is
        # three floats where a Marker is a nested tree with a CompressedImage
        # in it. py-spy had Marker.__init__ at 134 s of a 325 s run.
        #
        # z carries velocity, as before, so the path still stands up off the
        # floor where the car is meant to be quick.
        #
        # A fixed id means each publish replaces the last, which is what the
        # DELETEALL used to buy: no stale spheres when the path shortens, and
        # no flicker from clearing and drawing in separate messages.
        mrk = Marker()
        mrk.header.frame_id = "map"
        mrk.header.stamp = loc_wpnts.header.stamp
        mrk.id = 0
        mrk.type = Marker.SPHERE_LIST
        mrk.action = Marker.ADD
        mrk.scale.x = 0.15
        mrk.scale.y = 0.15
        mrk.scale.z = 0.15
        mrk.color.a = 1.0
        mrk.color.g = 1.0
        mrk.pose.orientation.w = 1.0
        mrk.points = [
            Point(x=wpnt.x_m, y=wpnt.y_m, z=wpnt.vx_mps) for wpnt in loc_wpnts.wpnts
        ]

        loc_markers = MarkerArray()
        loc_markers.markers.append(mrk)
        self.vis_loc_wpnt_pub.publish(loc_markers)

    def visualize_state(self, state: str):
        if self.first_visualization:
            self.first_visualization = False
            x0 = self.cur_gb_wpnts.list[0].x_m
            y0 = self.cur_gb_wpnts.list[0].y_m
            x1 = self.cur_gb_wpnts.list[1].x_m
            y1 = self.cur_gb_wpnts.list[1].y_m
            xy_norm = (
                -np.array([y1 - y0, x0 - x1]) / np.linalg.norm([y1 - y0, x0 - x1])
                * 1.25 * self.cur_gb_wpnts.list[0].d_left
            )
            self.x_viz = x0 + xy_norm[0]
            self.y_viz = y0 + xy_norm[1]

        mrk = Marker()
        mrk.type = mrk.SPHERE
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.pose.position.x = float(self.x_viz)
        mrk.pose.position.y = float(self.y_viz)
        mrk.pose.position.z = 0.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 1.0
        mrk.scale.y = 1.0
        mrk.scale.z = 1.0

        if state == "RACELINE":
            mrk.color.b = 1.0
        elif state == "OVERTAKE":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 0.0
        elif state == "TRAILING":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        elif state == "ATTACK":
            mrk.color.r = 1.0
            mrk.color.g = 0.0
            mrk.color.b = 1.0
        elif state == "FTGONLY":
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        elif state == "RECOVERY":
            mrk.color.r = 0.0
            mrk.color.g = 1.0
            mrk.color.b = 0.0
        else:
            mrk.color.r = 1.0
            mrk.color.g = 1.0
            mrk.color.b = 1.0
        self.state_mrk.publish(mrk)

    def publish_not_ready_marker(self):
        mrk = Marker()
        mrk.type = mrk.TEXT_VIEW_FACING
        mrk.id = 1
        mrk.header.frame_id = "map"
        mrk.header.stamp = self.get_clock().now().to_msg()
        mrk.color.a = 1.0
        mrk.color.r = 1.0
        mrk.color.g = 0.0
        mrk.color.b = 0.0
        mrk.pose.position.x = float(np.mean([wpnt.x_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.y = float(np.mean([wpnt.y_m for wpnt in self.cur_gb_wpnts.list]))
        mrk.pose.position.z = 1.0
        mrk.pose.orientation.w = 1.0
        mrk.scale.x = 4.69
        mrk.scale.y = 4.69
        mrk.scale.z = 4.69
        mrk.text = "BATTERY TOO LOW!!!"
        self.emergency_pub.publish(mrk)

    def update_waypoints(self):
        if not self.cur_gb_wpnts.is_init:
            self.cur_gb_wpnts.initialize_traj(self.gb_wpnts)
        else:
            self.cur_gb_wpnts.list = self.gb_wpnts.wpnts
        self.cur_obstacles_in_interest = self.obstacles_in_interest
        return

    def assign_trailing_target(self):
        """Fill behavior_strategy.trailing_targets for this tick.

        A method rather than the inline branch it replaces, so head to head can
        widen it without a second copy of loop() - the publish happens a few
        lines below, so there is no seam after this one. The body is exactly
        the branch that was here, so time trials is unchanged.

        NOTE: check_ot_cloest_target() intentionally NOT called -- it promoted
        the src to OVERTAKE while merely trailing (un-committed OT line). See
        get_farthest_target for the rationale; overtaking is gated by the state.
        """
        if self.cur_state == StateType.TRAILING:
            self.behavior_strategy.trailing_targets, self.local_wpnts_src = \
                self.get_farthest_target(self.local_wpnts_src)
        else:
            self.behavior_strategy.trailing_targets = []

    def get_overtaking_target(self):
        if self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        if self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        else:
            return []

    def get_traling_target(self):
        if self.local_wpnts_src == StateType.RACELINE and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target]
        elif self.local_wpnts_src == StateType.OVERTAKE and self.ot_closest_target is not None:
            return [self.ot_closest_target]
        else:
            return []

    def get_farthest_target(self, local_wpnts_src):
        # TRAILING must NOT hijack the src to OVERTAKE here: overtaking is gated by the
        # OVERTAKE state (sector/getting_closer/free_frenet). Pulling the raw avoidance
        # trajectory into local_wpnts while merely trailing would steer the car onto an
        # un-committed OT line -- exactly what the OT-blended recovery path exists to
        # avoid. Keep the src the transition chose (GB/RECOVERY); only pick the trailing
        # target (the farthest-ahead obstacle) off that same source.
        if local_wpnts_src == StateType.RACELINE and self.cur_gb_wpnts.closest_target is not None:
            return [self.cur_gb_wpnts.closest_target], local_wpnts_src

        if local_wpnts_src == StateType.RECOVERY and self.cur_recovery_wpnts.closest_target is not None:
            return [self.cur_recovery_wpnts.closest_target], local_wpnts_src

        return [], local_wpnts_src

    def check_ot_cloest_target(self):
        if self.gb_closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.RACELINE:
            if self.ot_closest_gap > self.gb_closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE
        elif self.cur_recovery_wpnts.closest_target is not None and self.ot_closest_target is not None and \
                self.local_wpnts_src == StateType.RECOVERY:
            if self.ot_closest_gap > self.cur_recovery_wpnts.closest_gap:
                self.local_wpnts_src = StateType.OVERTAKE

    def save_params_to_yaml(self):
        # ROS1 dynamic_statemachine_server.save_yaml: persist the dynamic tunables to
        # state_machine_params.yaml, preserving the other keys.
        import yaml
        if not self.config_dir:
            self.get_logger().error(f"[{self.name}] config_dir unset; cannot save params")
            return
        path = os.path.join(self.config_dir, "state_machine_params.yaml")
        keys = ["lateral_width_ot_m", "overtaking_ttl_sec",
                "splini_hyst_timer_sec", "splini_ttl", "pred_splini_ttl",
                "emergency_break_horizon", "ftg_speed_mps", "ftg_timer_sec",
                "ftg_active", "force_RACELINE"]
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            block = data.setdefault("state_machine", {}).setdefault("ros__parameters", {})
            for k in keys:
                if self.has_parameter(k):
                    block[k] = self.get_parameter(k).value
            block["save_params"] = False
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f"[{self.name}] saved params to {path}")
        except Exception as e:
            self.get_logger().error(f"[{self.name}] failed to save params: {e}")

    def _handle_momentary_params(self):
        # Act on the rqt buttons outside the on-set callback so set_parameters() is safe.
        if self._save_start_traj_requested:
            self._save_start_traj_requested = False
            self.save_start_traj_pub.publish(Bool(data=True))
            self.set_parameters([rclpy.parameter.Parameter('save_start_traj', rclpy.Parameter.Type.BOOL, False)])
        if self._save_params_requested:
            self._save_params_requested = False
            self.save_params_to_yaml()
            self.set_parameters([rclpy.parameter.Parameter('save_params', rclpy.Parameter.Type.BOOL, False)])

    #############
    # MAIN LOOP #
    #############
    def loop(self):
        self._splini_dbg = None
        self._recovery_dbg = None
        self._handle_momentary_params()
        if self.measuring:
            start = time.perf_counter()

        self.update_waypoints()
        self.gb_closest_target = None
        self.ot_closest_target = None
        need_vel_planner = False

        self.cur_gb_wpnts.closest_target = None
        self.cur_recovery_wpnts.closest_target = None
        self.cur_avoidance_wpnts.closest_target = None
        self.cur_static_avoidance_wpnts.closest_target = None
        self.cur_start_wpnts.closest_target = None

        # Expire any planner-output cache whose planner stopped emitting for >1 s, so
        # a frozen path (old avoidance/static/recovery output) can't keep being sliced
        # into local_wpnts or keep passing the sustainability/free checks. A live
        # planner re-inits every loop, so a path actively being driven never expires.
        self._expire_stale_cache(self.cur_avoidance_wpnts, 2.0)
        self._expire_stale_cache(self.cur_static_avoidance_wpnts, 2.0)
        self._expire_stale_cache(self.cur_recovery_wpnts, 2.0)

        # safety check
        if self.cur_volt < self.volt_threshold:
            self.get_logger().error(
                f"[{self.name}] VOLTS TOO LOW, STOP THE CAR", throttle_duration_sec=1.0
            )
            self.publish_not_ready_marker()

        if self.force_raceline_state:
            self.cur_state = StateType.RACELINE
            self.local_wpnts_src = StateType.RACELINE
        elif self._check_only_ftg_zone():
            self.cur_state = StateType.FTGONLY
            self.local_wpnts_src = StateType.FTGONLY
            self.get_logger().warn(f"[{self.name}] FTGONLY sector !!!")
        else:
            self.cur_state, self.local_wpnts_src = self.state_transitions[self.cur_state](self)

        self.assign_trailing_target()

        self.behavior_strategy.overtaking_targets = self.get_overtaking_target()

        # Rule 2 (source change): when the source switches to a different cache (e.g.
        # RECOVERY->GB as the car reaches the raceline), drop the cache we just left so
        # its output can't linger and be re-selected as a ghost. A live planner re-fills
        # its cache via _check_latest_wpnts next time that source is needed, so this only
        # discards a path we actually stopped driving.
        cur_src_cache = self._src_cache(self.local_wpnts_src)
        if self._prev_src_cache is not None and self._prev_src_cache is not cur_src_cache:
            self._prev_src_cache.is_init = False
            self._prev_src_cache.closest_target = None
        self._prev_src_cache = cur_src_cache

        # Freeze the recovery/blended path while it is the active source: capture it on
        # entry (the transition already re-inited the cache with fresh output this loop)
        # and hold that single path until we leave RECOVERY, so the controller target
        # stops jumping as recovery_spliner re-anchors the blended path every frame.
        self._hold_recovery_freeze()

        # Post-OVERTAKE grace: keep the blended-recovery source eligible for a short
        # window after leaving OVERTAKE (see _select_recovery_source), so the OT->GB
        # blend can smooth the return instead of snapping to GB the instant OVERTAKE
        # ends. Refresh while overtaking, decrement once we are out.
        if self.cur_state == StateType.OVERTAKE:
            self._blended_grace_count = self.blended_recovery_grace_loops
        elif self._blended_grace_count > 0:
            self._blended_grace_count -= 1

        local_wpnts = self.states[self.local_wpnts_src](self)

        # Safety net: never publish an empty local path (an invalidated OT/recovery
        # cache makes its slice return []). An empty WpntArray crashes the controller
        # (1-D waypoint array indexed as 2-D). Fill from the global raceline, which is
        # regenerated at the car every loop. Only the PATH source is swapped -- cur_state
        # is left as the transition decided (e.g. TRAILING keeps trailing/braking), so
        # this never turns "obstacle ahead" into a full-speed GB run.
        if not local_wpnts:
            self.local_wpnts_src = StateType.RACELINE
            local_wpnts = self.states[StateType.RACELINE](self)

        self._publish_debug(local_wpnts)

        if self.cur_state == StateType.LOSTLINE:
            self.cur_state = StateType.RACELINE

        need_vel_planner = False
        self.behavior_strategy.header.stamp = self.get_clock().now().to_msg()
        self.behavior_strategy.local_wpnts = local_wpnts if local_wpnts is not None else []
        self.behavior_strategy.state = self.cur_state.value
        self.behavior_strategy.need_vel_planner = need_vel_planner

        self.behavior_strategy_pub.publish(self.behavior_strategy)

        self.state_pub.publish(String(data=self.cur_state.value))

        draw = self._viz_due()
        if draw:
            self.visualize_state(state=self.cur_state.value)

        self._pub_local_wpnts(local_wpnts, draw=draw)

        if self.cur_state != StateType.TRAILING and self.cur_state != StateType.ATTACK:
            self.ftg_counter = 0

        if not draw:
            if self.measuring:
                self.latency_pub.publish(Float32(data=1.0 / (time.perf_counter() - start)))
            return

        overtaking_target_mrk = Marker()
        overtaking_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.overtaking_targets) != 0:
            overtaking_target_mrk.type = Marker.SPHERE
            overtaking_target_mrk.scale.x = 0.5
            overtaking_target_mrk.scale.y = 0.5
            overtaking_target_mrk.scale.z = 0.5
            overtaking_target_mrk.color.a = 1.0
            overtaking_target_mrk.color.b = 1.0
            overtaking_target_mrk.pose.position.x = self.behavior_strategy.overtaking_targets[0].x_m
            overtaking_target_mrk.pose.position.y = self.behavior_strategy.overtaking_targets[0].y_m
            overtaking_target_mrk.pose.orientation.w = 1.0
        else:
            overtaking_target_mrk.action = Marker.DELETEALL
        self.overtaking_marker_pub.publish(overtaking_target_mrk)

        trailing_target_mrk = Marker()
        trailing_target_mrk.header.frame_id = "map"  # set always so the DELETEALL marker isn't dropped by RViz (empty frame)
        if len(self.behavior_strategy.trailing_targets) != 0:
            trailing_target_mrk.type = Marker.SPHERE
            trailing_target_mrk.scale.x = 0.5
            trailing_target_mrk.scale.y = 0.5
            trailing_target_mrk.scale.z = 0.5
            trailing_target_mrk.color.a = 1.0
            trailing_target_mrk.color.g = 1.0
            trailing_target_mrk.pose.position.x = self.behavior_strategy.trailing_targets[0].x_m
            trailing_target_mrk.pose.position.y = self.behavior_strategy.trailing_targets[0].y_m
            trailing_target_mrk.pose.orientation.w = 1.0
        else:
            trailing_target_mrk.action = Marker.DELETEALL
        self.trailing_marker_pub.publish(trailing_target_mrk)

        if self.measuring:
            end = time.perf_counter()
            self.latency_pub.publish(Float32(data=1.0 / (end - start)))


# defined as entry point in setup.py:
def main(args=None):
    rclpy.init(args=args)
    state_machine = StateMachine()
    try:
        rclpy.spin(state_machine)
    except KeyboardInterrupt:
        pass
    state_machine.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
