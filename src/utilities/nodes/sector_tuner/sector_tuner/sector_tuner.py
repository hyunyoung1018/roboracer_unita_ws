import rclpy
from rcl_interfaces.msg import ParameterType, ParameterDescriptor, FloatingPointRange
from rclpy.node import Node
from f110_msgs.msg import WpntArray
import numpy as np
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from tf_transformations import quaternion_from_euler

# rqt slider bounds for the optional per-sector t_clip_min [m]. Wide enough for
# every raceline speed we run and narrow enough that a fat-fingered drag cannot
# unpin the lookahead entirely.
T_CLIP_MIN_RANGE = (0.3, 3.0)

# Humble's rclpy has no ParameterEventHandler; newer distros do. Try the
# distro first so this disappears on its own after an upgrade, and fall back to
# the backport in utilities/libraries.
try:
    from rclpy.parameter_event_handler import ParameterEventHandler
except ImportError:
    from parameter_event_handler.parameter_event_handler import ParameterEventHandler

class SectorTuner(Node):
    """
    Sector scaler for the velocity of the global waypoints
    """
    def __init__(self):
        super().__init__('sector_tuner',
                         allow_undeclared_parameters=True,
                         automatically_declare_parameters_from_overrides=True)
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.vis_timer = self.create_timer(1.0, self.marker_callback)
        
        # sectors params
        self.glb_wpnts_og = None
        self.glb_wpnts_scaled = None
        self.glb_wpnts_sp_og = None
        self.glb_wpnts_sp_scaled = None

        # get initial scaling
        self.sectors_params=self.parameters_to_dict()
        self.n_sectors = self.sectors_params['n_sectors']
        self._fallback_warned = set()
        self.get_logger().info(
            f"{self.n_sectors} sector(s) from speed_scaling.yaml, "
            f"global_limit {self.sectors_params.get('global_limit')}")

        # optional per-sector L1 lookahead floor - see _refresh_t_clip_min
        self._t_clip_min_ready = False
        self._t_clip_min_fallback = 1.1
        self._t_clip_min_partial_warned = False
        self._refresh_t_clip_min()

        desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])
        self.set_descriptor('global_limit',descriptor=desc)
        # A lookahead is metres, not a 0-1 fraction, so it gets its own slider range.
        t_clip_desc = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=T_CLIP_MIN_RANGE[0], to_value=T_CLIP_MIN_RANGE[1], step=0.01)])
        for i in range(self.n_sectors):
            self.set_descriptor('Sector'+str(i)+'.scaling',descriptor=desc)
            if self.has_parameter(f'Sector{i}.t_clip_min'):
                self.set_descriptor(f'Sector{i}.t_clip_min', descriptor=t_clip_desc)

        # dyn params sub
        self.glb_wpnts_name = "/global_waypoints"
        self.handler = ParameterEventHandler(self)
        self.callback_handle = self.handler.add_parameter_event_callback(
            callback=self.dyn_param_cb,
        )
        self.global_waypoint_sub = self.create_subscription(
            WpntArray,
            self.glb_wpnts_name,
            self.global_waypoints_cb,
            10)
        self.global_waypoint_sp_sub = self.create_subscription(
            WpntArray,
            self.glb_wpnts_name+"/shortest_path",
            self.global_waypoints_sp_cb,
            10)
        
        # new glb_waypoints pub
        self.scaled_points_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled", 10)
        self.scaled_points_sp_pub = self.create_publisher(WpntArray, "/global_waypoints_scaled/shortest_path", 10)
        # One L1 lookahead floor per global waypoint, boundary-blended exactly like
        # the speed scaling. Only published when the map defines it; the controller
        # falls back to its own controller.yaml value otherwise.
        self.t_clip_min_pub = self.create_publisher(Float32MultiArray, "/sector_t_clip_min", 10)
        
        # Visualizations
        self.sector_visualization_pub = self.create_publisher(MarkerArray, '/sector_markers', 10)
        
        self.get_logger().info("Waiting for global waypoints...")
        
    def parameters_to_dict(self):
        params = {}
        for key in self._parameters:
            keylist = key.split('.')
            paramit = params
            for subkey in keylist[:-1]:
                paramit = paramit.setdefault(subkey, {})
            paramit[keylist[-1]] = self.get_parameter(key).value
        return params

    def global_waypoints_cb(self, data:WpntArray):
        """
        Saves the global waypoints of the main trajectory (e.g. min curvature)
        """
        self.glb_wpnts_og = data
     
    def global_waypoints_sp_cb(self, data:WpntArray):
        """
        Saves the global waypoints of the shortest path
        """
        self.glb_wpnts_sp_og = data

    def dyn_param_cb(self, parameter_event):
        """
        Notices the change in the parameters and scales the global waypoints
        """
        if(parameter_event.node != '/sector_tuner'):
            return
        self.sectors_params = self.parameters_to_dict()
        # update params 
        for i in range(self.n_sectors):
            self.sectors_params[f"Sector{i}"]['scaling'] = np.clip(
                self.sectors_params[f"Sector{i}"]['scaling'], 0, self.sectors_params['global_limit']
            )

        self._refresh_t_clip_min()
        self.get_logger().info(str(self.sectors_params))

    def _refresh_t_clip_min(self):
        """Decide whether this map drives the controller's L1 lookahead floor.

        All-or-nothing on purpose. A map with no t_clip_min at all is every map
        written before this existed: publish nothing and the controller keeps the
        single value from controller.yaml, exactly as before. A map with the field
        on only some sectors is a typo, not an intent - inventing a number for the
        rest would silently override a tuned controller.yaml on part of the lap.
        """
        defined = [self.sectors_params.get(f"Sector{i}", {}).get('t_clip_min')
                   for i in range(self.n_sectors)]
        present = [float(v) for v in defined if v is not None]
        self._t_clip_min_ready = bool(present) and len(present) == self.n_sectors
        if present:
            # Where the lookup falls through, err long: a floor that is too low
            # unpins the lookahead and the car weaves.
            self._t_clip_min_fallback = max(present)
        if present and not self._t_clip_min_ready and not self._t_clip_min_partial_warned:
            self._t_clip_min_partial_warned = True
            self.get_logger().warn(
                f"t_clip_min set on {len(present)}/{self.n_sectors} sectors; "
                f"needs all or none. Not publishing /sector_t_clip_min - the "
                f"controller will use its own controller.yaml value.")

    def _publish_t_clip_min(self):
        if not self._t_clip_min_ready or self.glb_wpnts_og is None:
            return
        msg = Float32MultiArray()
        msg.data = [float(self.get_sector_value(i, 't_clip_min', self._t_clip_min_fallback))
                    for i in range(len(self.glb_wpnts_og.wpnts))]
        self.t_clip_min_pub.publish(msg)

    def get_vel_scaling(self, s):
        """
        Gets the dynamically reconfigured velocity scaling for the points.
        """
        return self.get_sector_value(s, 'scaling', 0.0)

    def _field(self, i, field, fallback):
        """One sector's value for `field`, or `fallback` if it does not carry it.

        A missing sector or a missing key is not fatal here: maps written before
        a field existed simply do not have it, and the caller decides what that
        means (see _refresh_t_clip_min).
        """
        sec = self.sectors_params.get(f"Sector{i}", {}) or {}
        val = sec.get(field)
        return fallback if val is None else float(val)

    def get_sector_value(self, s, field='scaling', fallback=0.0):
        """
        Gets the dynamically reconfigured value of `field` for the points.
        Linearly interpolates for points between two sectors

        Parameters
        ----------
        s
            s parameter whose sector we want to find
        field
            the sector key to read: 'scaling', 't_clip_min', ...
        fallback
            value for sectors that do not define `field`
        """
        hl_change = 10
        scaler = None

        if self.n_sectors > 1:
            for i in range(self.n_sectors):
                if i == 0 :
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self._field(self.n_sectors-1, field, fallback), self._field(i, field, fallback)]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self._field(i, field, fallback)
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self._field(i, field, fallback), self._field(i+1, field, fallback)]
                    )
                elif i != self.n_sectors-1:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self._field(i-1, field, fallback), self._field(i, field, fallback)]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start'] - hl_change):
                        scaler = self._field(i, field, fallback)
                    elif (s >= self.sectors_params[f'Sector{i+1}']['start'] - hl_change) and (s < self.sectors_params[f'Sector{i+1}']['start']):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i+1}']['start']-hl_change, self.sectors_params[f'Sector{i+1}']['start']+hl_change],
                        fp=[self._field(i, field, fallback), self._field(i+1, field, fallback)]
                    )
                else:
                    if (s >= self.sectors_params[f'Sector{i}']['start']) and (s < self.sectors_params[f'Sector{i}']['start'] + hl_change):
                        scaler = np.interp(
                            x=s,
                            xp=[self.sectors_params[f'Sector{i}']['start']-hl_change, self.sectors_params[f'Sector{i}']['start']+hl_change],
                            fp=[self._field(i-1, field, fallback), self._field(i, field, fallback)]
                        )
                    elif (s >= self.sectors_params[f'Sector{i}']['start'] + hl_change) and (s < self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = self._field(i, field, fallback)
                    elif (s >= self.sectors_params[f'Sector{i}']['end'] - hl_change):
                        scaler = np.interp(
                        x=s,
                        xp=[self.sectors_params[f'Sector{i}']['end']-hl_change, self.sectors_params[f'Sector{i}']['end']+hl_change],
                        fp=[self._field(i, field, fallback), self._field(0, field, fallback)]
                    )
        elif self.n_sectors == 1:
            scaler = self._field(0, field, fallback)

        if scaler is None:
            # The branches above interpolate across sector boundaries and do not
            # cover every s: a sector shorter than 2*hl_change has a gap in the
            # middle, and n_sectors of 0 matches nothing at all. Falling off the
            # end raised UnboundLocalError and killed the node, which stops the
            # state machine, which stops the car. Use the value of whatever
            # sector actually contains s.
            scaler = fallback
            for i in range(self.n_sectors):
                sec = self.sectors_params.get(f"Sector{i}", {})
                if sec.get('start', 0) <= s <= sec.get('end', 0):
                    scaler = self._field(i, field, fallback)
                    break
            if field not in self._fallback_warned:
                self._fallback_warned.add(field)
                self.get_logger().warn(
                    f"no sector interpolation covered index {s} for '{field}' "
                    f"(n_sectors={self.n_sectors}); using {scaler}")

        return scaler

    def scale_points(self):
        """
        Scales the global waypoints' velocities
        """
        if self.glb_wpnts_scaled is None:
            self.glb_wpnts_scaled = self.glb_wpnts_og
            self.glb_wpnts_sp_scaled = self.glb_wpnts_sp_og

        for i, wpnt  in enumerate(self.glb_wpnts_og.wpnts):
            vel_scaling = self.get_vel_scaling(i)
            new_vel = wpnt.vx_mps*vel_scaling
            self.glb_wpnts_scaled.wpnts[i].vx_mps = new_vel

    def timer_callback(self):
        if(self.glb_wpnts_og is None):
            return
        self.scale_points()
        self.scaled_points_pub.publish(self.glb_wpnts_scaled)
        self._publish_t_clip_min()
        
    def marker_callback(self):
        if self.glb_wpnts_og is None:
            return
        
        global_waypoints_vis = []
        for waypoint in self.glb_wpnts_og.wpnts:
            global_waypoints_vis.append([waypoint.x_m, waypoint.y_m, waypoint.s_m])
        
        n_sectors = self.sectors_params['n_sectors']
        sec_markers = MarkerArray()

        for i in range(n_sectors):
            s = self.sectors_params[f"Sector{i}"]['start']
            if s == (len(global_waypoints_vis) - 1):
                theta = np.arctan2((global_waypoints_vis[0][1] - global_waypoints_vis[s][1]),(global_waypoints_vis[0][0] - global_waypoints_vis[s][0]))
            else:
                theta = np.arctan2((global_waypoints_vis[s+1][1] - global_waypoints_vis[s][1]),(global_waypoints_vis[s+1][0] - global_waypoints_vis[s][0]))
            quaternions = quaternion_from_euler(0, 0, theta)
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.ARROW
            marker.scale.x = 0.5
            marker.scale.y = 0.05
            marker.scale.z = 0.15
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.position.x = global_waypoints_vis[s][0]
            marker.pose.position.y = global_waypoints_vis[s][1]
            marker.pose.position.z = 0.0
            marker.pose.orientation.x = quaternions[0]
            marker.pose.orientation.y = quaternions[1]
            marker.pose.orientation.z = quaternions[2]
            marker.pose.orientation.w = quaternions[3]
            marker.id = i
            sec_markers.markers.append(marker)

            marker_text = Marker()
            marker_text.header.frame_id = "map"
            marker_text.header.stamp = self.get_clock().now().to_msg()
            marker_text.type = marker_text.TEXT_VIEW_FACING
            marker_text.text = f"Start Sector {i}"
            marker_text.scale.z = 0.4
            marker_text.color.r = 0.2
            marker_text.color.g = 0.1
            marker_text.color.b = 0.1
            marker_text.color.a = 1.0
            marker_text.pose.position.x = global_waypoints_vis[s][0]
            marker_text.pose.position.y = global_waypoints_vis[s][1]
            marker_text.pose.position.z = 1.5
            marker_text.pose.orientation.x = 0.0
            marker_text.pose.orientation.y = 0.0
            marker_text.pose.orientation.z = 0.0436194
            marker_text.pose.orientation.w = 0.9990482
            marker_text.id = i + n_sectors
            sec_markers.markers.append(marker_text)
        self.sector_visualization_pub.publish(sec_markers)


def main():
    rclpy.init()
    node = SectorTuner()
    rclpy.spin(node)
    rclpy.shutdown()
