# spline

UNITA local Frenet spline obstacle avoidance.

The package uses interfaces already present in this workspace:

- `/tracking/obstacles` (`f110_msgs/ObstacleArray`)
- `/car_state/odom_frenet` (`nav_msgs/Odometry`)
- `/global_waypoints` and `/global_waypoints_scaled` (`f110_msgs/WpntArray`)
- `/planner/avoidance/otwpnts` (`f110_msgs/OTWpntArray`)

Run with `ros2 run spline spline_node` after building and sourcing the workspace.
