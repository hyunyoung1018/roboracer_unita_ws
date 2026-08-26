# spline_planner

Multi-obstacle avoidance adapted from UNICORN's `spliner_planner` to the
interfaces currently available in `roboracer_unita_ws`.

The original `BehaviorStrategy`, `PredictionArray`, `grid_filter`, and `ccma`
dependencies are not present here. This port therefore uses
`/tracking/obstacles`, Frenet track bounds, cubic splines, and a Savitzky-Golay
filter. Set `require_overtaking_permission:=true` to gate planning with the
`/ot_section_check` Boolean topic.

Executables:

- `ros2 run spline_planner dynamic_avoidance_node`
- `ros2 run spline_planner update_waypoints`
