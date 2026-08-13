# perception

UNITA LiDAR obstacle detection and tracking
to the Python-only utility libraries in `roboracer_unita_ws`.

## Data flow

```text
/scan + /global_waypoints_scaled + TF(map <- laser)
  -> detect
  -> /detect/raw_obstacles
  -> tracking_node
  -> /tracking/obstacles
  -> spline / spline_planner
```

## Run

```bash
ros2 run perception detect
ros2 run perception tracking_node
```

Both nodes require `/global_waypoints_scaled`. `detect` also requires the
`map <- laser` transform and `/scan`.
