# roboracer_unita_ws

UNITA's F1TENTH / RoboRacer 2026 stack, built on
[ForzaETH race_stack](https://github.com/ForzaETH/race_stack).

| | |
|---|---|
| Car | Traxxas 1/10, VESC 6 MkVI, Hokuyo UST-10LX, Logitech F710 |
| Computer | Jetson Orin Nano Super (arm64, JetPack 6 / Ubuntu 22.04) |
| ROS | Humble |

Everything runs on the Jetson, raceline generation included. arm64 is the
deployment target, so an x86-only wheel is never acceptable.

## Setup

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m colcon build --symlink-install
source install/setup.bash
```

`python -m colcon build`, not `colcon build`. The plain form runs the system
colcon outside the venv, installs against the wrong interpreter, and the nodes
then fail on imports that work in the shell. Rebuilding does not undo it —
`rm -rf build install log` and start again.

With `--symlink-install`, python and yaml edits take effect on the next launch
with no rebuild. Adding a *new* file to a directory that is installed whole
(`config/`, `launch/`, `maps/`) still needs one, and so does anything C++.

## Running

```bash
# make a map
ros2 launch stack_master mapping.launch.xml map:=<name>

# turn it into a raceline
ros2 launch stack_master raceline_generator.launch.xml map:=<name>

# drive it
ros2 launch stack_master time_trials.launch.xml map:=<name> obstacles:=true
```

`obstacles:=true` adds detection, tracking and the avoidance planner. Without
it the car follows the raceline and ignores everything.

Nothing moves until the joystick says so: **LB** hands control to the stick,
**RB** to the controller, **B** stops. Releasing both publishes zero rather
than repeating the last command, so a crashed node stops the car.

Set the initial pose before expecting anything from perception — until the
particle filter has one there is no `map` frame and the detector is blind.
Publish `/initialpose` from RViz or Foxglove (`foxglove:=true`, port 8765).

## Watch out

- **Speed lives in the map**, not the code: `maps/<name>/speed_scaling.yaml`.
  Its sectors are waypoint index ranges tied to where the raceline starts, so
  regenerating from a different start pose moves every boundary.
- **`t_clip_min` moves with the speed scaling.** The L1 lookahead sits on that
  floor for most of a lap, which is what makes the line steady; leave it
  behind and the car weaves. The pairing is in `config/car/controller.yaml`.
- **One distance, three names.** `evasion_distance`, `raceline_clearance_m`
  and `gb_ego_width_m/2 + lateral_width_m` all describe when the car should
  leave the raceline. They must agree, or it trails obstacles it could pass.
- **Do not develop on the car.** An editor server on the Jetson costs about
  2.5 GB and enough CPU to starve the particle filter. SSH in, edit elsewhere.

## Open

- `state_machine` holds a full core; the loop has not been profiled.
- Static avoidance plans but rarely commits — see the `path_free` refusals.
- Head-to-head (prediction, lane change, opponent tracking) is separate work
  under `head_to_head.launch.xml` and is not covered here.
