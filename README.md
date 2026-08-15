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

**Decide where the workspace lives before creating the venv.** A venv bakes
absolute paths into `bin/activate`, every console script's shebang, and
editable-install `.pth` files. Move it afterwards and `source
.venv/bin/activate` appears to succeed while doing nothing.

```bash
# prerequisites: ROS 2 Humble, rosdep initialised
sudo apt install python3-venv python3-pip

git clone https://github.com/hyunyoung1018/roboracer_unita_ws.git
cd roboracer_unita_ws

# --system-site-packages is mandatory, or rclpy is invisible to the venv
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# nav2, cartographer, urg_node, rviz, skimage, opencv, ...
rosdep install --from-paths src --ignore-src -y

# the vendored simulator, without its renderer
pip install -e src/f1tenth_gym_ros/f1tenth_gym --no-deps
pip install "numpy<2" "gymnasium>=0.29.1,<0.30" "numba>=0.59.0,<0.61" \
            "pandas>=2.0.0" "pillow>=9.1.0" "requests>=2.31.0" \
            "scipy>=1.13.0" "yamldataclassconfig>=1.5.0,<2"

# the raceline optimiser's solvers
pip install --no-deps -r requirements.txt

# range_libc for the particle filter. A Cython extension, not a pip package.
pip install cython
git clone https://github.com/f1tenth/range_libc /tmp/range_libc
cd /tmp/range_libc/pywrapper && pip install --no-build-isolation . && cd -

python -m colcon build --symlink-install
source install/setup.bash

# verify
python -c "from f1tenth_gym.envs import F110Env; print('gym OK')"
python -c "from raceline.raceline_generator import trajectory_optimizer; print('raceline OK')"
```

Every pin and flag above is load-bearing:

- **`numpy<2`.** `skimage` and `opencv` come from apt, built against numpy 1.x.
  A pip numpy 2 in the venv shadows apt's copy and breaks all of them at
  import — `numpy.dtype size changed` from skimage, `_ARRAY_API not found`
  from cv2. Nothing here needs numpy 2.
- **`numba<0.61`.** 0.61 requires numpy 2. 0.66 also wants a newer `coverage`
  than apt ships and dies on `module 'coverage' has no attribute 'types'`.
- **`transforms3d` from pip**, in `requirements.txt`: apt's copy uses
  `np.float`, removed in numpy 1.24, and every node touching
  `tf_transformations` dies on it.
- **Both `--no-deps`.** Without them pip resolves the gym's and the
  optimiser's declared ranges and pulls numpy 2 straight back in.
- **`--no-build-isolation` for range_libc.** The repo has no `pyproject.toml`,
  so pip would build in a clean environment where the cython just installed is
  invisible. Do not use its `compile.sh` — it runs `sudo` and installs to the
  system python. For the much faster GPU ray casting on the Jetson,
  `WITH_CUDA=ON pip install --no-build-isolation .` and set
  `range_method: 'rmgpu'` in `config/car/pf.yaml`.
- **`python -m colcon build`, never bare `colcon`.** Bare colcon runs the
  system install with a `/usr/bin/python3` shebang and bakes that interpreter
  into every generated node script, which then cannot see the venv. Rebuilding
  does not undo it — the wrong shebang is already written. Recover with
  `rm -rf build install log` and build again.

With `--symlink-install`, python and yaml edits take effect on the next launch
with no rebuild. Adding a *new* file to a directory installed whole (`config/`,
`launch/`, `maps/`) still needs one, and so does anything C++.

## Running

```bash
# simulator
ros2 launch f1tenth_gym_ros unita_gym_bridge_launch.py

# map a track, then close it
ros2 launch stack_master mapping.launch.xml map:=<name>
ros2 service call /finish_mapping std_srvs/srv/Trigger {}

# raceline from that map. The first run waits 30 s for an RViz
# "2D Pose Estimate": click the start line pointing the way the car drives.
# It is saved to track_meta.yaml and reused.
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
Publish `/initialpose` from RViz, or `foxglove:=true` and connect to port 8765.
A tight particle cloud on `/pf/viz/particles` means it has converged.

| Argument | Default | Notes |
|---|---|---|
| `safety_width` | `0.7` | `0.45` on `26_inu_track_6x12`, which is 0.9 m wide |
| `sectors` | `true` | `false` to redo only the raceline |
| `rviz` / `foxglove` | `true` / `false` | RViz on the Jetson costs real CPU |

## Watch out

- **Speed lives in the map**, not the code: `maps/<name>/speed_scaling.yaml`.
  Its sectors are waypoint index ranges tied to where the raceline starts, so
  regenerating from a different start pose moves every boundary.
- **`t_clip_min` moves with the speed scaling.** The L1 lookahead sits on that
  floor for most of a lap, which is what makes the line steady; leave it
  behind and the car weaves. Give each sector its own `t_clip_min` in
  `maps/<name>/speed_scaling.yaml` (all sectors or none) and the pairing holds
  per sector; otherwise the single fallback in `config/car/controller.yaml`
  applies to the whole lap.
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
