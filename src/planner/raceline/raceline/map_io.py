"""
File I/O for the raceline pipeline.

Everything a stage produces lands in the map directory, so any stage can run on
its own as long as its inputs are on disk:

    maps/<map>/
      <map>.png            occupancy map image      (from mapping)
      <map>.yaml           nav2 map metadata        (from mapping)
      track_meta.yaml      start pose + direction   (this module)
      centerline.csv       x_m,y_m,w_tr_right_m,w_tr_left_m
      global_waypoints.json                         (readwrite_global_waypoints)

Two deliberate departures from ForzaETH race_stack:

  * The start pose lives in track_meta.yaml, not inside the nav2 <map>.yaml.
    race_stack writes an `initial_pose` key into the map yaml during mapping,
    which means a map that never went through their mapping node cannot be
    planned on at all. Keeping the nav2 yaml pristine lets any existing map
    enter the pipeline.

  * centerline.csv is written into the map directory. race_stack passes it
    between the planner and the optimizer through ~/.ros/map_centerline.csv,
    an invisible global side channel that makes the two stages impossible to
    run independently.
"""

import csv
import os

import cv2
import numpy as np
import yaml

# Re-exported for the generator's convenience. It lives in paths.py so that
# raceline_publisher can reach it without importing OpenCV - see the note there.
from .paths import resolve_source_dir  # noqa: F401


def load_map(map_dir: str, map_name: str):
    """
    Load the occupancy map image and its nav2 metadata.

    Returns
    -------
    image : np.ndarray
        Map image, uint8, flipped vertically so that array row 0 is the bottom
        of the map. Cell (row, col) then maps to metres as
        origin + resolution * (col, row), matching the nav2 convention.
    resolution : float
        [m/cell]
    origin : tuple[float, float]
        (x, y) of the map origin in metres.
    meta : dict
        The parsed yaml, so binarize() can apply the map's own thresholds.
    """
    img_path = os.path.join(map_dir, map_name + '.png')
    yaml_path = os.path.join(map_dir, map_name + '.yaml')

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f'Map image not found: {img_path}')
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f'Map metadata not found: {yaml_path}')

    # cv2.flip(..., 0) puts the image into the nav2 frame (origin bottom-left).
    image = cv2.flip(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), 0)
    if image is None:
        raise IOError(f'Could not decode map image: {img_path}')

    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    return (image, float(meta['resolution']),
            (float(meta['origin'][0]), float(meta['origin'][1])), meta)


def binarize(image: np.ndarray, meta: dict, filter_kernel_size: int = 0) -> np.ndarray:
    """
    Turn the map image into the binary form the centerline extraction expects:
    free space 255, everything else 0.

    The threshold comes from the map's own `free_thresh`, following the nav2
    convention: occupancy = (255 - pixel) / 255 when negate is 0, and a cell is
    free only when that is below free_thresh.

    This matters more than it looks. A cartographer map has three levels - 0
    occupied, 205 unknown, 254/255 free - and 205 is exactly the free_thresh
    boundary. A naive mid-grey threshold puts unknown on the free side, so every
    never-observed cell becomes drivable, the skeleton grows through it, and the
    centerline comes out of whatever contour that produced. race_stack's mapping
    node makes the same call from the other direction, mapping unknown to
    occupied before thresholding.

    `filter_kernel_size` > 0 applies a morphological opening, which removes
    isolated lidar speckle that would otherwise grow spurious skeleton branches.
    """
    free_thresh = float(meta.get('free_thresh', 0.196))
    negate = int(meta.get('negate', 0))

    occupancy = image.astype(float) / 255.0 if negate else (255.0 - image) / 255.0
    binary = np.where(occupancy < free_thresh, 255, 0).astype(np.uint8)

    if filter_kernel_size and filter_kernel_size > 1:
        kernel = np.ones((filter_kernel_size, filter_kernel_size), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    return binary


def track_meta_path(map_dir: str) -> str:
    return os.path.join(map_dir, 'track_meta.yaml')


def read_track_meta(map_dir: str):
    """
    Read track_meta.yaml. Returns None when the file does not exist yet, which
    is the normal case for a map that has not been planned on before.
    """
    path = track_meta_path(map_dir)
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f) or None


def write_track_meta(map_dir: str, start_pose, reverse: bool, source: str) -> str:
    """
    Persist the start pose so the next run reproduces the same s=0 point and the
    same driving direction without any clicking.

    `source` records where the pose came from (param / rviz / meta / default) so
    a default-origin fallback is visible later instead of looking authoritative.
    """
    path = track_meta_path(map_dir)
    meta = {
        'start_pose': {
            'x': float(start_pose[0]),
            'y': float(start_pose[1]),
            'theta': float(start_pose[2]),
        },
        'reverse': bool(reverse),
        'start_pose_source': str(source),
    }
    os.makedirs(map_dir, exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(meta, f, default_flow_style=False, sort_keys=False)
    return path


def centerline_path(map_dir: str, variant: str = '') -> str:
    name = 'centerline.csv' if not variant else f'centerline_{variant}.csv'
    return os.path.join(map_dir, name)


def write_centerline_csv(map_dir: str, centerline: np.ndarray, variant: str = '') -> str:
    """
    Write the reference track in the format the TUM optimizer imports:
    x_m, y_m, w_tr_right_m, w_tr_left_m.
    """
    path = centerline_path(map_dir, variant)
    os.makedirs(map_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in centerline:
            writer.writerow([row[0], row[1], row[2], row[3]])
    return path
