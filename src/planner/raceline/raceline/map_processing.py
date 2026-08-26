"""
Occupancy map -> centerline + track bounds.

Ported from ForzaETH race_stack (planner/global_planner/global_planner_utils.py).
Changes from the original:
  * No ROS types and no file writing in here - it is pure geometry, so the node
    layer owns all I/O.
  * `show_plots` renders with matplotlib only when explicitly asked for; the
    original popped blocking TkAgg windows from inside node callbacks.
"""

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.segmentation import watershed

# Importable as a top level name because raceline/__init__.py puts this
# package's directory on sys.path. See the note there.
from global_racetrajectory_optimization import helper_funcs_glob


def interp_track(reftrack: np.ndarray, stepsize: float) -> np.ndarray:
    """Resample a [x, y, w_right, w_left] track to an approximately fixed step."""
    return helper_funcs_glob.src.interp_track.interp_track(
        reftrack=reftrack, stepsize_approx=stepsize)


def compare_direction(alpha: float, beta: float) -> bool:
    """True when two heading angles [rad] point the same way (within +-90 deg)."""
    delta = np.abs(alpha - beta)
    if delta > np.pi:
        delta = 2 * np.pi - delta
    return delta < np.pi / 2


def conv_psi(psi: float) -> float:
    """
    Convert a heading measured from the y-axis (the convention used by
    trajectory_planning_helpers) into one measured from the x-axis (ROS).
    """
    new_psi = psi + np.pi / 2
    if new_psi > np.pi:
        new_psi -= 2 * np.pi
    return new_psi


def extract_centerline(skeleton: np.ndarray, map_resolution: float,
                       cent_length: float = 0.0,
                       min_length_m: float = 5.0) -> np.ndarray:
    """
    Pull the centerline out of the skeletonized binary map, in cells.

    The skeleton of a closed track contains exactly one closed contour running
    down the middle. Any other closed contours come from noise or from rooms
    outside the track, so the shortest closed contour wins.

    `cent_length` is an optional expected centerline length in metres, used to
    reject contours whose length is off by more than 15%. Pass 0.0 to accept any.
    """
    contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    closed_contours = []
    for i, elem in enumerate(contours):
        # A contour with neither child nor parent in the CCOMP hierarchy is closed.
        opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
        if not opened:
            closed_contours.append(elem)

    if len(closed_contours) == 0:
        raise IOError('No closed contour found in the map skeleton. '
                      'The track is probably not closed - check for gaps in the walls.')

    line_lengths = [np.inf] * len(closed_contours)
    for i, cont in enumerate(closed_contours):
        line_length = 0.0
        for k, pnt in enumerate(cont):
            line_length += np.sqrt((pnt[0][0] - cont[k - 1][0][0]) ** 2 +
                                   (pnt[0][1] - cont[k - 1][0][1]) ** 2)
        line_length *= map_resolution
        if cent_length == 0.0 or np.abs(cent_length / line_length - 1.0) < 0.15:
            line_lengths[i] = line_length

    # Anything shorter than min_length_m is not a race track. Without this the
    # shortest contour wins outright, and on a real SLAM map that is a speckle
    # ring rather than the track - the map looks fine, the centerline comes out
    # a few points long, and the failure surfaces much later as a savgol error.
    candidates = [(length, i) for i, length in enumerate(line_lengths)
                  if length != np.inf and length >= min_length_m]

    if not candidates:
        found = sorted(round(x, 2) for x in line_lengths if x != np.inf)
        raise ValueError(
            f'No closed contour longer than {min_length_m} m. Closed contours found: '
            f'{found} m. Either the track is not closed - look for gaps in the walls - '
            f'or these are all noise, in which case raise filter_kernel_size '
            f'(7-9 suits a raw SLAM map) so the speckle is removed before skeletonizing.')

    # Shortest survivor: the track ring yields two contours, the outer and inner
    # side of the one-pixel skeleton loop, and the inner one hugs the centre.
    _, best = min(candidates)
    smallest = np.array(closed_contours[best]).flatten()
    return smallest.reshape(int(len(smallest) / 2), 2)


def smooth_centerline(centerline: np.ndarray) -> np.ndarray:
    """
    Savitzky-Golay smoothing that stays smooth across the wrap-around point.

    A plain savgol pass leaves a kink where the contour closes, because the
    filter has no data past either end. Running it a second time on the track
    rotated by half a lap puts that seam in the middle, where the filter is well
    posed, and the first/last window is taken from that second result.
    """
    n = len(centerline)
    if n < 10:
        raise ValueError(
            f'Centerline has only {n} points - too few to smooth. That is a bad '
            f'contour, not a smoothing problem; see extract_centerline.')

    if n > 2000:
        filter_length = int(n / 200) * 10 + 1
    elif n > 1000:
        filter_length = 81
    elif n > 500:
        filter_length = 41
    else:
        filter_length = 21

    # The window has to fit the data twice over: savgol rejects a window longer
    # than the input, and the wrap-around pass below indexes half + window.
    filter_length = min(filter_length, n // 2)
    if filter_length % 2 == 0:
        filter_length -= 1
    filter_length = max(filter_length, 5)

    smooth = savgol_filter(centerline, filter_length, 3, axis=0)

    half = int(n / 2)
    rotated = np.append(centerline[half:], centerline[0:half], axis=0)
    smooth_rotated = savgol_filter(rotated, filter_length, 3, axis=0)

    smooth[0:filter_length] = smooth_rotated[half:(half + filter_length)]
    smooth[-filter_length:] = smooth_rotated[(half - filter_length):half]
    return smooth


def extract_track_bounds(centerline: np.ndarray, filtered_bw: np.ndarray,
                         map_resolution: float, map_origin, start_pose,
                         show_plots: bool = False):
    """
    Watershed the free space, using the centerline as the seed, to get the left
    and right track boundaries in metres.

    Raises IOError when the flood does not resolve into exactly two closed
    contours (inner + outer wall). The caller is expected to fall back to a
    plain distance transform in that case - that is how the original handles
    maps with rooms or open areas bleeding into the track.
    """
    cent_img = np.zeros((filtered_bw.shape[0], filtered_bw.shape[1]), dtype=np.uint8)
    cv2.drawContours(cent_img, [centerline.astype(int)], 0, 255, 2, cv2.LINE_8)

    _, cent_markers = cv2.connectedComponents(cent_img)

    dist_transform = cv2.distanceTransform(filtered_bw, cv2.DIST_L2, 5)
    labels = watershed(-dist_transform, cent_markers, mask=filtered_bw)

    closed_contours = []
    for label in np.unique(labels):
        if label == 0:
            continue

        mask = np.zeros(filtered_bw.shape, dtype='uint8')
        mask[labels == label] = 255

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for i, cont in enumerate(contours):
            opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
            if not opened:
                closed_contours.append(cont)

        if len(closed_contours) != 2:
            raise IOError('Watershed did not produce exactly two track bounds')

        cv2.drawContours(cent_img, closed_contours, 0, 255, 4)
        cv2.drawContours(cent_img, closed_contours, 1, 255, 4)

    def _to_meter(contour):
        flat = np.array(contour).flatten()
        pts = flat.reshape(int(len(flat) / 2), 2)
        out = np.zeros(np.shape(pts), dtype=float)
        out[:, 0] = pts[:, 0] * map_resolution + map_origin[0]
        out[:, 1] = pts[:, 1] * map_resolution + map_origin[1]
        return out

    # The outer wall is always the longer contour.
    bound_long_meter = _to_meter(max(closed_contours, key=len))
    bound_short_meter = _to_meter(min(closed_contours, key=len))

    # Decide which of the two is on the right of the car by walking the outer
    # bound at the point nearest the start pose and comparing its direction to
    # the car's heading rotated by 180 degrees.
    bound_distance = np.hypot(bound_long_meter[:, 0] - start_pose[0],
                              bound_long_meter[:, 1] - start_pose[1])
    min_dist_ind = int(np.argmin(bound_distance))

    bound_direction = np.angle([complex(
        bound_long_meter[min_dist_ind, 0] - bound_long_meter[min_dist_ind - 1, 0],
        bound_long_meter[min_dist_ind, 1] - bound_long_meter[min_dist_ind - 1, 1])])

    norm_angle_right = start_pose[2] - np.pi
    if norm_angle_right < -np.pi:
        norm_angle_right += 2 * np.pi

    if compare_direction(norm_angle_right, bound_direction):
        bound_right, bound_left = bound_long_meter, bound_short_meter
    else:
        bound_right, bound_left = bound_short_meter, bound_long_meter

    if show_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(bound_right[:, 0], bound_right[:, 1], 'b', label='Right bound')
        ax.plot(bound_left[:, 0], bound_left[:, 1], 'g', label='Left bound')
        ax.plot(centerline[:, 0] * map_resolution + map_origin[0],
                centerline[:, 1] * map_resolution + map_origin[1], 'r', label='Centerline')
        ax.set_aspect('equal', 'datalim')
        ax.legend()
        plt.show()

    return bound_right, bound_left


def dist_to_bounds(trajectory: np.ndarray, bound_r: np.ndarray, bound_l: np.ndarray,
                   reverse: bool = False):
    """
    Nearest distance from every trajectory point to each track boundary.

    `trajectory` is either [s, x, y, psi, kappa, vx, ax] or plain [x, y].
    When `reverse` is set the two results are swapped, because driving the track
    the other way exchanges left and right.
    """
    help_trajectory = trajectory[:, 1:3] if len(trajectory[0]) > 2 else trajectory

    bound_r_int = interp_track(
        np.column_stack((bound_r, np.zeros((bound_r.shape[0], 2)))), stepsize=0.1)
    bound_l_int = interp_track(
        np.column_stack((bound_l, np.zeros((bound_l.shape[0], 2)))), stepsize=0.1)

    n_wpnt = len(help_trajectory)
    dists_right = np.zeros(n_wpnt)
    dists_left = np.zeros(n_wpnt)

    for i, wpnt in enumerate(help_trajectory):
        dists_right[i] = np.amin(np.hypot(bound_r_int[:, 0] - wpnt[0],
                                          bound_r_int[:, 1] - wpnt[1]))
        dists_left[i] = np.amin(np.hypot(bound_l_int[:, 0] - wpnt[0],
                                         bound_l_int[:, 1] - wpnt[1]))

    if reverse:
        return dists_left, dists_right
    return dists_right, dists_left


def add_dist_to_cent(centerline_smooth: np.ndarray, centerline_meter: np.ndarray,
                     map_resolution: float, dist_transform=None,
                     bound_r=None, bound_l=None, reverse: bool = False) -> np.ndarray:
    """
    Build the reference track [x_m, y_m, w_tr_right_m, w_tr_left_m] the optimizer
    imports, by attaching a half-width to every centerline point.

    Prefers the watershed boundaries. The distance-transform path is the
    degraded fallback: it can only produce one symmetric width, so an off-centre
    centerline reports the same width on both sides and the optimizer loses the
    room it actually has on the wider side.
    """
    centerline_comp = np.zeros((len(centerline_meter), 4))

    if bound_r is not None and bound_l is not None:
        width_right, width_left = dist_to_bounds(centerline_meter, bound_r, bound_l,
                                                 reverse=reverse)
    elif dist_transform is not None:
        width_right = dist_transform[centerline_smooth[:, 1].astype(int),
                                     centerline_smooth[:, 0].astype(int)] * map_resolution
        if len(width_right) != len(centerline_meter):
            width_right = np.interp(np.arange(0, len(centerline_meter)),
                                    np.arange(0, len(width_right)), width_right)
        width_left = width_right
    else:
        raise IOError('Neither track bounds nor a distance transform were supplied')

    centerline_comp[:, 0] = centerline_meter[:, 0]
    centerline_comp[:, 1] = centerline_meter[:, 1]
    centerline_comp[:, 2] = width_right
    centerline_comp[:, 3] = width_left
    return centerline_comp
