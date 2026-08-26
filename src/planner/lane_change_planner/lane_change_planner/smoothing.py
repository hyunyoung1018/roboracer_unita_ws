"""Small dependency-free curvature-corrected moving-average smoother."""

import numpy as np


def ccma_smooth(points, moving_average_window=7, curvature_window=3):
    """Smooth an open polyline while preserving its endpoints and corner shape.

    This retains the two-stage intent of CCMA (moving average followed by a
    curvature correction) without adding a pip dependency to the vehicle image.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < max(5, moving_average_window):
        return points.copy()

    window = max(3, int(moving_average_window) | 1)
    half = window // 2
    padded = np.pad(points, ((half, half), (0, 0)), mode='edge')
    kernel = np.ones(window, dtype=float) / window
    smooth = np.column_stack([
        np.convolve(padded[:, axis], kernel, mode='valid')
        for axis in range(2)
    ])

    # Add back a bounded fraction of the local curvature detail lost by the MA.
    detail = points - smooth
    correction_window = max(1, int(curvature_window))
    correction = np.zeros_like(detail)
    for index in range(1, len(points) - 1):
        lo = max(0, index - correction_window)
        hi = min(len(points), index + correction_window + 1)
        correction[index] = np.mean(detail[lo:hi], axis=0)
    smooth += 0.35 * correction
    smooth[0] = points[0]
    smooth[-1] = points[-1]
    return smooth
