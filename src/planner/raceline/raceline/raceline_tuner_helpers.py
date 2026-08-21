#!/usr/bin/env python3
"""Geometry the raceline tuner edits waypoints with.

Lifted from UNICORN's gb_optimizer/global_trajectory_tuner_helpers.py, minus
the two thirds of it the tuner never calls. Dropping those takes the skimage,
trajectory_planning_helpers and savgol_filter imports with them - none of the
eight functions below needs anything but numpy and scipy.interpolate, and a
tool that is only run between sessions should not pull half the optimizer in
to start.

Removed, all of them unreferenced from the tuner: OnOff, Vel_Offset,
Vel_Weight, cal_slope, cal_yaw, calc_curv, calculate_ey,
distance_between_int_markers, fined_wall, insert_new_points, is_pivots_nan,
path_contain_zero_wp, set_lookahead.

No ROS here on purpose - it is all arrays in, arrays out, which is what makes
it testable without an executor.

The anchors are (index, x, y) triples taken off the interactive markers, and
every function is written to wrap: anchor1 may sit at a HIGHER index than
anchor2 and the span still means "the short way round", which is why each one
starts by working out which side of the seam it is on.
"""

import numpy as np
from scipy import interpolate


def straighten_2d(anchor1, anchor2, data_2d):
    if anchor1 is None or anchor2 is None:
        return data_2d

    start_idx = None
    num_idx = None
    start_data = None
    diff_data = None

    if anchor1[0] == anchor2[0]:
        return data_2d
    elif anchor1[0] > anchor2[0]:
        if (anchor1[0] - anchor2[0]) > len(data_2d) / 2:
            start_idx = anchor1[0]
            num_idx = len(data_2d) - (anchor1[0] - anchor2[0])
            start_data = anchor1[1:3]
            diff_data = [anchor2[1] - anchor1[1], anchor2[2] - anchor1[2]]
        else:
            start_idx = anchor2[0]
            num_idx = anchor1[0] - anchor2[0]
            start_data = anchor2[1:3]
            diff_data = [anchor1[1] - anchor2[1], anchor1[2] - anchor2[2]]
    else:
        if (anchor2[0] - anchor1[0]) > len(data_2d) / 2:
            start_idx = anchor2[0]
            num_idx = len(data_2d) - (anchor2[0] - anchor1[0])
            start_data = anchor2[1:3]
            diff_data = [anchor1[1] - anchor2[1], anchor1[2] - anchor2[2]]
        else:
            start_idx = anchor1[0]
            num_idx = anchor2[0] - anchor1[0]
            start_data = anchor1[1:3]
            diff_data = [anchor2[1] - anchor1[1], anchor2[2] - anchor1[2]]

    for i in range(num_idx + 1):
        idx = (start_idx + i) % len(data_2d)
        data_2d[idx][0] = start_data[0] + i * diff_data[0] / num_idx
        data_2d[idx][1] = start_data[1] + i * diff_data[1] / num_idx

    return data_2d


def straighten_1d(anchor1, anchor2, data_1d):
    if anchor1 is None or anchor2 is None:
        return data_1d

    start_idx = None
    num_idx = None
    start_data = None
    diff_data = None

    if anchor1[0] == anchor2[0]:
        return data_1d
    elif anchor1[0] > anchor2[0]:
        if (anchor1[0] - anchor2[0]) > len(data_1d) / 2:
            start_idx = anchor1[0]
            num_idx = len(data_1d) - (anchor1[0] - anchor2[0])
            start_data = anchor1[3]
            diff_data = anchor2[3] - anchor1[3]
        else:
            start_idx = anchor2[0]
            num_idx = anchor1[0] - anchor2[0]
            start_data = anchor2[3]
            diff_data = anchor1[3] - anchor2[3]
    else:
        if (anchor2[0] - anchor1[0]) > len(data_1d) / 2:
            start_idx = anchor2[0]
            num_idx = len(data_1d) - (anchor2[0] - anchor1[0])
            start_data = anchor2[3]
            diff_data = anchor1[3] - anchor2[3]
        else:
            start_idx = anchor1[0]
            num_idx = anchor2[0] - anchor1[0]
            start_data = anchor1[3]
            diff_data = anchor2[3] - anchor1[3]

    for i in range(num_idx + 1):
        idx = (start_idx + i) % len(data_1d)
        data_1d[idx] = start_data + i * diff_data / num_idx

    return data_1d


def Vel_Set(anchor1, anchor2, vel, data_1d):
    if anchor1 is None or anchor2 is None:
        return data_1d

    start_idx = None
    num_idx = None

    if anchor1[0] == anchor2[0]:
        return data_1d
    elif anchor1[0] > anchor2[0]:
        if (anchor1[0] - anchor2[0]) > len(data_1d) / 2:
            start_idx = anchor1[0]
            num_idx = len(data_1d) - (anchor1[0] - anchor2[0])
        else:
            start_idx = anchor2[0]
            num_idx = anchor1[0] - anchor2[0]
    else:
        if (anchor2[0] - anchor1[0]) > len(data_1d) / 2:
            start_idx = anchor2[0]
            num_idx = len(data_1d) - (anchor2[0] - anchor1[0])
        else:
            start_idx = anchor1[0]
            num_idx = anchor2[0] - anchor1[0]

    for i in range(num_idx + 1):
        idx = (start_idx + i) % len(data_1d)
        data_1d[idx] = vel

    return data_1d


def cal_unit_vec(yaw):
    return np.array([np.cos(yaw), np.sin(yaw)])


def sampleCubicSplinesWithDerivative(reference, data, resolution, target, scale):
    '''
    Compute and sample the cubic splines for a set of input points with
    optional information about the tangent (direction AND magnitude). The 
    splines are parametrized along the traverse line (piecewise linear), with
    the resolution being the step size of the parametrization parameter.
    The resulting samples have NOT an equidistant spacing.

    Arguments:      points: a list of n-dimensional points
                    tangents: a list of tangents
                    resolution: parametrization step size
    Returns:        samples

    Notes: Lists points and tangents must have equal length. In case a tangent
        is not specified for a point, just pass None. For example:
                    points = [[0,0], [1,1], [2,0]]
                    tangents = [[1,1], None, [1,-1]]

    '''    # print(self.server.get(marker_name))
    ref = reference[1:]
    ref_back = data[(reference[0] - resolution) % len(data)]
    ref_forw = data[(reference[0] + resolution) % len(data)]
    # tan=self.cal_unit_vec(self.server.get(marker_name).controls[0].markers[0].color.r)
    # tan1=self.cal_unit_vec(self.server.get('wp'+str((reference[0]-resolution)%self.track_len)).controls[0].markers[0].color.r)
    # tan2=self.cal_unit_vec(self.server.get('wp'+str((reference[0]+resolution)%self.track_len)).controls[0].markers[0].color.r)

    points = []
    tangents = []
    if target == "Pose":
        points.append(ref_back[0:2])
        tangents.append(cal_unit_vec(ref_back[2]))
        points.append(ref[0:2])
        tangents.append(cal_unit_vec(data[reference[0] % len(data)][2]))
        points.append(ref_forw[0:2])
        tangents.append(cal_unit_vec(ref_forw[2]))

    elif target == "Vel":
        vec1 = np.array([1, (data[(reference[0] - resolution + 1) % len(data)] - \
                        data[(reference[0] - resolution - 1) % len(data)]) / 2])
        vec = np.array([1, (data[(reference[0] + 1) % len(data)] - \
                       data[(reference[0] - 1) % len(data)]) / 2])
        vec2 = np.array([1, (data[(reference[0] + resolution + 1) % len(data)] - \
                        data[(reference[0] + resolution - 1) % len(data)]) / 2])

        points.append([0, ref_back])
        tangents.append(vec1 / np.linalg.norm(vec1))
        points.append([resolution, ref[2]])
        tangents.append(vec / np.linalg.norm(vec))
        points.append([resolution * 2, ref_forw])
        tangents.append(vec2 / np.linalg.norm(vec2))

    tangents = np.dot(tangents, scale * np.eye(2))
    points = np.asarray(points)
    nPoints, dim = points.shape

    # Parametrization parameter s.
    dp = np.diff(points, axis=0)                 # difference between points
    dp = np.linalg.norm(dp, axis=1)              # distance between points
    d = np.cumsum(dp)                            # cumsum along the segments
    d = np.hstack([[0], d])                       # add distance from first point
    l = d[-1]                                    # length of point sequence
    nSamples = resolution * 2 + 1                 # number of samples
    s, r = np.linspace(0, l, nSamples, retstep=True)  # sample parameter and step

    # Bring points and (optional) tangent information into correct format.
    assert (len(points) == len(tangents))
    spline_result = np.empty([nPoints, dim], dtype=object)
    for i, ref in enumerate(points):
        t = tangents[i]
        # Either tangent is None or has the same
        # number of dimensions as the point ref.
        assert (t is None or len(t) == dim)
        fuse = list(zip(ref, t) if t is not None else zip(ref,))
        spline_result[i, :] = fuse

    # Compute splines per dimension separately.
    samples = np.zeros([nSamples, dim])
    for i in range(dim):
        poly = interpolate.BPoly.from_derivatives(d, spline_result[:, i])
        samples[:, i] = poly(s)

    for i in range(resolution * 2 + 1):
        if target == "Pose":
            data[(reference[0] - resolution + i) % len(data)][0] = samples[i][0]
            data[(reference[0] - resolution + i) % len(data)][1] = samples[i][1]
        elif target == "Vel":
            data[(reference[0] - resolution + i) % len(data)] = samples[i][1]

    return data


def entire_traj_translation(reference, data_2d):
    diff = np.array([reference[1] - data_2d[reference[0]][0],
                    reference[2] - data_2d[reference[0]][1]])
    data_2d += diff

    # for i in range(len(data_2d)):
    #     data_2d
    #     idx = (start_idx+i)%len(data_1d)
    #     data_1d[idx] = vel

    return data_2d


def rotate_point(point, angle, center):
    # Calculate the angle in radians

    # Translate the point to the origin
    translated_x = point[0] - center[0]
    translated_y = point[1] - center[1]

    # Perform the rotation
    rotated_x = translated_x * np.cos(angle) - translated_y * np.sin(angle)
    rotated_y = translated_x * np.sin(angle) + translated_y * np.cos(angle)

    # Translate the point back to its original position
    result_x = rotated_x + center[0]
    result_y = rotated_y + center[1]

    return result_x, result_y


def entire_traj_rotation(anchor1, anchor2, data_2d):
    dis_sqr_1 = (data_2d[anchor1[0]][0] - anchor1[1])**2 + (data_2d[anchor1[0]][1] - anchor1[2])**2
    dis_sqr_2 = (data_2d[anchor2[0]][0] - anchor2[1])**2 + (data_2d[anchor2[0]][1] - anchor2[2])**2
    if dis_sqr_1 < dis_sqr_2:
        rot_center = [anchor1[1], anchor1[2]]
        ref1_point = [data_2d[anchor2[0]][0], data_2d[anchor2[0]][1]]
        ref2_point = [anchor2[1], anchor2[2]]
    else:
        rot_center = [anchor2[1], anchor2[2]]
        ref1_point = [data_2d[anchor1[0]][0], data_2d[anchor1[0]][1]]
        ref2_point = [anchor1[1], anchor1[2]]

    ref1_vec = [ref1_point[0] - rot_center[0], ref1_point[1] - rot_center[1]]
    ref2_vec = [ref2_point[0] - rot_center[0], ref2_point[1] - rot_center[1]]
    dot = ref1_vec[0] * ref2_vec[0] + ref1_vec[1] * ref2_vec[1]
    cro = ref1_vec[0] * ref2_vec[1] - ref1_vec[1] * ref2_vec[0]
    mag_1 = np.sqrt(ref1_vec[0]**2 + ref1_vec[1]**2)
    mag_2 = np.sqrt(ref2_vec[0]**2 + ref2_vec[1]**2)

    # Calculate the rotation angle
    if cro > 0:
        angle = np.arccos(dot / (mag_1 * mag_2))
    else:
        angle = -np.arccos(dot / (mag_1 * mag_2))
    # Use one of the points as the center of rotation (e.g., m1_int_pos)

    for i in range(len(data_2d)):
        # marker_name = "wp" + str(i)
        # if i == m1_id or i==m2_id:
        #     ori_pose=pub_track.track.markers[i].pose
        # else:
        #     ori_pose = pub_track.server.get(marker_name).pose
        data_2d[i][0], data_2d[i][1] = rotate_point(data_2d[i], angle, rot_center)
    return data_2d
