import sys
import os
from os.path import join
import numpy as np
from itertools import groupby
from tqdm import tqdm
import pydicom
from collections import Counter
import re
from scipy import interpolate
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from scipy.spatial import distance
import pyvista as pv
import vtk







# core function for interpolating profiles in 3D space
def interpolate_profiles(aligned_planes, fxdpts, intp_options):
    num_frames = len(aligned_planes)

    # Set boundary vectors to zero
    dr = intp_options['zero_boundary_dist']  # percentage threshold for zero boundary
    edges = [aligned_planes[k].extract_feature_edges().connectivity() for k in range(num_frames)]  # extract edges at each frame.
    # extract_feature_edges().connectivity: extract connectivity information
    large_edge_id = [np.argmax(np.bincount(edges[k]['RegionId'])) for k in range(num_frames)] # ng.argmax: returns the index of the largest count;
    # np.bincount: counts the occurrences of each region ID
    #edge_pts = [edges[k].points[np.where(edges[k]['RegionId'] == large_edge_id[k])] for k in range(num_frames)] # find out 3D coordinates information of large_edge_id
    edge_pts = [aligned_planes[k].extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False).points for k in range(num_frames)]
    dist2edge = [distance.cdist(aligned_planes[k].points, edge_pts[k]).min(axis=1) for k in range(num_frames)] #distance.cdist: computes pairwise distance between 2 group of points
    boundary_ids = [np.where(dist2edge[k] < (dr * dist2edge[k].max()))[0] for k in range(num_frames)] # extract new boundary ID;
    # find out the edge IDs where its distance is shorter than the requirement
    for k in range(num_frames):
        aligned_planes[k]['Velocity'][boundary_ids[k], :] = 0.0 # set to 0 velocity

    # Set backflow to zero
    if intp_options['zero_backflow']: # boolean statement, True or False
        normals = [aligned_planes[k].compute_normals()['Normals'].mean(0) * -1 for k in
                   range(num_frames)]  # Careful with the sign;
        normals = [normals[k] / np.linalg.norm(normals[k]) for k in range(num_frames)]  # np.linalg.norm computes the magnitude/length of a vector
        for k in range(num_frames):
            signs = np.dot(aligned_planes[k]['Velocity'], normals[k])
            aligned_planes[k]['Velocity'][np.where(signs < 0)] = 0.0

    # interpolate velocity profile
    vel_interp = []
    # print('fitting...')
    for k in range(num_frames):
        nnVel = NearestNDInterpolator(aligned_planes[k].points, aligned_planes[k]['Velocity'])(fxdpts)
        I = RBFInterpolator(fxdpts, nnVel,
                            kernel=intp_options['kernel'], smoothing=intp_options['smoothing'],
                            epsilon=1, degree=intp_options['degree'])   #RBF:radial basis function

        vel_interp.append(I(fxdpts))

    # hard no slip condition (double check)
    if intp_options['hard_noslip']:
        for k in range(num_frames):
            vel_interp[k][boundary_ids, :] = 0

    # create new polydatas
    interp_planes = [pv.PolyData(fxdpts).delaunay_2d(alpha=0.1) for _ in range(num_frames)]
    #interp_planes = [pv.PolyData(fxdpts) for _ in range(num_frames)]
    for k in range(num_frames):
        interp_planes[k]['Velocity'] = vel_interp[k]

    return interp_planes

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b) # direction of normal
    c = np.dot(a, b) # degree of rotation
    s = np.linalg.norm(v) # length of v
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def rotation_matrix_from_axis_and_angle(u, theta):
    """:arg u is axis (3 components)
       :arg theta is angle (1 component) obtained by acos of dot prod
    """

    from math import cos, sin

    R = np.asarray([[cos(theta) + u[0] ** 2 * (1 - cos(theta)),
             u[0] * u[1] * (1 - cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1 - cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1] ** 2 * (1 - cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1 - cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1 - cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2] ** 2 * (1 - cos(theta))]])

    return R



def time_interpolation(interp_planes, time_intp_options):
    num_frames = len(interp_planes)
    t_4dflow = np.linspace(0, time_intp_options['T4df'], num_frames)
    t_fxd = np.linspace(0, time_intp_options['T4df'], time_intp_options['num_frames_fxd'])

    U = np.array([np.array(interp_planes[k]['Velocity']) for k in range(num_frames)]) # extract velocity data from each frame on the 4Dflow plane
    vel_t_interp = interpolate.interp1d(t_4dflow, U, kind='cubic', axis=0)(t_fxd) # interpolation from 24 timeframes to 20 timeframes

    new_planes = [interp_planes[0].copy() for _ in range(time_intp_options['num_frames_fxd'])]
    for k in range(len(new_planes)):
        new_planes[k]['Velocity'] = vel_t_interp[k]

    return new_planes

def time_interpolation_extend(interp_planes):
    extend_planes = [interp_planes[0].copy() for _ in range(30)]
    num_planes = len(interp_planes)
    # mean_vel = np.mean([np.mean(interp_planes[k]['Velocity']) for k in range(num_planes)])
    # mean_plane = interp_planes[0].copy()
    # mean_plane['Velocity'] = mean_vel
    for k in range(30):
        if k <20:
            extend_planes[k] = interp_planes[k]
        else:
            # extend_planes[k] = interp_planes[14]
            #extend_planes[k] = mean_plane.copy()
            rand_plane_id = np.random.randint(14, num_planes) 
            extend_planes[k] = interp_planes[rand_plane_id]
    return extend_planes

def ratio_scale(interp_planes, time_intp_options):
    num_frames = len(interp_planes)
    t_4dflow = np.linspace(0, time_intp_options['T4df'], num_frames)
    t_fxd = np.linspace(0, time_intp_options['T4df'], time_intp_options['num_frames_fxd']) # 20 frames by default
    t_sys = time_intp_options['systole_end']
    t_dia = t_fxd-t_sys
    t_sys_target = time_intp_options['tuned_end']

    U = np.array([np.array(interp_planes[k]['Velocity']) for k in range(num_frames)]) # extract velocity data from each frame on the 4Dflow plane
    t_new_sys = t_fxd[:t_sys_target]
    t_new_dia = t_fxd[t_sys_target:]

    vel_t_interp_sys = interpolate.interp1d(t_4dflow[:t_sys+1],U[:t_sys+1], kind='cubic',axis=0)(t_new_sys)
    vel_t_interp_dia = interpolate.interp1d(t_4dflow[t_sys:], U[t_sys:], kind='cubic', axis=0)(t_new_dia)
    vel_t_interp = interpolate.interp1d(t_4dflow, U, kind='cubic', axis=0)(t_fxd) # interpolation from 24 timeframes to 20 timeframes

    new_planes = [interp_planes[0].copy() for _ in range(time_intp_options['num_frames_fxd'])]
    for k in range(len(new_planes)):
        if k < t_sys_target:
            new_planes[k]['Velocity'] = vel_t_interp_sys[k]
        else:
            new_planes[k]['Velocity'] = vel_t_interp_dia[k-12]

    return new_planes

# generate fixed plane points
def set_fixed_points(r_spac=0.05, circ_spac=5):
    r = np.arange(0.0, 2.0 + r_spac, r_spac) # It represents the radial distance from the origin.
    n = np.arange(1, 100 + circ_spac, circ_spac) # It represents the number of points to be placed along each circle
    coordinates = []
    for rr, nn in zip(r, n):
        t = np.linspace(0, 2*np.pi, nn, endpoint=False)
        x = rr * np.cos(t)
        y = rr * np.sin(t)
        coordinates.append(np.c_[x, y])  # create a 2D array of coordinates for each circle
    fxdpts = np.concatenate(coordinates, axis=0)
    fxdpts = np.column_stack((fxdpts, np.zeros(len(fxdpts))))   # This adds a column of zeros to the fxdpts array, representing the z-coordinate of the points (which is set to 0 in this case).

    # landmark in fixed plane
    fxd_lm_id = np.argmax(fxdpts[:, 0])
    fxd_lm = fxdpts[fxd_lm_id]

    return fxdpts, fxd_lm

def set_fixed_points_kh(r=1.0):
    r = np.arange(0.0, r, 0.05) # It represents the radial distance from the origin.
    n = 1 + np.arange(len(r))*5 # It represents the number of points to be placed along each circle
    coordinates = []
    for rr, nn in zip(r, n):
        t = np.linspace(0, 2*np.pi, nn, endpoint=False)
        x = rr * np.cos(t)
        y = rr * np.sin(t)
        coordinates.append(np.c_[x, y])  # create a 2D array of coordinates for each circle
    fxdpts = np.concatenate(coordinates, axis=0)
    fxdpts = np.column_stack((fxdpts, np.zeros(len(fxdpts))))   # This adds a column of zeros to the fxdpts array, representing the z-coordinate of the points (which is set to 0 in this case).

    # landmark in fixed plane
    fxd_lm_id = np.argmax(fxdpts[:, 0])
    fxd_lm = fxdpts[fxd_lm_id]

    return fxdpts, fxd_lm

# autoscaling function


def compute_flowrate(vtps):
    flowRate = []
    for i in range(len(vtps)):
        dummyPD = vtps[0]
        normal = dummyPD.compute_normals()['Normals'].mean(0)  # calculate the mean of each column
        dummyPD['Velocity'] = vtps[i]['Velocity']
        dummyPD = dummyPD.point_data_to_cell_data(pass_point_data=True)
        Q = np.sum(np.dot(dummyPD['Velocity'], normal) * dummyPD.compute_cell_sizes()['Area'])   # ScalarVelocity * area = Q
        flowRate.append(Q)
    flowRate = np.array(flowRate)
    if flowRate[np.argmax(np.abs(flowRate))] < 0:
        flowRate *= -1

    out = {'Q(t)': flowRate, 'Q_mean': np.mean(flowRate), 'Q_max': np.max(flowRate)}
    return out
