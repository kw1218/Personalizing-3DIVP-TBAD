import sys
import os
import os.path as osp
import numpy as np
from glob import glob
import pyvista as pv

import utils as ut
import descriptors_utils as dut
import vtk


#---------------------------------------------------------Set path--------------------------------------------------------
## Options
saveName = 'Test'   # filename of resamples .vtp files
outputDir = r'D:/mapped/' + saveName  # path for saving resampled .vtp files
source_profile_dir = r'D:\Profiles\Case01'  # path to the selected synthetic IVP folder
target_profile_fn = r'D:/stl/' + saveName + '_inlet.stl'  #  path to the stl file of the target plane
flip_normals = False # usually set to True, but might have to change depending on target plane orientation
os.makedirs(outputDir,exist_ok=True)

##---------------------------------------------------------Do Not Change--------------------------------------------------------
intp_options = {
    'zero_boundary_dist': 0.03, # percentage of border with zero velocity (smooth damping at the border)
    'zero_backflow': False, # set all backflow components to zero
    'kernel': 'linear', # RBF interpolation kernel (linear is recommended)
    'smoothing': 0.1, # interpolation smoothing, range recommended [0, 2]
    'degree': 0,
    'hard_noslip':False} # degree of polynomial added to the RBF interpolation matrix

#-----------------------------------------------------------------------------------------------------------------------
## Read data
source_profiles = [pv.read(i) for i in sorted(glob(osp.join(source_profile_dir, '*.vtp')))]
target_plane = pv.read(target_profile_fn)

num_frames = len(source_profiles)
source_pts = [source_profiles[k].points for k in range(num_frames)]  # extract coordinates of each timepoint.
source_coms = [source_pts[k].mean(0) for k in range(num_frames)]  # mean(0) calculates the average value of the first axis (x-axis), here it calculates the center of mass
target_pts = target_plane.points # coordinates of points on the reference plane

############################################# check the unit of the target plane, if it is in mm, then convert it to m
target_pts = target_pts / 1000


target_com = target_pts.mean(0) # centre of mass of points on the reference plane
target_normal = target_plane.compute_normals()['Normals'].mean(0) # computes the average normal vector of the reference plane
leftmost_idx_on_target = np.argmax(target_pts[:,0]) # index of the leftmost point in the target plane w.r.t the subject
# peak flowrate
flowrate_input = dut.compute_flowrate(source_profiles)['Q(t)']
peak = np.argmax(np.abs(flowrate_input))
#

normals = [source_profiles[k].compute_normals()['Normals'].mean(0) for k in range(num_frames)] # computes the average normal vector of the inlet plane at each timepoint

#
if flip_normals: normals = [normals[k] * -1 for k in range(num_frames)]    #  flips the direction of the normal vectors


#-----------------------------------------------------------------------------------------------------------------------
## Align source to target

# center at origin for simplicity
target_pts -= target_com   # move to the origin of the current coordinate
source_pts = [source_pts[k] - source_coms[k] for k in range(num_frames)]

# normalize w.r.t. max coordinate norm/ w.r.t: with respect to
targetmax = np.max(np.sqrt(np.sum(target_pts ** 2, axis=1)))  # calculate the max distance to the origin of each point on the reference plane. it should be the radius of the reference plane
sourcemax = [np.max(np.sqrt(np.sum(source_pts[k] ** 2, axis=1))) for k in range(num_frames)]
ratio = targetmax / sourcemax[1]
pts = [source_pts[k] * ratio for k in range(num_frames)]   # scaling the points by ratio.

# rotate to align normals
Rots = [ut.rotation_matrix_from_vectors(normals[k], target_normal) for k in range(num_frames)]
pts = [Rots[k].dot(pts[k].T).T for k in range(num_frames)]
vel = [Rots[k].dot(source_profiles[k]['Velocity'].T).T for k in range(num_frames)]

# second rotation to ensure consistent in-plane alignment
lm_ids = [np.argmax(source_pts[k][:, 0]) for k in range(num_frames)]
Rots_final = [ut.rotation_matrix_from_vectors(pts[k][lm_ids[k], :], target_pts[leftmost_idx_on_target, :]) for k in range(num_frames)]
pts = [Rots_final[k].dot(pts[k].T).T for k in range(num_frames)]
vel = [Rots_final[k].dot(vel[k].T).T for k in range(num_frames)]


# create new polydatas
aligned_planes = [source_profiles[k].copy() for k in range(num_frames)]
for k in range(num_frames):
    aligned_planes[k].points = pts[k]
    aligned_planes[k]['Velocity'] = vel[k]

# spatial interpolation
interp_planes = ut.interpolate_profiles(aligned_planes, target_pts, intp_options)
interp_planes[3].warp_by_vector(factor=0.005).plot(scalars='Velocity')
# recenter


for k in range(num_frames):
    interp_planes[k].points -= interp_planes[k].points.mean(0) - target_com


#-----------------------------------------------------------------------------------------------------------------------
## Save profiles to .vtp

os.makedirs(outputDir, exist_ok=True)
for k in range(num_frames):
    interp_planes[k].save(osp.join(outputDir, saveName + '_{:02d}.vtp'.format(k)))


