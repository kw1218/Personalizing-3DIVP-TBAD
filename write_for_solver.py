import sys
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pyvista as pv
import pandas as pd
from glob import glob
from tqdm import tqdm
from scipy.interpolate import interp1d
import utils as ut
import descriptors_utils as dut
from scipy import interpolate

#-----------------------------------------------------------------------------------------------------------------------
## Options
Name = 'test' # name of the case
profilesDir = r'D:/Scaled/' + Name + '/'  # path to the folder containing the resampled .vtp files
geoDir = r'D:/stl/' + Name + '_aorta.stl' # path to the aorta stl file for aligning directions
outputDir = r'D:/SolverOutput/output/' + Name + '/'

cfd_delta_t = 0.001  # simulation time steps
cardiac_cycle_period = 0.82 # period of cardiac cycle
time_interpolation = 'cubic' # can be linear, nearest, quadratic, ...
solver = 'cfx' # cfx 


## determine the direction of aorta
aorta = pv.read(geoDir)
aorta_pts = aorta.points
aorta_max = aorta_pts.max(0)
aorta_min = aorta_pts.min(0)
aorta_length = aorta_max - aorta_min
sequence = ['xx','xx','xx']
sorted_with_indices = sorted(enumerate(aorta_length), key = lambda x: x[1])   # sort the length of aorta in x,y,z direction
#, and put it in the ascending order
indices = [item[0] for item in sorted_with_indices]
sequence[indices[0]]= 'rl'
sequence[indices[1]]= 'ap'
sequence[indices[2]]= 'fh'
# sequence = ['rl','ap','fh']





#-----------------------------------------------------------------------------------------------------------------------
## Prepare variables
interp_planes = [pv.read(fn) for fn in sorted(glob(osp.join(profilesDir, '*.vtp')))]
num_frames = len(interp_planes)

os.makedirs(outputDir, exist_ok=True)

# tcfd = np.arange(0, cardiac_cycle_period, cfd_delta_t)
tcfd = np.arange(0, cardiac_cycle_period , cfd_delta_t)
timepoints = len(tcfd)
t4df = np.linspace(0, cardiac_cycle_period, num_frames)
pos = interp_planes[0].points
npts = pos.shape[0]
vel4df = np.array([interp_planes[k]['Velocity'] for k in range(len(interp_planes))]) #change to lower case v in 'velocity' if using vtk files previously created in matlab
velcfd = interp1d(t4df, vel4df, axis=0, kind=time_interpolation)(tcfd)



if solver == 'cfx':
    cfx_arr = pos[:, :2] #taking x and y from position table
    cfx_arr = np.tile(cfx_arr,reps=(timepoints,1)) #repeat whole array of numbers x times (not repeating individual elements)

    time_stem = np.repeat(tcfd,npts,axis=0) # repeat individual elements of tcfd
    time_stem = time_stem.reshape(-1,1) # reshape to one dimension ahead of concatenation

    cfx_arr = np.concatenate((cfx_arr, time_stem),axis=1) # combine position and time arrays

    df_list = []
    assert velcfd.shape[2] == 3
    #for i in range(velcfd_sampled.shape[2]):
    for i, direction in enumerate(sequence): #CHECK ORDER OF DIRECTIONS ARE CORRECT FOR INDIVIDUAL PATIENT #og was rl, ap, fh
        #creating dataframe with required 7 lines of text for cfx to read the file
        df_header = pd.DataFrame(
            {
                '1': ['[Name]','InletV' + direction, None, '[Spatial Fields]', 'x', None, '[Data]', 'x[m]'],
                '2': [None, None, None, None, 'y', None, None, 'y[m]'],
                '3': [None, None, None, None, 't', None, None, 't[s]'],
                '4': [None, None, None, None, None, None, None, 'Velocity[m s^-1]']
            }
        )
        vel_arr = velcfd[:,:,i]
        vel_arr = vel_arr.reshape(-1,1) # single column format
        output_arr = np.concatenate((cfx_arr, vel_arr), axis=1)
        output_df = pd.DataFrame(output_arr,columns=['1','2','3','4'])
        output_df = pd.concat((df_header,output_df), axis=0)
        output_df.to_csv(outputDir + '/vel' + direction.upper() + '.csv',index=False,header=False)


if solver == 'cfx_xyz':
    cfx_arr = pos[:,:3] #taking x and y and z from position table
    cfx_arr = np.tile(cfx_arr,reps=(timepoints,1)) #repeat whole array of numbers x times (not repeating individual elements)

    time_stem = np.repeat(tcfd,npts,axis=0) # repeat individual elements of tcfd
    time_stem = time_stem.reshape(-1,1) # reshape to one dimension ahead of concatenation

    cfx_arr = np.concatenate((cfx_arr, time_stem),axis=1) # combine position and time arrays

    df_list = []
    assert velcfd.shape[2] == 3
    #for i in range(velcfd_sampled.shape[2]):
    sequence = ['rl','ap','fh']
    for i, direction in enumerate(sequence): #CHECK ORDER OF DIRECTIONS ARE CORRECT FOR INDIVIDUAL PATIENT #og was rl, ap, fh
        #creating dataframe with required 7 lines of text for cfx to read the file
        df_header = pd.DataFrame(
            {
                '1': ['[Name]','InletV' + direction, None, '[Spatial Fields]', 'x', None, '[Data]', 'x[m]'],
                '2': [None, None, None, None, 'y', None, None, 'y[m]'],
                '3': [None, None, None, None, 'z', None, None, 'z[m]'],
                '4': [None, None, None, None, 't', None, None, 't[s]'],
                '5': [None, None, None, None, None, None, None, 'Velocity[m s^-1]']
            }
        )
        vel_arr = velcfd[:,:,i]
        vel_arr = vel_arr.reshape(-1,1)
        output_arr = np.concatenate((cfx_arr, vel_arr), axis=1)
        output_df = pd.DataFrame(output_arr,columns=['1','2','3','4','5'])
        output_df = pd.concat((df_header,output_df), axis=0)
        output_df.to_csv(outputDir + '/vel' + direction.upper() + '.csv',index=False,header=False)



if solver == 'fluent':
    # write .prof for ansys fluent
    xx, yy, zz = pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist()
    fu = np.swapaxes(velcfd[:, :, 0], 0, 1)
    fv = np.swapaxes(velcfd[:, :, 1], 0, 1)
    fw = np.swapaxes(velcfd[:, :, 2], 0, 1)
    for i in tqdm(range(len(tcfd))):
        with open(osp.join(outputDir, saveName + '_{:05d}.prof'.format(i)), 'w') as fn:
            fn.write('((velocity point {})\n'.format(npts))
            fn.write('(x\n')
            for xi in xx:
                fn.write(str(xi) + '\n')
            fn.write(')\n')
            fn.write('(y\n')
            for yi in yy:
                fn.write(str(yi) + '\n')
            fn.write(')\n')
            fn.write('(z\n')
            for zi in zz:
                fn.write(str(zi) + '\n')
            fn.write(')\n')
            fn.write('(u\n')
            for ui in fu[:, i]:
                fn.write(str(ui) + '\n')
            fn.write(')\n')
            fn.write('(v\n')
            for vi in fv[:, i]:
                fn.write(str(vi) + '\n')
            fn.write(')\n')
            fn.write('(w\n')
            for wi in fw[:, i]:
                fn.write(str(wi) + '\n')
            fn.write(')\n')
            fn.write(')')
print("files written.")
