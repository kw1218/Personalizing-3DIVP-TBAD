import sys
import os
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd
import pyvista as pv
pv.set_plot_theme("Document")
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import descriptors_utils as dut
from scipy import interpolate
from collections import deque
import utils as ut


time_intp_options = {
    'T4df': 1,   # adjust to the real period.  for patients with 4D-flow, it should be the value from 4D-flow.
                                                # For patients without 4D-flow, it should be the same value as expected period
    'Tfxd': 1,   #

    'num_frames_fxd': 20}

Number = 20 #### Adjust this number [0-29] to ensure the same SDR between profile and synthetic flowwaveform ####
Name = 'Test'  # name of the patient
patient_specific_4D = True   # True: point by point scaling; False: general scaling
preprocDir = r'D:/mapped' # Path to mapped .vtp files from inlet_mapping.py
csv_path = r'D:\SV_60\Waveform_60_01.csv'  # path to the selected inlet flowwaveform .csv file
outputDir = r'D:/Scaled/' + Name
filename = Name+ '_scaled_mean_flowrate.csv'    #### Output the mean flowrate for 3EWM tuning
filepath= os.path.join(outputDir, filename)
##---------------------------------------------------------Do Not Change--------------------------------------------------------


os.makedirs(outputDir, exist_ok=True)
#synthetic_planes = [pv.read(fn) for fn in sorted(glob(osp.join(preprocDir, 'XX', '*.vtp')))]  ## select the mapped file folder
synthetic_planes = [pv.read(fn) for fn in sorted(glob(osp.join(preprocDir, Name, '*.vtp')))]  ## select the mapped file folder



## read flow waveform from Matlab output
tuned_flowrate_csv = pd.read_csv(csv_path, header=None)
tuned_flowrate_csv.rename(columns={0:'Velocity'},inplace=True)
tuned_velocity = tuned_flowrate_csv['Velocity']
t_tuned = np.linspace(0,1,len(tuned_velocity))
t_new = np.linspace(0,1,20)
tinterp_tuned_velocity = interpolate.interp1d(t_tuned,tuned_velocity,kind = 'cubic', axis=0)(t_new)
flowrate_tuned = tinterp_tuned_velocity
SV_tuned =[flowrate_tuned[0]/20]
for k in range(len(t_new)-1):
    SV_tuned.append((flowrate_tuned[k]+flowrate_tuned[k+1])/2 * 1/20)

total_SV_tuned = np.sum(SV_tuned) * 1000 * 1000 # unit: ml/s
tuned_peak = np.argmax(abs(flowrate_tuned)) # used for following shifting
tuned_min = np.argmin(flowrate_tuned) # used for following shifting


## interpolation on systole and diastole
extended_planes = ut.time_interpolation_extend(synthetic_planes)  # extend to 30 timeframes in case of shorter RCSD (<0.7)
synthetic_plannes_aligned = ut.time_interpolation(extended_planes[:Number],time_intp_options) # interpolate to the specific number of timeframes to scale the SDR

# Peak systole alignment
SV_synthetic_plannes_aligned = dut.compute_flowrate(synthetic_plannes_aligned)['Q(t)']
synthetic_peak = np.argmax(SV_synthetic_plannes_aligned)
q= deque(np.arange(len(synthetic_plannes_aligned)))
circshift = tuned_peak - synthetic_peak
q.rotate(circshift)
synthetic_plannes_aligned = [synthetic_plannes_aligned[int(k)] for k in q]


# Compute SV from the synthetic IVPs
SV_syn = dut.compute_flowrate(synthetic_plannes_aligned)['Q(t)']
SV_area = [SV_syn[0]/20]
for k in range(len(t_new)-1):
    SV_area.append((SV_syn[k] + SV_syn[k+1])/2 * 1/20)

total_SV_syn_chloe = np.sum(SV_area)*1000*1000 # unit ml/s

ratio = np.max(SV_tuned) / np.max(SV_area)




## scale
n=[]

######################   general scaling  ############################
if patient_specific_4D == False:
    for k in range(len(synthetic_plannes_aligned)):
        if k !=0 and k!= len(synthetic_plannes_aligned)-1:
            if  (SV_syn[k]>0 and flowrate_tuned[k] > 0) or (SV_syn[k]<0 and flowrate_tuned[k] < 0):
                synthetic_plannes_aligned[k]['Velocity'] *= ratio
                #synthetic_plannes_aligned[k]['Velocity'] *= 1
                n.append(k)

        else:
            synthetic_plannes_aligned[k]['Velocity'] *= 1
            n.append(k)
        synthetic_plannes_aligned[k].save(osp.join(outputDir, 'scaledProf_{:02d}.vtp'.format(k)))

######################   4D-flow scaling  ############################  
if patient_specific_4D == True:
    ratio_all = np.ones(len(SV_syn))
    for k in range(len(SV_area)):
        if SV_area[k] != 0 and k !=19:
            ratio_test = SV_tuned[k] / SV_area[k]
            synthetic_plannes_aligned[k]['Velocity'] *= ratio_test
                #synthetic_plannes_aligned[k]['Velocity'] *= 1
            n.append(k)
        synthetic_plannes_aligned[k].save(osp.join(outputDir, 'scaledProf_{:02d}.vtp'.format(k)))



# if Number < 20:

#     for k in range(len(synthetic_plannes_aligned)):
#         if k !=0 and k != 19:
#             if  (SV_syn[k]>0 and flowrate_tuned[k] > 0) or (SV_syn[k]<0 and flowrate_tuned[k] < 0):
#                 synthetic_plannes_aligned[k]['Velocity'] *= ratio
#                 n.append(k)
#         else:
#             synthetic_plannes_aligned[k]['Velocity'] *= ratio_all[k]
#             n.append(k)
#         synthetic_plannes_aligned[k].save(osp.join(outputDir, 'scaledProf_{:02d}.vtp'.format(k)))

# else:
#     for k in range(len(synthetic_plannes_aligned)):
#         if k !=0 and k!= 19:
#             if  (SV_syn[k]>0 and flowrate_tuned[k] > 0) or (SV_syn[k]<0 and flowrate_tuned[k] < 0):
#                 synthetic_plannes_aligned[k]['Velocity'] *= ratio
#                 n.append(k)

#         else:
#             synthetic_plannes_aligned[k]['Velocity'] *= ratio_all[k]
#             n.append(k)
#         synthetic_plannes_aligned[k].save(osp.join(outputDir, 'scaledProf_{:02d}.vtp'.format(k)))
            
                






# flow_extended = dut.compute_flowrate(extended_planes)['Q(t)']
# plt.plot(flow_extended,label='extended')
# plt.show()

flow = dut.compute_flowrate(synthetic_plannes_aligned )['Q(t)']
Q_meanflow = dut.compute_flowrate(synthetic_plannes_aligned )['Q_mean']
x_axis = np.zeros(len(flow))
#plt.plot(SV_syn,label='synthetic')
plt.plot(flowrate_tuned,label='tuned Flat ')
plt.plot(flow,label='synthetic scaled')
plt.plot(x_axis)
plt.ylabel('flowrate L/min')
plt.legend()
plt.show()

mean_flowrate = flow.mean()
print('The mean flowrate of the synthetic flowwaveform is:', mean_flowrate)
df3= pd.DataFrame([Q_meanflow])
df3.to_csv(filepath, index=False, header=False, mode='a')


