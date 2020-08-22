from __future__ import division
import glob
import numpy as np
from scipy.io import savemat


def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


#path = '../test_samples/clean_prosodic_feat/feat_praat/'
path = '../test_samples/noisy_prosodic_feat/feat_praat/'

for filename in glob.glob(path+'*.mfcc'):
    timeStep = 0.032
    Total_time = 0
##############################################
    Mfcc = []
    Mfcc_time_Center = []
    n = 0    
    for _mfcc in open(filename):
        if n==3:
            Mfcc_time = float(_mfcc.split(' ')[-1])
            Total_time = float(_mfcc.split(' ')[1])
            n = n+1
        elif n>3:
            _mfcc = _mfcc.split(' ')
            Mfcc.append(_mfcc[:13])
            Mfcc_time_Center.append(Mfcc_time)
            Mfcc_time = Mfcc_time +timeStep
        else:
            n = n+1
    Mfcc = np.array(Mfcc)
    Mfcc = Mfcc.astype(np.float)  
############################################# 
    filename = filename.split('.mfcc')[0]
    Pitch = []
    Pitch_time_Center  = []
    n = 0
    for _pitch in open(filename+'.pitch'):
        if n==3:
            Pitch_time = float(_pitch.split(' ')[-1])   
            n = n+1
        elif n>3:
            Pitch.append(_pitch)
            Pitch_time_Center.append(Pitch_time)
            Pitch_time = Pitch_time +timeStep
        else:
            n = n+1
    Pitch = np.array(Pitch)
    Pitch = Pitch.astype(np.float)
    Pitch = Pitch.reshape((len(Pitch), 1))
#############################################
    filename = filename.split('.pitch')[0]
    Intensity = [] 
    Intensity_time_Center = []
    n =0 
    for _intensity in open(filename+'.intensity'):
        if n==3:
            Intensity_time = float(_intensity.split(' ')[-1])
            n = n+1
        elif n>3:
            Intensity.append(_intensity)
            Intensity_time_Center.append(Intensity_time)
            Intensity_time = Intensity_time +timeStep
        else:
            n = n+1
    Intensity = np.array(Intensity)
    Intensity = Intensity.astype(np.float)
    Intensity = Intensity.reshape((len(Intensity), 1))
#####################################################
    Mfcc_time_Center = np.array(Mfcc_time_Center)
    Pitch_time_Center = np.array(Pitch_time_Center)
    Intensity_time_Center = np.array(Intensity_time_Center)
    
    TimeCenter = []
    _tempSTART = 0
    _tempEND = _tempSTART + timeStep
    while(_tempEND<Total_time):
        TimeCenter.append((_tempSTART+_tempEND)/2)
        _tempSTART = _tempSTART + timeStep
        _tempEND = _tempEND + timeStep
    TimeCenter = np.array(TimeCenter)

    _index_mfcc = find_closest(Mfcc_time_Center, TimeCenter)
    _index_pitch = find_closest(Pitch_time_Center, TimeCenter)
    _index_intensity = find_closest(Intensity_time_Center, TimeCenter)
    
    TimeCenter = TimeCenter.reshape((len(TimeCenter), 1))
    Audio_data =  np.concatenate((TimeCenter, Pitch[_index_pitch, :], Intensity[_index_intensity, :], Mfcc[_index_mfcc, :] ),axis =1) 
    savemat(filename.replace('feat_praat','feat_mat')+'.mat', {'Audio_data':Audio_data})    


