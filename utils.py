#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:49:46 2020

@author: winston
"""
import scipy.io
import os
import librosa
import numpy as np
from scipy.stats import zscore
from scipy.io import loadmat


def wavNoisy_gen(cleanfile, noisefile, sr_clean, sr_noise, SNR, output_dir, maxVolumn=False):
    """
    This function mixes noise and clean speech to generate noisy speech 
    Args:
        cleanfile$ (str): clean speech file path
        noisefile$ (str): noise file path (e.g., different noise types like babble, white noise, ...etc)
         sr_clean$ (int): sample rate of clean speech
         sr_noise$ (int): sample rate of noise file
              SNR$ (int): SNR of the generated noisy speech in dB unit
       output_dir$ (str): output direction of the generated noisy speech     
    """
    y_clean, _srClean = librosa.load(cleanfile , sr_clean, mono=True)
    y_noise, _srNoise = librosa.load(noisefile , sr_noise, mono=True)

    # because (nosie & clean) wave file have different time length and sample rate
    if len(y_noise) < len(y_clean): 
        tmp = (len(y_clean) // len(y_noise)) + 1
        y_noise = np.array([x for j in [y_noise] * tmp for x in j])
        y_noise = y_noise[:len(y_clean)]
    else:
        y_noise = y_noise[:len(y_clean)]

    # Noise generate
    if np.std(y_noise) != 0:
        pwr_clean = sum(abs(y_clean)**2) / len(y_clean)
        y_noise = y_noise - np.mean(y_noise)
        noise_variance = pwr_clean / (10**(SNR / 10))
        noise = np.sqrt(noise_variance) * y_noise / np.std(y_noise)
        y_noisy = y_clean + noise

        # Ouput Noisy wave file
        sname_clean = cleanfile.split('/')[-1].split('.wav')[0]
        sname_noise = noisefile.split('/')[-1].split('.wav')[0]
        save_name = '{}_{}_{}.wav'.format( str(SNR)+'dB', sname_noise, sname_clean )
        if maxVolumn:
            maxv = np.iinfo(np.int16).max
            librosa.output.write_wav( output_dir+save_name, (y_noisy * maxv).astype(np.float), sr_clean )
        else:
            librosa.output.write_wav( output_dir+save_name, y_noisy, sr_clean )
    else:
        print('noise file is silence: '+noisefile)    

def Sp_and_Phase(path, Noisy=False):
    """
    This fuction computes DFT result of the input audio waveform
       Input audio spec: [mono, 16k]
    Default window size: 32ms (512 sample points)
       Default hop size: 16ms (256 sample points, 50% overlapping)
    Args:
            path$ (str): file path of audio
            Noisy$ (bool): 'False' for the clean audio, 'True' for the noisy audio 
    """
    signal, rate  = librosa.load(path, sr=16000)
    signal = signal/np.max(abs(signal))   # scale normalize 
    F = librosa.stft(signal, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    Lp = np.abs(F)
    phase = np.angle(F)
    if Noisy:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp = Lp
    NLp = np.reshape(NLp.T,(1,NLp.shape[1],257))
    return NLp, phase                    

def SP_to_wav(mag, phase):
    """
    This fuction computes inverse DFT which converts magnitude and phase back to audio waveform 
    Args:
              mag$ (np.array): magnitude matrix
            phase$ (np.array): phase matrix 
    """
    Rec = np.multiply(mag , np.exp(1j*phase))
    wav = librosa.istft(Rec,
                        hop_length=256,
                        win_length=512,
                        window=scipy.signal.hamming)
    return wav                             

def Prosodic_feat_process(path, Normalize=False):
    pros_data = loadmat(path)['Audio_data']
    pros_time = pros_data[:,0].reshape((len(pros_data),1))
    pros_data = pros_data[:,1:] # remove time feat
    # select prosodic feature: [Pitch + Intensity]
    pros_data = pros_data[:,[0,1]]
    # feature normalize
    if Normalize:
        pros_data = zscore(pros_data)
        pros_data = np.nan_to_num(pros_data)
    pros_data = np.concatenate((pros_time,pros_data),axis=1)
    return pros_data
    
def Exten_prosodic_feat(spec, pros):
    # Praat Prosodic Feature Interpolation to match size
    pros = pros[:,1:]
    pros_compen = []
    for i in range(len(pros)):
        arr = pros[i].reshape((1,len(pros[i])))
        pros_compen.append( np.repeat(arr,2,axis=0) )
    pros_compen = np.array(pros_compen)
    pros_compen = pros_compen.reshape((pros_compen.shape[0]*pros_compen.shape[1],pros_compen.shape[2]))
    pros_compen = pros_compen.reshape((1, pros_compen.shape[0], pros_compen.shape[1]))
    last_vec = pros_compen[:,-1,:].reshape((1, 1, pros_compen.shape[2] ))
    repeat_times = spec.shape[1] - pros_compen.shape[1]
    if repeat_times>=0:
        append_vec = np.repeat(last_vec,repeat_times,axis=1)
        pros_compen = np.concatenate((pros_compen,append_vec),axis=1)
    else:
        pros_compen = pros_compen[:,:spec.shape[1],:]
    return pros_compen          

def get_train_filepaths(directory, data_type, file_type):
    """
    Args:
        directory$ (str): SE dataset directory
        data_type$ (str): 'clean' or 'noisy'
        file_type$ (str): 'wav' for spec feature, 'mat' for prosodic feature
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for fname in files:
            if (file_type in fname)&(data_type in root):
                filepath = os.path.join(root, fname)
                file_paths.append(filepath)
    return sorted(file_paths)

def shuffle_list(x_old,index):
    x_new=[x_old[i] for i in index]
    return x_new 

def Pitch_VAD(clean_Pitch):
    start_idx = 0
    for j in range(len(clean_Pitch)):
        if clean_Pitch[j]==0:
            start_idx += 1
        else:
            break  
    end_idx = len(clean_Pitch)
    for j in range(len(clean_Pitch)):
        if clean_Pitch[-j]==0:
            end_idx -= 1
        else:
            break         
    return [start_idx,end_idx]    
