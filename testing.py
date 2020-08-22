#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:08:40 2020

@author: winston
"""
import numpy as np
import os
from keras.models import load_model
from keras import objectives
from utils import Sp_and_Phase, Prosodic_feat_process, Exten_prosodic_feat, Pitch_VAD
from utils import get_train_filepaths, SP_to_wav
from scipy.io import wavfile
import subprocess
from pystoi.stoi import stoi
from scipy.io.wavfile import read


def spec_loss(input_img, output_img):
    return 10*objectives.mse(input_img,output_img)

def prosodic_loss(input_img, output_img):
    return objectives.mae(input_img,output_img)/10

def parse_pesq_rsl( pesq_rsl ):
    for i in range(len(pesq_rsl.splitlines())):
        if i == len(pesq_rsl.splitlines())-1:
            return pesq_rsl.splitlines()[i]   
###############################################################################




###############################################################################
#  This is a sample code for the proposed prosodic-spec SE model training,    #
#  so the testing set we use here are the same as train/valid set.            #
###############################################################################

# Parameters        
path_dataset = './test_samples/'
path_output = './enhanced_samples/'
model_type = 'FixedConcat'      # 'JointConcat', 'FixedConcat' or 'MultiTask'
VAD = True

# create output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# file paths in the testing set
Test_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Test_noisy_audio = get_train_filepaths(path_dataset, data_type='noisy', file_type='wav')
Test_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

# loading model
model = load_model('./Models/BLSTM_'+model_type+'_SE.hdf5', custom_objects={'spec_loss': spec_loss,'prosodic_loss':prosodic_loss})

# Output WAV files (Clean & Noisy & Enhancement)
Clean_Noise_PESQ = []
Clean_Enhan_PESQ = []
File_Name = []
Clean_Noise_STOI = []
Clean_Enhan_STOI = []
for i in range(len(Test_clean_audio)):
    if VAD:
        noisy_Spec_output, Nphase_output = Sp_and_Phase(Test_noisy_audio[i], Noisy=False)
        clean_Spec, Cphase = Sp_and_Phase(Test_clean_audio[i], Noisy=False)            
        noisy_Spec, Nphase = Sp_and_Phase(Test_noisy_audio[i], Noisy=True)            
        # Obtain Pitch-based VAD index
        clean_Prosodic = Prosodic_feat_process(Test_clean_prosodic[i], Normalize=False)
        clean_Prosodic = Exten_prosodic_feat(clean_Spec, clean_Prosodic)
        clean_Prosodic = (clean_Prosodic[:,:,0]).reshape(-1) # Pitch-based VAD
        _idx_vad = Pitch_VAD(clean_Prosodic)
        start_idx = _idx_vad[0]
        end_idx = _idx_vad[1]
        # Output VAD results
        noisy_Spec_output, Nphase_output = noisy_Spec_output[:, start_idx:end_idx, :], Nphase_output[:,start_idx:end_idx]
        clean_Spec, Cphase = clean_Spec[:, start_idx:end_idx, :], Cphase[:,start_idx:end_idx]
        noisy_Spec, Nphase = noisy_Spec[:, start_idx:end_idx, :], Nphase[:,start_idx:end_idx]
        enhanced_Spec = model.predict(noisy_Spec)[0]
    else:
        noisy_Spec_output, Nphase_output = Sp_and_Phase(Test_noisy_audio[i], Noisy=False)
        clean_Spec, Cphase = Sp_and_Phase(Test_clean_audio[i], Noisy=False)
        noisy_Spec, Nphase = Sp_and_Phase(Test_noisy_audio[i], Noisy=True)
        enhanced_Spec = model.predict(noisy_Spec)[0]        

    # Reconstruct WAV file
    E_noisy = np.squeeze(noisy_Spec_output)
    E_clean = np.squeeze(clean_Spec)
    E_enhance = np.squeeze(enhanced_Spec)
    # inverse DFT
    noisy_wav = SP_to_wav(E_noisy.T, Nphase_output)
    noisy_wav = noisy_wav/np.max(abs(noisy_wav))
    clean_wav = SP_to_wav(E_clean.T, Cphase)
    clean_wav = clean_wav/np.max(abs(clean_wav))
    enhanced_wav = SP_to_wav(E_enhance.T, Nphase)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))
    
    # enhanced output WAV
    save_fname = path_output + model_type + '/' + Test_clean_audio[i].split('/')[-1]
    wavfile.write(save_fname, 16000, (enhanced_wav*(2**15)).astype('int16') )     

    # WAV file Output for PESQ & STOI calculation
    wavfile.write('clean.wav', 16000, (clean_wav*(2**15)).astype('int16') )
    wavfile.write('noisy.wav', 16000, (noisy_wav*(2**15)).astype('int16') )
    wavfile.write('enhan.wav', 16000, (enhanced_wav*(2**15)).astype('int16') )

    # Calculate PESQ
    pesq_rsl = subprocess.check_output('./PESQ +16000 clean.wav noisy.wav', shell=True)
    pesq_rsl = pesq_rsl.decode("utf-8")
    pesq_rsl = parse_pesq_rsl(pesq_rsl)[-5:]
    Clean_Noise_PESQ.append(float(pesq_rsl))
    pesq_rsl = subprocess.check_output('./PESQ +16000 clean.wav enhan.wav', shell=True)
    pesq_rsl = pesq_rsl.decode("utf-8")
    pesq_rsl = parse_pesq_rsl(pesq_rsl)[-5:]
    Clean_Enhan_PESQ.append(float(pesq_rsl))    
    File_Name.append(Test_clean_audio[i].split('/')[-1])
    
    # Calculate STOI
    fs, clean = read('./clean.wav')
    fs, noisy = read('./noisy.wav')
    fs, enhan = read('./enhan.wav') 
    # This STOI version doesn't consider VAD, so not every audio's correlation will sum to be 1 
    # (i.e., Higher non-voiced will lead lower sum results)
    stoi_base = stoi(clean, clean, fs, extended=False)
    stoi_clean_noise = stoi(clean, noisy, fs, extended=False)
    stoi_clean_enhan = stoi(clean, enhan, fs, extended=False)
    Clean_Noise_STOI.append(stoi_clean_noise/stoi_base) # Normalize for each audio for different based
    Clean_Enhan_STOI.append(stoi_clean_enhan/stoi_base) # ,thus we can compare different audio's STOI results

# Remove useless object
os.system('rm ./*.wav')
os.system('rm ./*.txt')

print('Avg. PESQ enhanced from '+str(np.mean(Clean_Noise_PESQ))+' to '+str(np.mean(Clean_Enhan_PESQ)))
print('Avg. STOI enhanced from '+str(np.mean(Clean_Noise_STOI))+' to '+str(np.mean(Clean_Enhan_STOI)))


