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
from utils import Sp_and_Phase, SP_to_wav
from scipy.io import wavfile


def spec_loss(input_img, output_img):
    return 10*objectives.mse(input_img,output_img)

def praat_loss(input_img, output_img):
    return objectives.mae(input_img,output_img)/10  
###############################################################################




###############################################################################
#        Prediction Process based on the trained mdoel in the paper           #
###############################################################################

# Parameters        
dir_input = './test_input_folder/'       # input noisy wav files folder
dir_output = './enhanced_output_folder/' # enhanced output folder
model_type = 'FixedConcat'                # 'JointConcat', 'FixedConcat' or 'MultiTask'

# create output folder
if not os.path.exists(dir_output):
    os.makedirs(dir_output)

# file paths in the input folder
input_wav_paths = []
for root, directories, files in os.walk(dir_input):
    for fname in files:
        filepath = os.path.join(root, fname)
        input_wav_paths.append(filepath)
input_wav_paths = sorted(input_wav_paths)   

# loading model
model = load_model('./trained_models_TIMIT/BLSTM_'+model_type+'_VadSE.hdf5', custom_objects={'spec_loss': spec_loss,'praat_loss':praat_loss})

# Prediction process & output enhanced WAV files
for i in range(len(input_wav_paths)):
    # data processing & model flow
    noisy_Spec, Nphase = Sp_and_Phase(input_wav_paths[i], Noisy=True)
    enhanced_Spec = model.predict(noisy_Spec)[0]  
    
    # Reconstruct WAV file
    E_enhance = np.squeeze(enhanced_Spec)
    
    # inverse DFT
    enhanced_wav = SP_to_wav(E_enhance.T, Nphase)
    enhanced_wav = enhanced_wav/np.max(abs(enhanced_wav))

    # enhanced output WAV
    save_fname = dir_output + model_type + '/' + input_wav_paths[i].split('/')[-1]
    wavfile.write(save_fname, 16000, (enhanced_wav*(2**15)).astype('int16') )     

