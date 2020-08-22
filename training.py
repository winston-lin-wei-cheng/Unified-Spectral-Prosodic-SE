#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:12:18 2020

@author: winston
"""
import os
import random
from keras.models import load_model
from keras import objectives
from utils import Sp_and_Phase, Prosodic_feat_process, Exten_prosodic_feat, Pitch_VAD
from utils import get_train_filepaths, shuffle_list
from model import FixedConcat_SE, JointConcat_SE, MultiTask_SE
from keras.callbacks import ModelCheckpoint


def data_generator(filelist_spec_clean, filelist_spec_noisy, filelist_prosodic_clean, VAD=True):
    index=0
    while True:
        # Spec Feature
        noisy_Spec, _= Sp_and_Phase(filelist_spec_noisy[index], Noisy=True) 
        clean_Spec, _= Sp_and_Phase(filelist_spec_clean[index], Noisy=False) 
        # Praat Feature    
        clean_Prosodic = Prosodic_feat_process(filelist_prosodic_clean[index], Normalize=False)
        clean_Prosodic = Exten_prosodic_feat(clean_Spec, clean_Prosodic) 
        # VAD Feature
        if VAD:
            clean_Pitch = (clean_Prosodic[:,:,0]).reshape(-1) # Pitch-based VAD 
            _idx_vad = Pitch_VAD(clean_Pitch)
            start_idx = _idx_vad[0]
            end_idx = _idx_vad[1]
            # VAD Data Output
            noisy_Spec = noisy_Spec[:, start_idx:end_idx, :]
            clean_Spec = clean_Spec[:, start_idx:end_idx, :]
            clean_Prosodic = clean_Prosodic[:, start_idx:end_idx, :]          
        index += 1
        if index == len(filelist_spec_clean):
            index = 0       
        yield noisy_Spec, [clean_Spec, clean_Prosodic]    

def spec_loss(input_img, output_img):
    return 10*objectives.mse(input_img,output_img)

def prosodic_loss(input_img, output_img):
    return objectives.mae(input_img,output_img)/10
###############################################################################



###############################################################################
#  This is a sample code for the proposed prosodic-spec SE model training,    #
#  so the validation and training set we use here are the same set.           #
###############################################################################

# Parameters        
path_dataset = './test_samples/'
model_type = 'MultiTask'      # 'JointConcat', 'FixedConcat' or 'MultiTask'
epoch = 20
shuffle = True

feat_inp_spec = 257           # DFT feature dimension
feat_out_spec = 257           # DFT feature dimension
feat_out_prosodic = 2         # f0 and intensity

# create Models folder
if not os.path.exists('Models'):
    os.makedirs('Models')

# file paths in the training set
Train_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Train_noisy_audio = get_train_filepaths(path_dataset, data_type='noisy', file_type='wav')
Train_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

if shuffle:
    permute = list(range(len(Train_clean_audio)))
    random.shuffle(permute)
    Train_clean_audio = shuffle_list(Train_clean_audio, permute)
    Train_noisy_audio = shuffle_list(Train_noisy_audio, permute)
    Train_clean_prosodic = shuffle_list(Train_clean_prosodic, permute)

# file paths in the validation set
Valid_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Valid_noisy_audio = get_train_filepaths(path_dataset, data_type='noisy', file_type='wav')
Valid_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

# loading model structure
if model_type=='JointConcat':
    model = JointConcat_SE(feat_inp_spec, feat_out_spec, feat_out_prosodic)
elif model_type=='MultiTask':  
    model = MultiTask_SE(feat_inp_spec, feat_out_spec, feat_out_prosodic)
elif model_type=='FixedConcat': 
    trained_prosodic_model = load_model('BLSTM_pretrained_FE.hdf5')
    model = FixedConcat_SE(feat_inp_spec, feat_out_spec, trained_prosodic_model)

# model loss and optimizer
model.compile(loss=[spec_loss, prosodic_loss], optimizer='rmsprop')

# model saving settings
checkpointer = ModelCheckpoint(filepath='./Models/BLSTM_'+model_type+'_SE.hdf5', verbose=1, save_best_only=True, mode='min') 
 
# model training/validation
gen_train = data_generator(Train_clean_audio, Train_noisy_audio, Train_clean_prosodic, VAD=True)
gen_valid = data_generator(Valid_clean_audio, Valid_noisy_audio, Valid_clean_prosodic, VAD=True)
hist = model.fit_generator(gen_train,
                           steps_per_epoch=len(Train_clean_audio),
                           epochs=epoch, 
                           verbose=1,
                           validation_data=gen_valid,
                           validation_steps=len(Valid_clean_audio),
                           max_queue_size=1, 
                           workers=1,
                           callbacks=[checkpointer])
