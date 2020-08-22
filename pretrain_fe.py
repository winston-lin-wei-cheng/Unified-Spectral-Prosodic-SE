#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 12:03:35 2020

@author: winston
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.models import load_model
from utils import Sp_and_Phase, Prosodic_feat_process, Exten_prosodic_feat
from utils import get_train_filepaths, shuffle_list
from model import FE_model
from scipy.stats import spearmanr
from keras.callbacks import ModelCheckpoint


def data_generator(filelist_spec_clean, filelist_prosodic_clean):
    index=0
    while True:
        # Spec Feature
        clean_Spec, _= Sp_and_Phase(filelist_spec_clean[index], Noisy=False) 
        # Praat Feature    
        clean_Prosodic = Prosodic_feat_process(filelist_prosodic_clean[index], Normalize=False)
        clean_Prosodic = Exten_prosodic_feat(clean_Spec, clean_Prosodic)        
        index += 1
        if index == len(filelist_spec_clean):
            index = 0       
        yield clean_Spec, clean_Prosodic    
###############################################################################


###############################################################################
#  Pre-trained prosodic model to predict clean prosodic features by input     #
#  clean spec features. The trained model is used for the 'FixedConcat'       #
#  model. This is a sample code, so the training/validation/testing are       #
#  using the same set.                                                        #
###############################################################################


# Parameters        
path_dataset = './test_samples/'
epoch = 20
shuffle = True

feat_inp_spec = 257           # DFT feature dimension
feat_out_prosodic = 2         # f0 and intensity

# file paths in the training set
Train_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Train_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

if shuffle:
    permute = list(range(len(Train_clean_audio)))
    random.shuffle(permute)
    Train_clean_audio = shuffle_list(Train_clean_audio, permute)
    Train_clean_prosodic = shuffle_list(Train_clean_prosodic, permute)

# file paths in the validation set
Valid_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Valid_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

# file paths in the testing set
Test_clean_audio = get_train_filepaths(path_dataset, data_type='clean', file_type='wav')
Test_clean_prosodic = get_train_filepaths(path_dataset, data_type='clean', file_type='mat')

# loading model structure
model = FE_model(feat_inp_spec, feat_out_prosodic)

# model loss and optimizer
model.compile(loss='mse', optimizer='rmsprop')

# model saving settings
checkpointer = ModelCheckpoint(filepath='BLSTM_pretrained_FE.hdf5', verbose=1, save_best_only=True, mode='min') 
 
# model training/validation
gen_train = data_generator(Train_clean_audio, Train_clean_prosodic)
gen_valid = data_generator(Valid_clean_audio, Valid_clean_prosodic)
hist = model.fit_generator(gen_train,
                           steps_per_epoch=len(Train_clean_audio),
                           epochs=epoch, 
                           verbose=1,
                           validation_data=gen_valid,
                           validation_steps=len(Valid_clean_audio),
                           max_queue_size=1, 
                           workers=1,
                           callbacks=[checkpointer])

# prosodic reconstruction performance testing
model = None
best_model = load_model('BLSTM_pretrained_FE.hdf5')

Recon_Spear_Intensity = []
Recon_Spear_Pitch = []
for i in range(len(Test_clean_audio)):  
    # Spec Feature
    clean_Spec, _= Sp_and_Phase(Test_clean_audio[i], Noisy=False) 
    # Praat Feature    
    clean_Prosodic = Prosodic_feat_process(Test_clean_prosodic[i] , Normalize=False)
    clean_Prosodic = Exten_prosodic_feat(clean_Spec, clean_Prosodic)   
    # model prediction    
    pred_Prosodic = best_model.predict(clean_Spec)
    # Spearman corr. evaluation metric
    p_spear_corr = spearmanr(clean_Prosodic[:,:,0].reshape(-1), pred_Prosodic[:,:,0].reshape(-1))
    Recon_Spear_Pitch.append(p_spear_corr)   
    e_spear_corr = spearmanr(clean_Prosodic[:,:,1].reshape(-1), pred_Prosodic[:,:,1].reshape(-1))
    Recon_Spear_Intensity.append(e_spear_corr) 
    
    # Plot Reconstruction Results
    plt.rc('font',family='Times New Roman')
    plt.plot( clean_Prosodic[:,:,0].reshape(-1), color='blue', linewidth=3.5)
    plt.plot( pred_Prosodic[:,:,0].reshape(-1), color='red', linewidth=3.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

print( 'Avg. Reconstruct Intensity SpearCorr: ' + str(np.mean(Recon_Spear_Intensity,axis=0)[0]) )  
print( 'Avg. Reconstruct Pitch SpearCorr: ' + str(np.mean(Recon_Spear_Pitch,axis=0)[0]) )

