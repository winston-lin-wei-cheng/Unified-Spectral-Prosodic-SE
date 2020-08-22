                                          
# This script measures intensity, pitch, MFCC for given audio files
# audio files need to be mono
# it will output features for each audio file

#directory$ = "../test_samples/clean_audio/"
directory$ = "../test_samples/noisy_audio/"

#outdir$ = "../test_samples/clean_prosodic_feat/feat_praat/"
outdir$ = "../test_samples/noisy_prosodic_feat/feat_praat/"


extension$ = ".wav"
Create Strings as file list... list 'directory$'*'extension$'
number_files = Get number of strings

for a from 1 to number_files
    select Strings list
    current_file$ = Get string... 'a'
    Read from file... 'directory$''current_file$'
    object_name$ = selected$("Sound") 
       
    To Intensity... 75 0.032 no
    Down to Matrix
    Transpose
    Write to matrix text file... 'outdir$'/'object_name$'.intensity
    Remove
    select Matrix 'object_name$'
    Remove
    select Intensity 'object_name$'
    Remove
     
    select Sound 'object_name$'
    To Pitch (ac)... 0.032 75 15 no 0.03 0.45 0.01 0.35 0.14 600
    select Pitch 'object_name$'
    Smooth... 10
    Interpolate
    To Matrix
    Transpose
    Write to matrix text file... 'outdir$'/'object_name$'.pitch
    Remove
    select Matrix 'object_name$'
    Remove
    select Pitch 'object_name$'
    Remove
    select Pitch 'object_name$'
    Remove
    select Pitch 'object_name$'
    Remove
    
    select Sound 'object_name$'
    To MFCC... 13 0.025 0.032 100 100 0
    To Matrix
    Transpose
    Write to matrix text file... 'outdir$'/'object_name$'.mfcc
    Remove
    select Matrix 'object_name$'
    Remove
    select MFCC 'object_name$'
    Remove
    select Sound 'object_name$'
    Remove

endfor
