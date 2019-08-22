#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Created on Mon Jul  1 15:41:48 2019
    
    @author: rsankar

    Get files from Raw_songs folder.
    Detects if it has syllables.
    If not, the file is moved to Noise_songs folder.
    Ensure the segmenting parameters are correct.
    To run: python Detect_syllables.py path_to_parent_folder
"""

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread
import sys
import matplotlib.widgets as mplw
from songbird_data_analysis import Song_functions


window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"

#Threshold for segmentation
#threshold=1.6e-4 # for files recorded with RTXI from recording box (above the chamber)
#threshold=2e-9 # for files recorded with Neuralynx
#threshold=2e-5
threshold=1.25e-9
min_syl_dur=0.02
min_silent_dur= 0.003
#min_silent_dur= 0.005
smooth_win=10

#Contains the labels of the syllables from a single .wav file
labels = []
syl_counter=0
Nb_syls=0
keep_song =''
#rec_system = 'Alpha_omega' # or 'Neuralynx' or 'Other'
rec_system = 'Neuralynx'





#################################

folder_name = sys.argv[1]
if os.path.isdir(folder_name) is False:
    raise ValueError("Not a folder.")

print(folder_name)
source_path = folder_name + '/Raw_songs'
target_path_noise = folder_name + '/Noise_songs'
target_path_clean = folder_name + '/Clean_songs'
if not os.path.exists(source_path):
    raise ValueError('Raw_songs folder does not exist.')
if not os.path.exists(target_path_noise):
    os.mkdir(target_path_noise)
    print('Created folder Noise.')
if not os.path.exists(target_path_clean):
    os.mkdir(target_path_clean)
    print('Created folder Clean.')

if rec_system == 'Alpha_omega':
    fs = 22321.4283
elif rec_system == 'Neuralynx':
    fs = 32000
print('fs:',fs)
songfiles_list = glob.glob(source_path + '/*.txt')
lsl = len(songfiles_list)

for file_num, songfile in enumerate(songfiles_list):
    print('File no:', file_num, 'from ', lsl)
    base_filename = os.path.basename(songfile)
    
    #Read song file
    print('File name: %s' % songfile)
    rawsong = np.loadtxt(songfile)
    
    #Bandpass filter, square and lowpass filter
    #cutoffs : 1000, 8000
    try:
        amp = Song_functions.smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000))
    except ValueError:
        print('Moving file to noise because length of rawsong is not greater than padlen.')
        file_path_target_noise = target_path_noise+'/'+base_filename
        os.rename(songfile, file_path_target_noise)
        continue
    
    #Segment song
    (onsets, offsets) = Song_functions.segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
    shpe = len(onsets)
    if shpe < 1:
        file_path_target_noise = target_path_noise+'/'+base_filename
        os.rename(songfile, file_path_target_noise)
    else:
        file_path_target_clean = target_path_clean+'/'+base_filename
        os.rename(songfile, file_path_target_clean)
