#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:49:10 2019

@author: rsankar
"""
"""
This script detects pitches of the syllables present in annotated files.
Data and confidence measures are stored in a json file.
Verify parameters: sample frequency, syllable window, chunk_size, varying_window_dur, etc.
Give argument to parent folder of Clean_songs and Annotations folders to run: python collect_pitch.py path_to_parent_folder
"""

import numpy as np
#import matplotlib.pyplot as plt
import glob
import sys
import os
import json

source_path = sys.argv[1]
#source_path = "Testing_Pitch/"

syll_set = ['a', 'b', 'c', 'd', 'e', 'f']                                       # Set of syllables considered
chunk_size = 120000                                                             # Chunk no. wrt to original smr file
Fs = 32000                                                                      # Sampling rate
syll_window = 20 #ms                                                            # Window used for syllable duration from onset
varying_window_dur = False                                                      # True: Window duration changes acc to syllable duration False: fixed at syll_window

# Function to compute pitch
def compute_pitch(f, onset, Fs, window=20):
    """
        Computes pitch from a given timepoint (onset) in the song (f) for a given duration (window (ms)) at a given sampling frequency (Fs).
        Returns the computed pitch and stats on confidence measure.
    """

    n_window_samples = int(Fs/1000 * window)                                    # Converting window duration, given in ms, to no. of samples
    f_chunk = f[onset:onset+n_window_samples]                                   # Picking the required file chunk for this syllable
    len_fc = len(f_chunk)                                   
    q = np.correlate(f_chunk,f_chunk,'same')                                    # Auto correlation -> q
    x = 1000/Fs * np.arange(-len_fc//2,len_fc//2) #ms                           # Converting x axis from sample no. to time in ms

    peaks = []                                                                  # Collecting peaks in auto correlation
    for i in np.arange(len(q)//2,len(q)-1):                                     # Only checks second half (assuming symmetry)
        if np.sign(q[i]-q[i-1]) == np.sign(q[i]-q[i+1]):                        # To detect peak
            peaks.append(i)
    q2 = np.take(q, peaks)                                                      # Selects q values at peaks
    q2_indices = np.argsort(q2)                                                 # Returns q2 indices for sorted peaks
    q_values = np.take(q2, q2_indices)                                          # Returns q values for sorted peaks
    q_indices = np.take(peaks, q2_indices)                                      # Returns q indices for sorted peak

    q_stats = {}                                                                # Storing statistics of 2nd and 3rd highest peak
    q_stats['2ndPeak'] = [(q_values[-2]), int(q_indices[-2])]                   
    q_stats['3rdPeak'] = [(q_values[-3]), int(q_indices[-3])]
    q_stats['confidence'] = (q_values[-2] - q_values[-3])/q_values[-2]          # Stores confidence measure

    pitch = 1000/x[q_indices[-2]]                                               # Calculates pitch (Hz) as per 2nd highest peak (1st peak is always 0)
    return pitch, q_stats



syll_data = {}                                                                  # Records syllable data and stats

for label in syll_set:                                                          # Initialises dictionary to collect data
    syll_data[label] = {}
    syll_data[label]['onsets'] = []
    syll_data[label]['pitches'] = []
    syll_data[label]['stats'] = []


songfiles_list = glob.glob(source_path + '/Clean_songs_3/*.txt')                  # Gathers song files

for songfile in songfiles_list:
    print(songfile)
    base_filename = os.path.basename(songfile)[:-4]
    song = np.loadtxt(songfile)                                                 # Loads songfile
    annot_file = source_path + '/Annotations_3/' + base_filename + '_annot.txt'
    annotations = np.loadtxt(annot_file, dtype='|U30')                          # Loads corresponding annotation files
    
    chunk_no = np.int(base_filename.split('_')[-1])
    padding = chunk_no * chunk_size                                             # To calculate onset times wrt to original smr file
    
    if annotations.shape == ():                                                 # Songfile with 1 syllable only
        annotations = [annotations.tolist()]
    
    for row in annotations:                                                     # For every row in annotation file
        line = row.split(',')                                                   # Splits row to get syllable onset, offset and label
        syll_label = line[2]
        
        if syll_label in syll_set:                                              
            syll_onset = np.int(line[0])
            if varying_window_dur == True:
                syll_offset = np.int(line[1])                                           # To calculate syll window varying acc to syll duration
                syll_duration = (syll_offset-syll_onset) / Fs * 1000 #ms
                if syll_duration < 20: syll_window = 20
                elif syll_duration < 50: syll_window = int(syll_duration)
                else: syll_window = 50
            syll_pitch, syll_stats = compute_pitch(song, syll_onset, Fs, syll_window)   # Computes pitch
            syll_data[syll_label]['onsets'].append(syll_onset + padding)                # Stores stats
            syll_data[syll_label]['pitches'].append(syll_pitch)
            syll_data[syll_label]['stats'].append(syll_stats)

if varying_window_dur == True:
    with open(source_path + 'syll_pitches_data_varying_w.json', 'w') as outfile:                                # Writes data to json file
        json.dump(syll_data, outfile, indent=4)

else:
    with open(source_path + 'syll_pitches_data_fixed_w.json', 'w') as outfile:                                  # Writes data to json file
        json.dump(syll_data, outfile, indent=4)
