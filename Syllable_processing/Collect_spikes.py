#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:34:36 2019

@author: rsankar
"""

"""
Extracting spikes data.
"""

##################################################

import sys
import neo
import numpy as np
import pylab as py
import glob
import os
import json
import matplotlib.pyplot as plt
from songbird_data_analysis import functions


##################################################


fs=32000                                                                        # Sampling Frequency (Hz)
chunk_size = 120000                                                             # Chunk size of each file chunk
premotor_win = 0.05

source_path = sys.argv[1]
smr_file = sys.argv[2]
smr_file_name = os.path.basename(smr_file)
stats_file_destination = source_path + '/Analysis/'

n_steps = 4
step = np.array([False, False, True, False])


##################################################


def rec_spiketrain(spike_train_index):
    """
    Collects a particular spike train and saves it.
    """
    
    print('Collecting signals.')
    analog, sp = functions.getarrays(smr_file)
    
    print('Saving chosen spike train to a file.')
    np.save(stats_file_destination + smr_file_name[:-4] + '_spike_train_' + str(spike_train_index) + '.npy', sp[spike_train_index])
  

def sliding_window_spike_train(spike_train, win_size=0.1):
    """
    Plots spike rate over time of given spike train.
    Default : win_size = 0.1s
    """
    sd = (spike_train * fs).astype(int)
    m_sd = np.max(sd)

    n_samples = m_sd+1
    spikes = np.zeros(n_samples)
    spikes[sd] = 1  
    
    window = np.ones(int(win_size*fs))
    sliding_spike_rate = np.convolve(spikes,window)/win_size
    plt.plot(sliding_spike_rate/max(sliding_spike_rate))
    plt.ylabel('Spike rate')
    plt.xlabel('Sample')
    plt.show()
    plt.savefig(stats_file_destination + smr_file_name[:-4] + '_spike_rate.png')
    

def calc_spike_rate(chunk_number, syll_onset, spike_train):
    """
    Calculating spike rate in a window given an onset.
    premotor_win -> seconds
    syll_onset -> sample no.
    """
    syll_onset += chunk_number * chunk_size
    # syll_onset_time = syll_onset * 1/fs                                       #s
    
    sd = (spike_train * fs).astype(int)
    m_sd = np.max(sd)

    n_samples = m_sd+1
    spike_train = spike_train.astype(int)
    spikes = np.zeros(n_samples)
    spikes[sd] = 1  
    spikes_segment = spikes[syll_onset-int(premotor_win*fs):syll_onset] 
#    time_segment = time[syll_onset-int(premotor_win*fs):syll_onset] 
    spike_rate = np.sum(spikes_segment)/premotor_win                               # Hz
    
    return spike_rate
    
    
def record_data(spike_train):
    """
    Collect stats on a given spike train.
    """
    
    annotfiles_list = glob.glob(source_path + 'Annotations/' + smr_file_name[:-4] + '*_annot.txt')
    syll_set = ['a', 'b', 'c', 'd', 'e', 'f']       # Set of syllable labels considered
    annotfiles_list.sort()
    spike_data = {}
    for label in syll_set:
        spike_data[label] = {}
        spike_data[label]['avg_spike_rate'] = 0.0     
        spike_data[label]['min_spike_rate'] = 0.0     
        spike_data[label]['max_spike_rate'] = 0.0
        spike_data[label]['onset'] = []
        spike_data[label]['spike_rate'] = []          
    
    for annotfile in annotfiles_list:
        print(annotfile)
        chunk_number = int(annotfile.split('_')[-2])
        padding = chunk_number * chunk_size
        annotations = np.loadtxt(annotfile, dtype='|U30')   # Loads annotation file
        if annotations.shape == ():                         # Corner case: 1 row in annotation file i.e. 1 syllable in songfile
            annotations = [annotations.tolist()]
    
        for row in annotations:
            line = row.split(',')                               # Splits row to get onset, offset and label
            syll_label = line[2]
            if syll_label in syll_set:
                syll_onset = np.int(line[0])
                spike_rate = calc_spike_rate(chunk_number, syll_onset, spike_train)
                spike_data[syll_label]['onset'].append(syll_onset + padding)
                spike_data[syll_label]['spike_rate'].append(spike_rate)
     
    for label in syll_set:
        spike_data[label]['avg_spike_rate'] = sum(spike_data[label]['spike_rate']) / len(spike_data[label]['spike_rate']) # Calculates avg
        spike_data[label]['min_spike_rate'] = min(spike_data[label]['spike_rate'])                                      # Calculates min
        spike_data[label]['max_spike_rate'] = max(spike_data[label]['spike_rate'])                                      # Calculates max
        
        
    np.save(stats_file_destination + smr_file_name[:-4] + '_spike_rate_data', spike_data)
    
    spike_stats = {}
    for label in syll_set:
        spike_stats[label] = {}
        spike_stats[label]['avg_spike_rate'] = spike_data[label]['avg_spike_rate']
        spike_stats[label]['min_spike_rate'] = spike_data[label]['min_spike_rate']
        spike_stats[label]['max_spike_rate'] = spike_data[label]['max_spike_rate']
    
    with open(stats_file_destination + smr_file_name[:-4] + '_spike_stats.json', 'w') as outfile:                          # json file to record only duration stats
        json.dump(spike_stats, outfile, indent=4)
    
 

##################################################




# Comment out whichever steps you aren't interested in

# ----- STEP 0: Choose which channel you want to analyse ----- #

if step[0] == True:
    functions.plotplots(smr_file)

spike_train_index = int(input('Enter index of spike train to analyse:'))        # Comment out if setting directly in next line.
# spike_train_index = 10                                                        # Set after step #0; Selecting channel 18 #2

# ----- STEP 1: Save chosen spike train ----- #

if step[1] == True:
    rec_spiketrain(spike_train_index)

spike_train_file = stats_file_destination + smr_file_name[:-4] + '_spike_train_' + str(spike_train_index) + '.npy'

spike_train = np.load(spike_train_file)
spike_train = np.array(spike_train)

# ----- STEP 2: Plot spike rate of chosen spike train over time ----- #

if step[2] == True:
    print('Plotting spike rate over time')
    sliding_window_spike_train(spike_train)
    
# ----- STEP 3: Save chosen spike train ----- #

if step[3] == True:
    print('Collecting statistics on given spike train.')
    record_data(spike_train)
    

    