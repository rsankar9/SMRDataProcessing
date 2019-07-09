#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:43:29 2019

@author: rsankar
"""

"""
Collects data and stats on duration of each syllable across renditions and records in a json file.
Verify sample frequency.
Give argument to parent folder of Clean_songs and Annotations folders to run: python collect_durations.py path_to_parent_folder
"""

import numpy as np
#import matplotlib.pyplot as plt
import pitch
import glob
import sys
import os
import json

source_path = sys.argv[1]
#source_path = "Testing_Pitch/"
Fs = 32000                                      # Sampling frequency

syll_data = {}                                  # Records data and stats
syll_set = ['a', 'b', 'c', 'd', 'e', 'f']       # Set of syllable labels considered
for label in syll_set:
    syll_data[label] = {}
    syll_data[label]['avg_durations'] = 0.0     # Average duration of a given syllable across renditions
    syll_data[label]['min_durations'] = 0.0     # Minimum duration of a given syllable across renditions
    syll_data[label]['durations'] = []          # Records syllable duration in each rendition

annotfiles_list = glob.glob(source_path + '/Annotations/*.txt')

for annotfile in annotfiles_list:
    print(annotfile)
    annotations = np.loadtxt(annotfile, dtype='|U30')   # Loads annotation file
        
    if annotations.shape == ():                         # Corner case: 1 row in annotation file i.e. 1 syllable in songfile
        annotations = [annotations.tolist()]
    
    for row in annotations:
        line = row.split(',')                               # Splits row to get onset, offset and label
        syll_label = line[2]
        
        if syll_label in syll_set:
            syll_onset = np.int(line[0])
            syll_offset = np.int(line[1])
            syll_duration = (syll_offset - syll_onset) * 1000 / Fs      # Converts duration in sample no. to ms
            syll_data[syll_label]['durations'].append(syll_duration)

for label in syll_set:
    syll_data[label]['avg_durations'] = sum(syll_data[label]['durations']) / len(syll_data[label]['durations']) # Calculates avg
    syll_data[label]['min_durations'] = min(syll_data[label]['durations'])                                      # Calculates min

with open(source_path + 'syll_duration_data.json', 'w') as outfile:                           # json file to record duration data and stats
    json.dump(syll_data, outfile, indent=4)
    
    
syll_stats = {}
for label in syll_set:
    syll_stats[label] = {}
    syll_stats[label]['avg_durations'] = syll_data[label]['avg_durations']
    syll_stats[label]['min_durations'] = syll_data[label]['min_durations']

with open(source_path + 'syll_duration_stats.json', 'w') as outfile:                          # json file to record only duration stats
    json.dump(syll_stats, outfile, indent=4)
    
  
                             
#                             
#
