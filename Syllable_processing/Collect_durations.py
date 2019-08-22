#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 00:43:29 2019

@author: rsankar
"""

"""
Collects data and stats on duration of each syllable across renditions and records in npy and json files.
Plots a histogram of the stats.
Verify sample frequency.
Give argument to parent folder of Labeled_songs and Annotations folders to run: python collect_durations.py path_to_parent_folder smr_file_name.smr
"""

import numpy as np
#import matplotlib.pyplot as plt
import os
import glob
import sys
import json
import matplotlib.pyplot as plt

source_path = sys.argv[1]
smr_file_name = sys.argv[2]
stats_file_destination = source_path + '/Analysis/' + smr_file_name[:-4]
if not os.path.exists(source_path + '/Analysis/'):
    os.mkdir(source_path + '/Analysis/')

Fs = 32000                                                                      # Sampling frequency
chunk_size = 120000                                                             # Chunk no. wrt to original smr file

# Collect durations and statistics

syll_data = {}                                                                  # Records data and stats
syll_set = ['a', 'b', 'c', 'd', 'e', 'f']                                       # Set of syllable labels considered
for label in syll_set:
    syll_data[label] = {}
    syll_data[label]['avg_durations'] = 0.0                                     # Average duration of a given syllable across renditions
    syll_data[label]['min_durations'] = 0.0                                     # Minimum duration of a given syllable across renditions
    syll_data[label]['max_durations'] = 0.0                                     # Maximum duration of a given syllable across renditions
    syll_data[label]['onsets'] = []
    syll_data[label]['durations'] = []                                          # Records syllable duration in each rendition

annotfiles_list = glob.glob(source_path + '/Annotations/' + '*.txt')

annotfiles_list.sort()

for annotfile in annotfiles_list:
    print(annotfile)
    annotations = np.loadtxt(annotfile, dtype='|U30')                           # Loads annotation file
    
    chunk_no = np.int(annotfile.split('_')[-2])
    padding = chunk_no * chunk_size       
        
    if annotations.shape == ():                                                 # Corner case: 1 row in annotation file i.e. 1 syllable in songfile
        annotations = [annotations.tolist()]
    
    for row in annotations:
        line = row.split(',')                                                   # Splits row to get onset, offset and label
        syll_label = line[2]
        
        if syll_label in syll_set:
            syll_onset = np.int(line[0])
            syll_offset = np.int(line[1])
            syll_duration = (syll_offset - syll_onset) * 1000 / Fs              # Converts duration in sample no. to ms
            syll_data[syll_label]['durations'].append(syll_duration)
            syll_data[syll_label]['onsets'].append(syll_onset + padding)        # Stores stats


for label in syll_set:
    if len(syll_data[label]['durations']) == 0:
        print(label, 'is missing.')
        continue
    syll_data[label]['avg_durations'] = sum(syll_data[label]['durations']) / len(syll_data[label]['durations']) # Calculates avg
    syll_data[label]['min_durations'] = min(syll_data[label]['durations'])                                      # Calculates min
    syll_data[label]['max_durations'] = max(syll_data[label]['durations'])                                      # Calculates min

# Store data and stats in npy and json files
np.save(stats_file_destination + '_duration_data.npy', syll_data)

with open(stats_file_destination + '_duration_data.json', 'w') as outfile:                           # json file to record duration data and stats
    json.dump(syll_data, outfile, indent=4)
    
syll_stats = {}
for label in syll_set:
    syll_stats[label] = {}
    syll_stats[label]['avg_durations'] = syll_data[label]['avg_durations']
    syll_stats[label]['min_durations'] = syll_data[label]['min_durations']
    syll_stats[label]['max_durations'] = syll_data[label]['max_durations']

with open(stats_file_destination + '_duration_stats.json', 'w') as outfile:                          # json file to record only duration stats
    json.dump(syll_stats, outfile, indent=4)



# Plots duration distribution for each syllable


font_s = 10  # fontsize
pos = list(range(len(syll_set)))
data = [syll_data[label]['durations'] for label in syll_set]

fig = plt.figure()
plt.violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)

x_names = [syll.upper() for syll in syll_set]
plt.xticks(pos, x_names)
plt.ylabel('Syllable Duration (ms)')
plt.xlabel('Syllables')
plt.show()

plt.savefig(stats_file_destination + '_duration_hist.png')