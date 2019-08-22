#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:26:43 2019

@author: rsankar
"""

"""
Loads duration data previously calculated for analysis.
Give path of duration data file as argument.
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt


npy_data_file = sys.argv[1]
duration_data = np.load(npy_data_file).item()
data_file_name = os.path.basename(npy_data_file).split('_duration')[0]
stats_file_destination = os.path.dirname(npy_data_file) + '/'

syll_set = ['a', 'b', 'c', 'd', 'e', 'f']


# Plots duration distribution for each syllable

fig = plt.figure()
fig.suptitle('Histogram of syllable durations.')

font_s = 10  # fontsize
pos = list(range(len(syll_set)))
data = [duration_data[label]['durations'] for label in syll_set]

plt.violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)

x_names = [syll.upper() for syll in syll_set]
plt.xticks(pos, x_names)
plt.ylabel('Duration (ms)')
plt.xlabel('Syllables')
plt.show()

plt.savefig(stats_file_destination + data_file_name + '_duration_hist.png')