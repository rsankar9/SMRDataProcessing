#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 12:11:28 2019

@author: rsankar
"""

"""
Loads spike data previously calculated.
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt


npy_data_file = sys.argv[1]
data_file_name = os.path.basename(npy_data_file).split('_spike')[0]
spikes_data = np.load(npy_data_file).item()
stats_file_destination = os.path.dirname(npy_data_file) + '/'


syll_set = ['a', 'b', 'c', 'd', 'e', 'f']


# Plots spike rate distribution for each syllable

fig = plt.figure()
fig.suptitle('Histogram of spike rates.')
font_s = 10  # fontsize
pos = list(range(len(syll_set)))
data = [spikes_data[label]['spike_rate'] for label in syll_set]

plt.violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)

x_names = [syll.upper() for syll in syll_set]
plt.xticks(pos, x_names)
plt.ylabel('Spike rate (Hz)')
plt.xlabel('Syllables')
#plt.show()

plt.savefig(stats_file_destination + data_file_name + '_spike_rate_hist.png')