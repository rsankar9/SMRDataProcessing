#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:10:15 2019

@author: rsankar
"""

"""
Loads spike data, duration data and pitch data.
Give npy files as argument.
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

pitch_data_file = sys.argv[1]
duration_data_file = sys.argv[2]
spikes_data_file = sys.argv[3]
pitch_data = np.load(pitch_data_file).item()
duration_data = np.load(duration_data_file).item()
spike_data = np.load(spikes_data_file).item()
stats_file_destination = os.path.dirname(pitch_data_file) + '/'
data_file_name = os.path.basename(npy_data_file).split('_spike')[0]

syll_set = ['a', 'b', 'c', 'd', 'e', 'f']

# Plots spike rate vs pitch

fig, ax = plt.subplots(3,2)
fig.suptitle('Spike rate vs pitch.')

for i in range(len(syll_set)):
    label = syll_set[i]
    
    ax[i%3,i//3].scatter(pitch_data[label]['pitches'], spike_data[label]['spike_rate'])
    ax[i%3,i//3].set_title(label)

ax[2,0].set_xlabel('Pitch (Hz)')
ax[2,1].set_xlabel('Pitch (Hz)')
ax[1,0].set_ylabel('Spike rate (Hz)')

plt.show()

plt.savefig(stats_file_destination + data_file_name + '_spike_vs_pitch.png')

# Plots spike rate vs duration

fig, ax = plt.subplots(3,2)
fig.suptitle('Spike rate vs duration.')

for i in range(len(syll_set)):
    label = syll_set[i]
    
    ax[i%3,i//3].scatter(duration_data[label]['durations'], spike_data[label]['spike_rate'])
    ax[i%3,i//3].set_title(label)

ax[2,0].set_xlabel('Duration (ms)')
ax[2,1].set_xlabel('Duration (ms)')
ax[1,0].set_ylabel('Spike rate (Hz)')

plt.show()

plt.savefig(stats_file_destination + data_file_name + '_spike_vs_duration.png')

# Plotting spike rate vs pitch where confidence level of pitch is above 0.2

fig, ax = plt.subplots(3, 2)
fig.suptitle('Spike rate vs pitch with confidence level > 0.2.')

for i in range(len(syll_set)):
    label = syll_set[i]
    n_pitches = len(pitch_data[label]['pitches'])
    confidence = [pitch_data[label]['stats'][j]['confidence'] for j in range(n_pitches)]
    pitches = [pitch_data[label]['pitches'][j] for j in range(n_pitches) if confidence[j]>0.2]
    spike_rates = [spike_data[label]['spike_rate'][j] for j in range(n_pitches) if confidence[j]>0.2]

    ax[i%3,i//3].set_xscale('log')
    ax[i%3,i//3].scatter(pitches, spike_rates)

ax[2,0].set_xlabel('Pitch (Hz)')
ax[2,1].set_xlabel('Pitch (Hz)')
ax[1,0].set_ylabel('Spike rate (Hz)')
plt.show()

plt.savefig(stats_file_destination + data_file_name + '_spike_vs_pitch_c2.png')

# Plotting spike rate vs pitch where confidence level of pitch is above 0.3

fig, ax = plt.subplots(3, 2)
fig.suptitle('Spike rate vs pitch with confidence level > 0.3.')

for i in range(len(syll_set)):
    label = syll_set[i]
    n_pitches = len(pitch_data[label]['pitches'])
    confidence = [pitch_data[label]['stats'][j]['confidence'] for j in range(n_pitches)]
    pitches = [pitch_data[label]['pitches'][j] for j in range(n_pitches) if confidence[j]>0.3]
    spike_rates = [spike_data[label]['spike_rate'][j] for j in range(n_pitches) if confidence[j]>0.3]

    ax[i%3,i//3].set_xscale('log')
    ax[i%3,i//3].scatter(pitches, spike_rates)
    
ax[2,0].set_xlabel('Pitch (Hz)')
ax[2,1].set_xlabel('Pitch (Hz)')
ax[1,0].set_ylabel('Spike rate (Hz)')
plt.show()

plt.savefig(stats_file_destination + data_file_name + '_spike_vs_pitch_c3.png')


