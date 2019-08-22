#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:27:44 2019

@author: rsankar
"""

"""
Loads pitches data previously calculated for analysis.
Give path of pitches data file as argument.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import log2, pow
import sys, os

npy_data_file = sys.argv[1]
pitch_data = np.load(npy_data_file).item()
data_file_name = os.path.basename(npy_data_file).split('_pitches')[0]
stats_file_destination = os.path.dirname(npy_data_file) + '/'

syll_set = ['a', 'b', 'c', 'd', 'e', 'f']


# Plotting histogram of pitches

fig = plt.figure()
fig.suptitle('Histogram of syllable pitches.')

font_s = 10  # fontsize
pos = list(range(len(syll_set)))
data = [pitch_data[label]['pitches'] for label in syll_set]

plt.violinplot(data, pos, points=60, widths=0.7, showmeans=True,
                      showextrema=True, showmedians=True, bw_method=0.5)

x_names = [syll.upper() for syll in syll_set]
plt.xticks(pos, x_names)
plt.ylabel('Pitches (Hz)')
plt.xlabel('Syllables')
plt.show()

plt.savefig(stats_file_destination + data_file_name + '_pitch_hist.png')


# Plot pitch at each data point

fig, ax = plt.subplots(3, 2)
fig.suptitle('Pitch vs sample.')

for i in range(len(syll_set)):
    label = syll_set[i]
    
    pitches = pitch_data[label]['pitches']
    np_pitches = np.array(pitches)
    
    x_p = np.arange(len(pitches))
    ax[i%3,i//3].set_yscale('log')
    l1 = ax[i%3,i//3].scatter(x_p, pitches)
    ax[i%3,i//3].set_ylim(top=10000)
    
ax[2,0].set_xlabel('Sample no.')
ax[2,1].set_xlabel('Sample no.')
ax[1,0].set_ylabel('Pitch (Hz)')
plt.show()

plt.savefig(stats_file_destination + data_file_name + '_pitch_vs_sample.png')


# Plotting confidence vs pitch

fig, ax = plt.subplots(3, 2)
fig.suptitle('Confidence of measurement vs syllable pitches.')

for i in range(len(syll_set)):
    label = syll_set[i]
    n_pitches = len(pitch_data[label]['pitches'])

    pitches = pitch_data[label]['pitches']
    confidence = [pitch_data[label]['stats'][j]['confidence'] for j in range(n_pitches)]
    ax[i%3,i//3].set_xscale('log')

    ax[i%3,i//3].scatter(pitches, confidence)

ax[2,0].set_xlabel('Pitch (Hz)')
ax[2,1].set_xlabel('Pitch (Hz)')
ax[1,0].set_ylabel('Confidence')

plt.savefig(stats_file_destination + data_file_name + '_confidence_vs_pitch.png')


# Plotting altered pitch vs sample no.
# Plotting (capped) multiples of each detected pitch to check if multiple of FF is being detected.

fig, ax = plt.subplots(3, 2)
fig.suptitle('Multiples of each detected pitch vs sample.')

for i in range(len(syll_set)):
    label = syll_set[i]
    
        
    pitches = pitch_data[label]['pitches']
    np_pitches = np.array(pitches)
    altered_pitches = np.array(pitches)
    altered_pitches_2 = np.array(np_pitches*2)
    altered_pitches_4 = np.array(np_pitches*4)
    altered_pitches_2 = altered_pitches_2[np.where(altered_pitches_2 < 10000)]
    altered_pitches_4 = altered_pitches_4[np.where(altered_pitches_4 < 10000)]
    
    x_p = np.arange(len(altered_pitches))
    x_p_2 = np.arange(len(altered_pitches_2))
    x_p_4 = np.arange(len(altered_pitches_4))
    ax[i%3,i//3].set_yscale('log')
    l1 = ax[i%3,i//3].scatter(x_p, altered_pitches, color='b')
    l2 = ax[i%3,i//3].scatter(x_p_2, altered_pitches_2, color='g', alpha=0.5)
    l3 = ax[i%3,i//3].scatter(x_p_4, altered_pitches_4, color='r', alpha=0.25)
    ax[i%3,i//3].set_ylim(0,10000)
    
fig.legend((l1, l2, l3), ('1x', '2x', '3x'))
ax[2,0].set_xlabel('Sample no.')
ax[2,1].set_xlabel('Sample no.')
ax[1,0].set_ylabel('Altered Pitch (Hz)')


plt.savefig(stats_file_destination + data_file_name + '_altered_pitch.png')

# Plot pitches to the closest note

A4 = 440
C0 = A4*pow(2, -4.75)
#name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def pitch(freq):
    h = 12*log2(freq/C0)
#    octave = h // 12
    n = h % 12
    return n

fig, ax = plt.subplots(3, 2)
fig.suptitle('Closest note vs sample.')

for i in range(len(syll_set)):
    label = syll_set[i]
    
    pitches = pitch_data[label]['pitches']
    np_pitches = np.array(pitches)
    
    x_p = np.arange(len(pitches))
#    ax[i%3,i//3].set_yscale('log')
#    l1 = ax[i%3,i//3].scatter(x_p, pitches)
    octaves = [pitch(p) for p in np_pitches]
    ax[i%3,i//3].scatter(x_p, octaves, marker='.')
    ax[i%3,i//3].set_ylim(0,12)
    ax[i%3,i//3].set_title(label.upper())
    
ax[2,0].set_xlabel('Sample no.')
ax[2,1].set_xlabel('Sample no.')
ax[1,0].set_ylabel('Note')

plt.savefig(stats_file_destination + data_file_name + '_note.png')