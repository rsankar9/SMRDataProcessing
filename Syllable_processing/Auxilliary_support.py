"""
    This scripts plots the song in an npy/txt file, in an interactive manner.
    Use line 76-77 to select portions of the file.
    Not meant to be a generic code for public use.
    Just an auxilliary file to quickly visualise the song and adjust the syllable segmenting parameters.
    Parameters should be adjusted per bird.
    Requires Python 3.7.3 and other packages.
    
    To run: python Auxilliary_support.py path_to_npy_file.npy
    or
    python Auxilliary_support.py path_to_txt_file.txt
"""


##Auxiliary code to plot amplitude of signal and smoothed amplitude
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import scipy.signal
import glob
from threading import Thread
import sys
from songbird_data_analysis import Song_functions


window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"

#threshold=2e-9 # for files recorded with Neuralynx
#threshold=2e-5
#threshold=2e-10
threshold=1.25e-9
min_syl_dur=0.02
min_silent_dur= 0.003
smooth_win=10

#rec_system = 'Alpha_omega' # or 'Neuralynx' or 'Other'
rec_system = 'Neuralynx'





##########

if rec_system is 'Alpha_omega':
    fs = 22321.4283
elif rec_system is 'Neuralynx':
    fs = 32000
print('fs:',fs)

songfile = sys.argv[1]                      # npy file
base_filename = os.path.basename(songfile)  # Extracts filename
print(songfile)
rawsong = np.array([])
if songfile[-4:] == '.npy':
    print('npy file')
    rawsong = np.load(songfile) # Loads file
elif songfile[-4:] == '.txt':
    print('txt file')
    rawsong = np.loadtxt(songfile) # Loads file
else:
    raise ValueError("Given path doesn't lead to a song file.")
    rawsong = np.loadtxt(songfile)
rawsong = rawsong.astype(float)
rawsong = rawsong.flatten()
print('size=',rawsong.size, 'shape=',rawsong.shape)

# To extract just a portion of the song
s=rawsong.size
xi = 10
rawsong = rawsong[xi*s//15:(xi+1)*s//15].reshape((xi+1)*s//15-xi*s//1,)     # Splits file according to how much data you want to view
print(len(rawsong))

amp = Song_functions.smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000))
print('amp:', amp, 'samp_freq:', fs)

(onsets, offsets) = Song_functions.segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)    # Detects syllables according to the threshold you set
shpe = len(onsets)                          # Use this to detect no. of onsets


### Building figure
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
plt.setp(ax1.get_xticklabels(), visible=True)
plt.setp(ax2.get_xticklabels(), visible=True)

# Plots spectrogram
(f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, mode='complex')
ax3.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none")

##Plot song signal amplitude
x_amp=np.arange(len(amp))
ax1.plot(x_amp*len(t)/x_amp[-1],rawsong)
ax1.set_xlim([0, len(t)])
for i in range(0,shpe):
    ax1.axvline(x=onsets[i]*len(t)/x_amp[-1],color='b')
    ax1.axvline(x=offsets[i]*len(t)/x_amp[-1],color='r')

##Plot smoothed amplitude of the song
ax2.plot(x_amp*len(t)/x_amp[-1], amp)
ax2.set_xlim([0, len(t)])
ax2.set_ylim([0, 3e-8])
for i in range(0,shpe):
    ax2.axvline(x=onsets[i]*len(t)/x_amp[-1])
    ax2.axvline(x=offsets[i]*len(t)/x_amp[-1],color='r')
ax2.axhline(y=threshold,color='g')
#ax2.xaxis.set_tick_params(which='both', labelbottom=True)

plt.show()


