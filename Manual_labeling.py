#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:41:48 2019

@author: rsankar
"""

###################################################################################
# README.md
###################################################################################
# - Give the path to the parent folder of Raw_songs as an argument when you run the python script: python Manual_labeling.py parent_folder_name
# - Ensure this folder has a folder called Raw_songs with the songfiles (.txt) to be labelled.
# - Ensure this folder doesn't have a clashing Annotations, Clean_songs or Noise_songs folder (as some files will be moved/created in these folders).
# - To change the threshold, rec_system, or other parameters, you'll have to directly change the code.

###################################################################################

import numpy as np
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread
import sys
import matplotlib.widgets as mplw

###################################################################################
# Block 0: Define variables and functions
###################################################################################
#Spectrogram parameters
#Default windowing function in spectrogram function
#window =('tukey', 0.25) 
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

# Bellow, definition of functions used for filtering and segmentation

# Initial default for smooth_win = 2
def smooth_data(rawsong, samp_freq, freq_cutoffs=None, smooth_win=10):

    if freq_cutoffs is None:
        # then don't do bandpass_filtfilt
        filtsong = rawsong
    else:
        filtsong = bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs)

    squared_song = np.power(filtsong, 2)

    len = np.round(samp_freq * smooth_win / 1000).astype(int)
    h = np.ones((len,)) / len
    smooth = np.convolve(squared_song, h)
    offset = round((smooth.shape[-1] - filtsong.shape[-1]) / 2)
    smooth = smooth[offset:filtsong.shape[-1] + offset]
    return smooth

	
def bandpass_filtfilt(rawsong, samp_freq, freq_cutoffs=(500, 10000)):
    """filter song audio with band pass filter, run through filtfilt
    (zero-phase filter)

    Parameters
    ----------
    rawsong : ndarray
        audio
    samp_freq : int
        sampling frequency
    freq_cutoffs : list
        2 elements long, cutoff frequencies for bandpass filter.
        If None, no cutoffs; filtering is done with cutoffs set
        to range from 0 to the Nyquist rate.
        Default is [500, 10000].

    Returns
    -------
    filtsong : ndarray
    """
    if freq_cutoffs[0] <= 0:
        raise ValueError('Low frequency cutoff {} is invalid, '
                         'must be greater than zero.'
                         .format(freq_cutoffs[0]))

    Nyquist_rate = samp_freq / 2
    if freq_cutoffs[1] >= Nyquist_rate:
        raise ValueError('High frequency cutoff {} is invalid, '
                         'must be less than Nyquist rate, {}.'
                         .format(freq_cutoffs[1], Nyquist_rate))

    if rawsong.shape[-1] < 387:
        numtaps = 64
    elif rawsong.shape[-1] < 771:
        numtaps = 128
    elif rawsong.shape[-1] < 1539:
        numtaps = 256
    else:
        numtaps = 512

    cutoffs = np.asarray([freq_cutoffs[0] / Nyquist_rate,
                          freq_cutoffs[1] / Nyquist_rate])
    # code on which this is based, bandpass_filtfilt.m, says it uses Hann(ing)
    # window to design filter, but default for matlab's fir1
    # is actually Hamming
    # note that first parameter for scipy.signal.firwin is filter *length*
    # whereas argument to matlab's fir1 is filter *order*
    # for linear FIR, filter length is filter order + 1
    b = scipy.signal.firwin(numtaps + 1, cutoffs, pass_zero=False)
    a = np.zeros((numtaps+1,))
    a[0] = 1  # make an "all-zero filter"
    padlen = np.max((b.shape[-1] - 1, a.shape[-1] - 1))
    filtsong = scipy.signal.filtfilt(b, a, rawsong, padlen=padlen)
    #filtsong = filter_song(b, a, rawsong)
    return (filtsong)
	
	
def segment_song(amp,
                 segment_params={'threshold': 5000, 'min_syl_dur': 0.2, 'min_silent_dur': 0.02},
                 time_bins=None,
                 samp_freq=None):
    """Divides songs into segments based on threshold crossings of amplitude.
    Returns onsets and offsets of segments, corresponding (hopefully) to syllables in a song.
    Parameters
    ----------
    amp : 1-d numpy array
        Either amplitude of power spectral density, returned by compute_amp,
        or smoothed amplitude of filtered audio, returned by evfuncs.smooth_data
    segment_params : dict
        with the following keys
            threshold : int
                value above which amplitude is considered part of a segment. default is 5000.
            min_syl_dur : float
                minimum duration of a segment. default is 0.02, i.e. 20 ms.
            min_silent_dur : float
                minimum duration of silent gap between segment. default is 0.002, i.e. 2 ms.
    time_bins : 1-d numpy array
        time in s, must be same length as log amp. Returned by Spectrogram.make.
    samp_freq : int
        sampling frequency

    Returns
    -------
    onsets : 1-d numpy array
    offsets : 1-d numpy array
        arrays of onsets and offsets of segments.

    So for syllable 1 of a song, its onset is onsets[0] and its offset is offsets[0].
    To get that segment of the spectrogram, you'd take spect[:,onsets[0]:offsets[0]]
    """

    if time_bins is None and samp_freq is None:
        raise ValueError('Values needed for either time_bins or samp_freq parameters '
                         'needed to segment song.')
    if time_bins is not None and samp_freq is not None:
        raise ValueError('Can only use one of time_bins or samp_freq to segment song, '
                         'but values were passed for both parameters')

    if time_bins is not None:
        if amp.shape[-1] != time_bins.shape[-1]:
            raise ValueError('if using time_bins, '
                             'amp and time_bins must have same length')

    above_th = amp > segment_params['threshold']
    h = [1, -1]
    # convolving with h causes:
    # +1 whenever above_th changes from 0 to 1
    # and -1 whenever above_th changes from 1 to 0
    above_th_convoluted = np.convolve(h, above_th)

    if time_bins is not None:
        # if amp was taken from time_bins using compute_amp
        # note that np.where calls np.nonzero which returns a tuple
        # but numpy "knows" to use this tuple to index into time_bins
        onsets = time_bins[np.where(above_th_convoluted > 0)]
        offsets = time_bins[np.where(above_th_convoluted < 0)]
    elif samp_freq is not None:
        # if amp was taken from smoothed audio using smooth_data
        # here, need to get the array out of the tuple returned by np.where
        # **also note we avoid converting from samples to s
        # until *after* we find segments** 
        onsets = np.where(above_th_convoluted > 0)[0]
        offsets = np.where(above_th_convoluted < 0)[0]

    if onsets.shape[0] < 1 or offsets.shape[0] < 1:
        return onsets, offsets  # because no onsets or offsets in this file

    # get rid of silent intervals that are shorter than min_silent_dur
    silent_gap_durs = onsets[1:] - offsets[:-1]  # duration of silent gaps
    if samp_freq is not None:
        # need to convert to s
        silent_gap_durs = silent_gap_durs / samp_freq
    keep_these = np.nonzero(silent_gap_durs > segment_params['min_silent_dur'])
    onsets = np.concatenate(
        (onsets[0, np.newaxis], onsets[1:][keep_these]))
    offsets = np.concatenate(
        (offsets[:-1][keep_these], offsets[-1, np.newaxis]))

    # eliminate syllables with duration shorter than min_syl_dur
    syl_durs = offsets - onsets
    if samp_freq is not None:
        syl_durs = syl_durs / samp_freq
    keep_these = np.nonzero(syl_durs > segment_params['min_syl_dur'])
    onsets = onsets[keep_these]
    offsets = offsets[keep_these]

#    if samp_freq is not None:
#        onsets = onsets / samp_freq
#        offsets = offsets / samp_freq

    return onsets, offsets

#Used to circumvent filtfilt if needed. See	bandpass filtfilt function
def filter_song(b,a,rawsong):
    filt_len = len(b)
    rawsong_len = len(rawsong)
    filtered_song = []
    extended_rawsong = np.zeros(filt_len+rawsong_len)
    extended_rawsong[filt_len:len(extended_rawsong)] = rawsong
    for n in range(0,rawsong_len):
        result=0
        local_signal = extended_rawsong[n:(filt_len+n)]
        local_signal = local_signal[::-1]
        local_signal = local_signal*b
        result = np.sum(local_signal)
        filtered_song.append(result)	

    filtered_song = np.asarray(filtered_song)	
    return filtered_song

# Text box widget to collect labels
class myWidget():
    def __init__(self, ax, label, initial='',color='.95', hovercolor='1', label_pad=.01):
        self.widget = mplw.TextBox(ax, label, initial='',color='.95', hovercolor='1', label_pad=.01)
        self.annotation = ''        # The label for corresponding syllable manually entered by user
        self.label = label          # The label of the box (text that appears next to textbox)
    def submit(self,text):
        self.annotation = text
        print(self.label, ':', text)

# Button widget to end labeling
class myButton():
    def __init__(self, ax, label, image=None, color='0.85', hovercolor='0.95'):
        self.widget = mplw.Button(ax, label, image=None, color='0.85', hovercolor='0.95')
    def finalise(self,val):
        file_path_target_clean = target_path_song+'/'+base_filename
        file_path_target_annot = target_path_annot+'/'+base_filename[0:-4]+'_annot.txt'
        os.rename(songfile, file_path_target_clean)
        file_to_write= open(file_path_target_annot,"w+")
        for j in range(0, shpe):
            file_to_write.write("%d,%d,%s\n" % (onsets[j],offsets[j],textboxes[j].annotation))
            #Write to file from buffer, i.e. flush the buffer
        file_to_write.flush()
        file_to_write.close
        print('---Final labels---')
        for index, item in enumerate(textboxes):
            print(item.label,':',item.annotation,':',onsets[index],'-',offsets[index])
        plt.close('all')
    def reject_labeling(self,val):
        file_path_target_noise = target_path_noise+'/'+base_filename
        os.rename(songfile, file_path_target_noise)
        plt.close('all')
    def stop_labeling(self,val):
        plt.close('all')
        sys.exit()





# Thread to ask if the file should be considered for labeling or not
#class Ask_Labels(Thread):
#
#    def __init__(self, nb_segmented_syls):
#        Thread.__init__(self)
#        self.nb_segmented_syls = nb_segmented_syls
#
#    def run(self):
#        global labels 
#        global syl_counter
#        global Nb_syls
#        global keep_song 
#        syl_counter=0        
#        labels = []
#        Nb_syls=self.nb_segmented_syls
#        print("Total of %d segmented syllables" % Nb_syls)
#        keep_song = input("Keep file for segmentation? y or (anything) ")
#
## Thread to check if you want to stop labeling
#class Ask_Stop(Thread):
#    
#    def __init__(self):
#        Thread.__init__(self)
#    
#    def run(self):
#        global stop_labelling
#        stop_labelling = input("Stop now? (anything/no)")

#######################################################################################
## Block 1: Go to right folder, select the song file to be processed, filter and segment the song
#######################################################################################
##Go to the right location
#pwd = os.getcwd()
#os.chdir('C:/Users/Roman/Documents/Documents/SyllablesClassification/Koumura Data Set/Song_Data/Test/wav/raw')
#source_path = 'C:/Users/Roman/Documents/Documents/SyllablesClassification/Koumura Data Set/Song_Data/Test/wav/raw'
#target_path_song = 'C:/Users/Roman/Documents/Documents/SyllablesClassification/Koumura Data Set/Song_Data/Test/wav/clean'
#target_path_annot = 'C:/Users/Roman/Documents/Documents/SyllablesClassification/Koumura Data Set/Song_Data/Test/wav/annot'
#if len(sys.argv) < 2:
#    raise ValueError('Folder name is not provided.')

# Parse folder
folder_name = sys.argv[1]
if os.path.isdir(folder_name) is False:
    raise ValueError("Not a folder.")

print(folder_name)
source_path = folder_name + '/Raw_songs'
target_path_song = folder_name + '/Clean_songs'
target_path_annot = folder_name + '/Annotations'
target_path_noise = folder_name + '/Noise_songs'
if not os.path.exists(source_path):
    raise ValueError('Raw_songs folder does not exist.')
#os.chdir(source_path)
if not os.path.exists(target_path_song):
    os.mkdir(target_path_song)
    print('Created folder Clean songs.')
if not os.path.exists(target_path_annot):
    os.mkdir(target_path_annot)
    print('Created folder Annotations.')
if not os.path.exists(target_path_noise):
    os.mkdir(target_path_noise)
    print('Created folder Noise.')

#os.chdir('/Users/rsankar/Documents/Pipeline/SongbirdNeuralDataAnalsysis-master/Syllable sorting/Sample/Raw_songs')
#source_path = '/Users/rsankar/Documents/Pipeline/SongbirdNeuralDataAnalsysis-master/Syllable sorting/Sample/Raw_songs'
#target_path_song = '/Users/rsankar/Documents/Pipeline/SongbirdNeuralDataAnalsysis-master/Syllable sorting/Sample/Clean_songs'
#target_path_annot = '/Users/rsankar/Documents/Pipeline/SongbirdNeuralDataAnalsysis-master/Syllable sorting/Sample/Annotations'
#Enter name of the file to be processed
#songfile = 'file_1550590539.txt'
#songfile = 'file_1550590539.txt'

#Take all files from the directory
#songfiles_list = glob.glob('*.wav')
if rec_system == 'Alpha_omega':
    fs = 22321.4283
elif rec_system == 'Neuralynx':
    fs = 32000
print('fs:',fs)
songfiles_list = glob.glob(source_path + '/*.txt')


#file_num is the index of the file in the songfiles_list
for file_num, songfile in enumerate(songfiles_list):
    base_filename = os.path.basename(songfile)

    #Read song file	
    print('File name: %s' % songfile)
    rawsong = np.loadtxt(songfile)
    
	#Bandpass filter, square and lowpass filter
	#cutoffs : 1000, 8000
    amp = smooth_data(rawsong,fs,freq_cutoffs=(1000, 8000))

	#Segment song
    (onsets, offsets) = segment_song(amp,segment_params={'threshold': threshold, 'min_syl_dur': min_syl_dur, 'min_silent_dur': min_silent_dur},samp_freq=fs)
    shpe = len(onsets)
    if shpe < 1:
        file_path_target_noise = target_path_noise+'/'+base_filename
        os.rename(songfile, file_path_target_noise)
        continue
	
    ########################################################################################
    # Create thread for labels input
#    thread_1 = Ask_Labels(shpe)
    # Start thread
#    thread_1.start()
    ########################################################################################
    
    # Create figure
    fig1, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    plt.setp(ax1.get_xticklabels(), visible=True)
    plt.setp(ax2.get_xticklabels(), visible=True)

    #Compute and plot spectrogram
    (f,t,sp)=scipy.signal.spectrogram(rawsong, fs, window, nperseg, noverlap, mode='complex')
    ax3.imshow(10*np.log10(np.square(abs(sp))), origin="lower", aspect="auto", interpolation="none")
    #fig.colorbar()
    
    #Plot rawsong
    x_amp=np.arange(len(amp))
    ax1.plot(x_amp*len(t)/x_amp[-1], rawsong)
    ax1.set_xlim([0, len(t)])
    ax1.xaxis.set_tick_params(which='both', labelbottom=True)
    for i in range(0,shpe):    #Plot onsets and offsets
        ax1.axvline(x=onsets[i]*len(t)/x_amp[-1])
        ax1.axvline(x=offsets[i]*len(t)/x_amp[-1],color='r')
        ax1.text((onsets[i]+offsets[i])/2*len(t)/x_amp[-1], -max(rawsong)*0.75, str(i), va='top', ha='center', size='xx-small')


    ##Plot smoothed amplitude of the song
    ax2.plot(x_amp*len(t)/x_amp[-1], amp)
    ax2.set_xlim([0, len(t)])
    ax2.set_ylim([0, 3e-8])
    for i in range(0,shpe):    #Plot onsets and offsets
        ax2.axvline(x=onsets[i]*len(t)/x_amp[-1],color='b')
        ax2.axvline(x=offsets[i]*len(t)/x_amp[-1],color='r')
        ax2.text((onsets[i]+offsets[i])/2*len(t)/x_amp[-1], 3e-8*0.75, str(i), va='top', ha='center', size='xx-small')
    ax2.axhline(y=threshold,color='g')

    #Plotting widgets for collecting labels
    fig2 = plt.figure()
    axbox = [plt.axes([0.15+0.1*(i%8), 0.1+0.1*(i//8), 0.05, 0.05]) for i in range(shpe)]
    ax_bd = plt.axes([0.15, 0.1+0.1*(1+shpe//8), 0.5, 0.05])
    ax_bs = plt.axes([0.15, 0.1+0.1*(2+shpe//8), 0.5, 0.05])
    ax_br = plt.axes([0.15, 0.1+0.1*(3+shpe//8), 0.5, 0.05])
    textboxes = [myWidget(axbox[i], label=str(i)) for i in range(shpe)]
    for text_box in textboxes:
        text_box.widget.on_submit(text_box.submit)
    button_done = myButton(ax_bd, 'Finalise labeling')
    button_done.widget.on_clicked(button_done.finalise)
    button_reject = myButton(ax_br, 'Reject labeling')
    button_reject.widget.on_clicked(button_reject.reject_labeling)
    button_stop = myButton(ax_bs, 'Stop labeling')
    button_stop.widget.on_clicked(button_stop.stop_labeling)
    plt.show()

    plt.close('all')


    #Wait for the labeling thread to finish
#    thread_1.join()

    #Write file with onsets, offsets, labels

	#File with song, keep it
#    if keep_song=='y':
#       current_dir = os.getcwd()
#       file_path_source = source_path+'/'+songfile

    #File with song, remove it
#    else:
#       file_path_target_noise = target_path_noise+'/'+base_filename
#       os.rename(songfile, file_path_target_noise)
#       plt.close('all')
#       os.remove(songfile)


    # Create thread for continue input
#    thread_2 = Ask_Stop()
#    thread_2.start()
#    thread_2.join()
#    if stop_labelling=='no':
#        plt.close('all')
#    else:
#        plt.close('all')
#        break
