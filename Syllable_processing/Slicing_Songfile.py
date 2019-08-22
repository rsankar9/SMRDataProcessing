"""
    This scripts splits the song into smaller chunks (txt), for ease in further processing.
    To run: python Slicing_Songfile.py path_to_file.npy / .txt
"""

import numpy as np
import scipy as sp
import neo
from scipy.io import loadmat
import scipy.signal
from scipy.io import wavfile
import matplotlib as mplt
import matplotlib.pyplot as plt
import os
import glob
from threading import Thread
import sys
from songbird_data_analysis import functions


window =('hamming')
overlap = 64
nperseg = 1024
noverlap = nperseg-overlap
colormap = "jet"

#threshold=2e-9 # for files recorded with Neuralynx
#threshold=2e-5
threshold=1.25e-9
min_syl_dur=0.02
min_silent_dur= 0.003
smooth_win=10

#rec_system = 'Alpha_omega' # or 'Neuralynx' or 'Other'
rec_system = 'Neuralynx'



#---------------------------------------------------------------------------#

songfile = sys.argv[1]                      # npy file
if len(sys.argv) < 2:
    raise ValueError('Filename is not provided.')
    
base_filename = os.path.basename(songfile)  # Extracts filename
base_foldername = os.path.dirname(songfile)

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
rawsong = rawsong.astype(float)
rawsong = rawsong.flatten()
print('size=',rawsong.size, 'shape=',rawsong.shape)

raw_songs_dir_name = base_foldername + "/Raw_songs"
if not os.path.exists(raw_songs_dir_name):
    os.mkdir(raw_songs_dir_name)

chunk_size = 120000
cn = 0

print('no. of chunks:', np.ceil(rawsong.size / chunk_size))
while chunk_size * cn < rawsong.size:
    rs = rawsong[chunk_size * cn : chunk_size * (cn+1)]
    
    #Write chunk of raw data
    print('Writing file no:', cn)
    file_path = raw_songs_dir_name + '/' + base_filename[0:-13]+'_raw_chunk_'+str(cn)+'.txt'
    np.savetxt(file_path, rs, '%13.11f')

    cn += 1


