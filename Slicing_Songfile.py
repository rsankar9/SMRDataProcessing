"""
    This scripts extracts the song from the smr file and splits it into smaller chunks (txt), for ease in further processing.
    To run: python Slicing_Songfile.py path_to_file.smr
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

def read(file):
    reader = neo.io.Spike2IO(filename=file) #This command will read the file defined above
    #print(reader)
    data = reader.read()[0] #This will get the block of data of interest inside the file
    #print(data)
    data_seg=data.segments[0] #This will get all the segments
    #print(data_seg)
    return data, data_seg

def getsong(file):
    data, data_seg = read(file)
    song=[]
    for i in range(len(data_seg.analogsignals)):
        if data_seg.analogsignals[i].name == 'Channel bundle (CSC5) ':
            #        if data_seg.analogsignals[i].name == 'Channel bundle (RAW 009) ':
            song=data_seg.analogsignals[i].as_array()
        else:
            continue
        np.save(file[:-4]+"_songfile", song)
    return song

#---------------------------------------------------------------------------#

if len(sys.argv) < 2:
    raise ValueError('Filename is not provided.')
smr_file = sys.argv[1]
if os.path.isfile(smr_file) is False:
    raise ValueError("Not a file.")
if smr_file[-4:] != '.smr' and smr_file[-4:] != '.SMR':
    raise ValueError("Given path doesn't lead to an smr file.")
base_foldername = os.path.dirname(smr_file)
base_filename = os.path.basename(smr_file)

rawsong = getsong(smr_file)
rawsong = rawsong.astype(float)
rawsong = rawsong.flatten()

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
    file_path = raw_songs_dir_name + '/' + base_filename[0:-4]+'_raw_chunk_'+str(cn)+'.txt'
    np.savetxt(file_path, rs, '%13.11f')

    cn += 1


