"""
Testing pitch detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_song():
    """ To read song files. """
    
    print("Loading signal.")
    with open("Combined_songs.txt") as song_file:
        song = song_file.readlines()                                            

    print("Loading labels.")
    labels = np.loadtxt("Combined_labels_tampered.txt", delimiter=',',
                        dtype={'names': ('onset', 'offset', 'syll_label'),
                               'formats': ('int', 'int', 'S1')}
                        )
    return song, labels


def plot_signal(fs, sig, N, pitch_Hz=0, onset=0, offset=0, lag=0):
    """ To plot signal amplitude and spectrogram. """
    
    t = np.arange(len(sig))

    # Plotting signal amplitude and spectrogram
    fig1, (ax1, ax2) = plt.subplots(nrows=2, sharex=False)
    ax1.plot(t,sig,color='grey')                                                            # Signal amplitude
    ax1.axvline(x=onset,color='b',alpha=0.8)                                                # Beginning of syllable considered
    ax1.axvline(x=offset,color='b',alpha=0.8)                                               # End of syllable considered
    ax1.axvline(x=onset+lag,color='g', linestyle='--')                                      # Beginning of samples considered
    ax1.axvline(x=onset+lag+N,color='g', linestyle='--')                                    # End of samples considered
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    ax2.specgram(sig, Fs=fs, cmap='inferno')                                                # Signal spectrogram
    ax2.axhline(y=np.abs(pitch_Hz),color='black', label='Pitch', linestyle='-.')            # Pitch detected
    ax2.axvline(x=onset/fs,color='blue',alpha=0.8, label='Syllable range')                  # Beginning of syllable considered
    ax2.axvline(x=offset/fs,color='blue',alpha=0.8)                                         # End of syllable considered
    ax2.axvline(x=(onset+lag)/fs,color='g', linestyle='--', label='Sampling range')         # Beginning of samples considered
    ax2.axvline(x=(onset+lag+N)/fs,color='g', linestyle='--')                               # End of samples considered
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')
    ax2.legend(loc='upper left')
    plt.show()
    
    return
    

def segment_signal(sig, N, start=0):
    """ To extract required samples with a lag from syllable onset. """
        
    start_ind = start
    end_ind = start + N
    signal = sig[start_ind:end_ind]
    
    return signal
  

def construct_fft(signal, N, fs):
    """ Construct a fine FFT by computing multiple FFTs of given signal segment. """
    
    # Computing multiple FFTs with different no. of samples
    ffts = np.empty(shape=[0,2])                                                    # To store multiple FFTs

    m=0                                                                             # Counter for no. of FFTs
    while m<20:
        sample = signal[:N-m]                                                                                         
        sp = np.fft.fft(sample)                                                     # FFT energy values
        freq = np.fft.fftfreq(N-m)                                                  # FFT frequency values
        fft_course = np.column_stack((freq, sp))                                       
        ffts = np.concatenate((ffts, fft_course))                                   # To record each FFT
        m+=2                                                                        # 10 FFTs calculated each with 2 sample less
        

    ffts = ffts[np.argsort(ffts[:,0].real)]                                         # Sorts the FFT according to frequency values
    fft_freq = ffts[:,0]
    fft_fine = ffts[:,1]

    return fft_freq, fft_fine


def calc_pitch(fft_freq, fft_fine, fs):
    """ Calculate pitch from fine FFT. """

    range_beg = (np.abs(fft_freq.real*fs-600)).argmin()                               # Detecting pitch only between 600 Hz and 1200 Hz
    range_end = (np.abs(fft_freq.real*fs-1200)).argmin()
    # Pitch = frequency at highest energy within a given range.
    pitch = fft_freq[range_beg + np.argmax(np.abs(fft_fine[range_beg:range_end].real))]
    # plot_fft(fft_freq, fft_fine, pitch * fs, fs, range_beg, range_end)
    
    return pitch * fs


def plot_fft(fft_freq, fft_fine, pitch_Hz, fs, range_beg, range_end):
    """ Plots final FFT. """
    
    resolution = np.mean([ fft_freq[i+1].real - fft_freq[i].real for i in range(len(fft_freq)-1) ]) # Finds resolution of FFT
    # print('Resolution:', resolution*fs, 'Hz.')
    
    fig2, (ax3) = plt.subplots(nrows=1)
    ax3.set_title('FFT with resolution: ' + str(resolution*fs) + 'Hz.')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Energy')
    ax3.plot(fft_freq.real*fs, np.abs(fft_fine.real), linewidth=0.5, color='grey')                  # Plots FFT
    ax3.axvline(x=fft_freq[range_beg].real*fs,color='b', label='Search range')                      # Lower limit of pitch detection range
    ax3.axvline(x=fft_freq[range_end].real*fs,color='b')                                            # Upper limit of pitch detection range
    ax3.axvline(x=pitch_Hz.real,color='green', label='Pitch', linestyle='-.')                       # Pitch detected
    ax3.legend()
    plt.show()
    
    return
    
def find_pitch(sig, N, fs, lag=0):
    """ Calculate pitch, given a signal. """
    # plot_signal(fs, sig)
    if len(sig) < lag + N:    return 0
    signal = segment_signal(sig, N, lag)
    fft_freq, fft_fine = construct_fft(signal, N, fs)
    pitch_Hz = calc_pitch(fft_freq, fft_fine, fs)
    
    return pitch_Hz

def collect_pitches(song, labels, N, fs, lag):
    """ Iterates over all renditions and processes syllable F occurences. """
    
    collected_pitches = []
    for l in labels:
        syll = str(l['syll_label'])
        if syll == str(b'f'):                                                   # Detects syllable F
            sig = song[ l['onset'] : l['offset'] ]
            sig = [ float(val) for val in sig ]
            pitch_Hz = find_pitch(sig, N, fs, lag)                              # Finds pitch
            # print('Pitch of ', syll, 'is ', pitch_Hz, 'Hz.')

            if pitch_Hz != 0:   collected_pitches.append(np.abs(pitch_Hz))      # Stores calculated pitch
            
            # To plot signal segment
            # sigview = song[ l['onset']-10000 : l['offset']+5000 ]               # Arbitrary padding to view the signal
            # sigview = [ float(val) for val in sigview ]
            # plot_signal(fs, sigview, N, pitch_Hz, 10000, len(sigview)-5000, lag)

    return collected_pitches

def analyse_pitches(pitches, N, lag):
    """ To run various analysis on collected pitches."""
            
    fig3 = plt.figure()
    
    violin_params = plt.violinplot(pitches, showmeans=True, showextrema=True, showmedians=True) # Plots histogram of distribution of pitches
    for pc in violin_params['bodies']:
        pc.set_facecolor('lightskyblue')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        vp = violin_params[partname]
        vp.set_edgecolor('grey')
        vp.set_linewidth(1)
    
    plt.ylabel('Pitches')
    plt.xticks([])
    fig3.suptitle('Histogram with N='+str(N)+', lag='+str(lag)+'.', fontsize=11)
    plt.savefig('Analysis/Histogram with N='+str(N)+', lag='+str(lag)+'.')

    
    fig4 = plt.figure()
    x_axis = np.arange(len(pitches))
    plt.scatter(x_axis, pitches, s=5, color='grey')                                             # Scatter plot of pitch distribution
    plt.ylabel('Pitches')
    plt.xlabel('Sample no.')
    fig4.suptitle('Distribution of pitches with N='+str(N)+', lag='+str(lag)+'.', fontsize=11)
    plt.savefig('Analysis/Distribution of pitches with N='+str(N)+', lag='+str(lag)+'.', fontsize=11)

    plt.show()

    
    

N = 512                                                                        # No. of samples considered
fs = 32000                                                                     # Sampling rate
lag = 200                                                                      # Lag w.r.t. syllable onset

song, labels = read_song()

# for N in [256, 512]:                                                          # For comparison
#    for lag in [0, 100, 200]:
collected_pitches = collect_pitches(song, labels, N, fs, lag)
analyse_pitches(collected_pitches, N, lag)
