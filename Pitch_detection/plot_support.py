
"""
Provides support to plot and analyse FFTs and pitches.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def read_fft():
    """ To read fft files. """
    
    print("Loading fft.")

    fft_f, fft_e = np.loadtxt('fft_file.txt', delimiter=',', unpack=True)
    pitch = np.loadtxt('pitch_file.txt')

    return fft_f, fft_e, pitch


def read_pitches():
    """ To read fft files. """
    
    print("Loading pitches.")

    pitches = np.loadtxt('pitch_file.txt')

    return pitches


def plot_fft(fft_freq, fft_fine, pitch_Hz, fs=32000):
    """ Plots final FFT. """

    range_beg = np.abs(fft_freq-600).argmin()                                                   # Restricts search range for pitch 
    range_end = np.abs(fft_freq-1200).argmin()

    
    resolution = np.mean([ fft_freq[i+1] - fft_freq[i] for i in range(len(fft_freq)-1) ])       # Finds resolution of FFT
    print('Resolution:', resolution, 'Hz.')
    
    fig2, (ax3) = plt.subplots(nrows=1)
    ax3.set_title('FFT with resolution: ' + str(resolution) + 'Hz.')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Energy')
    ax3.plot(fft_freq, np.abs(fft_fine), linewidth=0.5, color='grey')                           # Plots FFT
    ax3.axvline(x=fft_freq[range_beg],color='b', label='Search range')                          # Lower limit of pitch detection range
    ax3.axvline(x=fft_freq[range_end],color='b')                                                # Upper limit of pitch detection range
    ax3.axvline(x=pitch_Hz,color='green', label='Pitch', linestyle='-.')                        # Detected pitch
    ax3.legend()
    plt.show()
    
    return


def analyse_pitches(pitches, N, lag):
    """ To run various analysis on pitches."""
            
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




# fft_f, fft_e, pitch = read_fft()                                                              # Uncomment if you wish to read and plot a single FFT and pitch
# plot_fft(fft_f, fft_e, pitch)

pitches = read_pitches()                                                                        # Loads and analyses pitches calculated over multiple renditions of syllable F
analyse_pitches(pitches, 512, 200)

