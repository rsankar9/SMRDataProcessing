"""
author : R Sankar

Just a script to test the smr data file with some basic functions

"""

from songbird_data_analysis import functions
import sys

filename = sys.argv[1]
print('Reading file.')
functions.read(filename)
print('Getting info.')
functions.getinfo(filename)
print('Getting arrays.')
functions.getarrays(filename)
print('Getting song.')
functions.getsong(filename)
#print('Plotting plots.')
#functions.plotplots(filename)
print('All good.')
