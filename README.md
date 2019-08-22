##README

## Data Analysis of SMR files
**In progress**

The goal is to eventually correlate the cross-rendition average of syllable pitch with their premotor activity.

#### Acknowledgements
- Eduarda Centeno
- Roman Ursu
- Arthur Leblois
- For HVC - automatic syllable labeller - The Sober Lab at Emory University [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1475481.svg)](https://doi.org/10.5281/zenodo.1475481)

Python 3.7 is required, unless specified otherwise.

---

1. Given: An SMR file.
2. Checks if neo is able to read the file. ```python testDataFile.py file_name.smr``` This should also extract the songfile.
3. Use ```Auxilliary_support.py``` to check the thresholds for syllable segmentation.
4. Use ```Slicing_Songfile.py``` to slice the songfile into smaller chunks.
5. Use ```Detect_syllables.py``` to remove the chunks that don't contain any syllables.
6. Use an IDE which allows plots to be rendered outside the window (or the terminal), and run the Manual_labeling.py script to annotate as many songs as required.
7. Use ```HVC``` to automatically label the rest of the syllables.
7. Use ```Collect_durations.py``` to collect statistics on syllable durations.
8. You can use ```Plot_histogram_durations.py``` to plot directly from previously collected durations.
9. Use ```Collect_pitches.py``` to collect statistics on syllable pitches.
10. Use ```Plot_histogram_pitches.py``` to plot different analyses on syllable pitches.
11. Use ```Collect_spikes.py``` to collect statistics on spikes.
12. Use ```Plot_histogram_spikes.py``` to plot different analyses on spikes.
13. Use ```Plot_comparison.py``` to plot comparisons between spikes and syllable features.
