##Data Analysis of SMR files
**In progress**

The goal is to eventually correlate the cross-rendition average of syllable pitch with their premotor activity.

####Acknowledgements
Eduarda Centeno
Roman Ursu
Arthur Leblois
For HVC - automatic syllable labeller - The Sober Lab at Emory University
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1475481.svg)](https://doi.org/10.5281/zenodo.1475481)


### Auxilliary_support.py

This scripts plots the song in an npy/txt file, in an interactive manner.
Use line 228-229 to select portions of the file.
Not meant to be a generic code for public use.
Just an auxilliary file to quickly visualise the song and adjust the syllable segmenting parameters.
Parameters should be adjusted per bird.
Requires Python 3.7.3 and other packages.

To run:

`python Auxilliary_support.py path_to_npy_file.npy`

or

`python Auxilliary_support.py path_to_txt_file.txt`


### Manual_labeling.py

- Give the path to the parent folder of Raw_songs as an argument when you run the python script: `python Manual_labeling.py parent_folder_name`
- Ensure this folder has a folder called Raw_songs with the songfiles (.txt) to be labelled.
- Ensure this folder doesn't have a clashing Annotations, Clean_songs or Noise_songs folder (as some files will be moved/created in these folders).
- To change the threshold, rec_system, or other parameters, you'll have to directly change the code.
