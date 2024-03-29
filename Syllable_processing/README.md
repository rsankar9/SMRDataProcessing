### Folder: Syllables_processing

#### Script: Auxilliary_support.py

- This scripts plots the song in an npy/txt file, in an interactive manner.
- Use line 228-229 to select portions of the file.
- Not meant to be a generic code for public use.
- Just an auxilliary file to quickly visualise the song and adjust the syllable segmenting parameters.
- Parameters should be adjusted per bird.
- Requires Python 3.7.3 and other packages.
- To run: `python Auxilliary_support.py path_to_npy_file.npy` or `python Auxilliary_support.py path_to_txt_file.txt`.


#### Script: Manual_labeling.py

- Give the path to the parent folder of Raw_songs as an argument when you run the python script: `python Manual_labeling.py parent_folder_name`.
- Ensure this folder has a folder called Raw_songs with the songfiles (.txt) to be labelled.
- Ensure this folder doesn't have a clashing Annotations, Clean_songs or Noise_songs folder (as some files will be moved/created in these folders).
- To change the threshold, rec_system, or other parameters, you'll have to directly change the code.

#### Script: Slicing_Songfile.py

- This scripts extracts the song from the smr file and splits it into smaller chunks (txt), for ease in further processing.
- To run: `python Slicing_Songfile.py path_to_file.smr`.

#### Script: Detect_syllables.py

- Get files from Raw_songs folder.
- Detects if it has syllables.
- If not, the file is moved to Noise_songs folder.
- Ensure the segmenting parameters are correct.
- To run: `python Detect_syllables.py path_to_parent_folder`

#### Script: collect_durations.py

- Collects data and stats on duration of each syllable across renditions and records in a json file.
- Verify sample frequency.
- Give argument to parent folder of Clean_songs and Annotations folders to run: `python collect_durations.py path_to_parent_folder`

#### Script: collect_pitch.py

- This script detects pitches of the syllables present in annotated files.
- Data and confidence measures are stored in a json file.
- Verify parameters: sample frequency, syllable window, chunk_size, varying_window_dur, etc.
- Give argument to parent folder of Clean_songs and Annotations folders to run: `python collect_pitch.py path_to_parent_folder`

---

#### Folder: HVC\_config\_files

- Folder with sample config files to run HVC syllable labeler.
- Path and segmenting parameteres in config files should be updated for each run.
- To run:

`hvc.extract('Extract.yml')`

`hvc.select('Select.yml')`

`hvc.predict('Predict.yml')`


---
