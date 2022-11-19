<h1>Deep Learning Big Project</h1>

<b>Team name</b>: Golden Oreos

<b>Team members</b>:
- Köller Donát Ákos (D813GB)
- Vastag Emese (OTNB7G)
- Vlaszov Artúr (XKMPWF)

<b>Project</b>:
Our main objective is to reconstruct the spectral features of speech from intracranial iEEG data. (BRAIN2SPEECH)<br>

This project is heavily reliant on the research conducted by Maxime Verwoert
et al. for which the article can be found <a href="https://www.nature.com/articles/s41597-022-01542-9" rel='nofollow'>here</a> and
the dataset that we use can be obtained from <a href="https://osf.io/download/g6q5m/" rel='nofollow'>here</a> (after downloading it, unzip it). We are aiming to replace the linear regression model described in the paper with 
recurrent and fully connected dense neural network models while we are also trying to implement a speaker-independent system.

The notebooks and scripts require Python >= 3.6 and the following packages:
- numpy (1.21.6)
- scipy (1.7.3)
- matplotlib (3.2.2)
- scikit-learn (1.0.2)
- pandas (1.3.5)
- pywnb (2.2.0)
- tensorflow (2.9.2)
- (List will be updated as the project progresses)

The scripts used for data preparation and preprocessing are the same that Maxime Verwoert et al. 
<a href="https://github.com/neuralinterfacinglab/SingleWordProductionDutch">used</a> with minor to no changes. 
The function of each script:
- <b>MelFilterBank.py</b>: This is used to apply mel filter banks to the spectograms.
- <b>extract_features.py</b>: Reads in the iBIDS dataset and creates the features used for modeling.
- <b>reconstructWave.py</b>: Used for audio waveform synthesis. (Applies Fourier-transformations)

The first script serves as a module used by the other scripts, while the latter two of the scripts can be run from 
command line. The only requirement is that the directory which contains the scripts must also contain the main directory 
(named "SingleWordProductionDutch-iBIDS") of the dataset.


The notebook <b>"Modeling2.ipynb"</b> will be used for the spectrogram reconstruction, and is partially based on the "reconstruction_minimal.py" script 
that Maxime Verwoert et al. used. It also requires the "MelFilterBank.py" and "reconstructWave.py" scripts. The required data and scripts will be downloaded from Google Drive during the modeling.

The <b>"Important urls"</b> directory contains one Word documentum named "URLS.docx", which lists the references we have checked so far.

<h2> Data Preparation and Modeling </h2>

The dataset contains various information about the 10 test subjects (gender, age) and their recordings (coordinates of the implanted electrodes, raw data streams etc.) described in detail in the article mentioned above. From the <i>.nwb</i> files which contain the iEEG, Audio and Stimulus raw data streams we obtained the desired feature and label vectors by running the "extract_features.py" script (The resulted files can be viewed <a href="https://drive.google.com/drive/folders/1pdc95RPUk-Zh0J8kaYo8cXz_ickSOwcB?usp=sharing">here</a> along with the original audiofiles). 
Thus, for each 10 subjects, we got the following attributes stored as numpy arrays:
- The spectrogram of the original audio (this is what we aim to reconstruct)
- The features transformed from the EEG data
- The featurename vector
- The words corresponding to the spectrogram in each timestep

To understand the data better, in the <b>"Data_visualization.ipynb"</b> notebook we plotted some example audio files. Using short time Fourier-transformation we created spectrograms from the wave files and plotted them on a short interval, and from these we reconstructed the wave form to see if it matches the original data. On another spectrogram we also showed the words that were pronounced in the record.

From now on, we will focus mainly on the first two attributes, that is, we want to reconstruct the spectrogram from the EEG feature vectors.
For the one-speaker model, the main method of reconstruction is the following for each subject:
- We divide the set of feature vectors into <i>k</i> equal parts (where the initial value for <i>k</i> is 10, but we would like to experiment with 
other options as well).
- We do <i>k</i> iterations. In each iteration, we label one of the parts as a test set (a different, never previously used part in each iteration),
and another part as a validation set, then we train the neural networks on the other <i>k-2</i> parts, validate them on the validation set,
and finally, we reconstruct that part of the spectrogram which corresponds to the test set.
- After <i>k</i> iterations, we completely reconstructed the spectrogram, so we compare it to the original.

For the speaker-independent model, we chose 6 individuals to serve as train set, 2 other as validation set and the remaining 2 as test set. 
The distribution of the subject into sets were based on sex and age:
- Train set subject: 5, 6, 7, 8, 9, 10 (3 male, 3 female, mean age: 31.83)
- Validation set subjects: 1, 2 (20 years old female, 43 years old male)
- Test set subjects: 3, 4  (24 years old male, 46 years old female)

The modeling for the speaker-independent system is straight-forward: we train the neural networks on the train set, validate them on the 
validation set, and test them on the test set.

  
