<h1>Deep Learning Big Project</h1>

<b>Team name</b>: Golden Oreos

<b>Team members</b>:
- Köller Donát Ákos (D813GB)
- Vastag Emese (OTNB7G)
- Vlaszov Artúr (XKMPWF)

<b>Project</b>:
Our main objective is to reconstruct the spectral features of speech from intracranial EEG data. (BRAIN2SPEECH)<br>

This project is heavily reliant on the research conducted by Maxime Verwoert
et al. (for which the <a href="https://osf.io/nrgx6/" rel='nofollow'>dataset</a> that we use
and the <a href="https://www.nature.com/articles/s41597-022-01542-9" rel='nofollow'>article about the research</a> 
can be found here),
and we are trying to improve on it. We are aiming to replace the linear regression model described in the research with 
autoencoders, transformers and conventional neural network models, while we are also trying to make a speaker-independent system.

The notebooks and scripts require Python >= 3.6 and the following packages:
- numpy (1.21.6)
- scipy (1.7.3)
- scikit-learn (1.0.2)
- pandas (1.3.5)
- pywnb (2.2.0)

The scripts used for data preparation and preprocessing are the same that Maxime Verwoert et al. 
<a href="https://github.com/neuralinterfacinglab/SingleWordProductionDutch">used</a> with minor changes. 
The function of each script:
- <b>MelFilterBank.py</b>: This is used to apply mel filter banks to the spectograms.
- <b>extract_features.py</b>: Reads in the iBIDS dataset and creates the features used for modeling.
- <b>reconstructWave.py</b>: Used for audio waveform synthesis. (Applies Fourier-transformations)

They first script serves as a module used by the other scripts, while the latter two of the scripts can be run from 
command line. The only requirement is that the directory which contains the scripts must also contain the main directory 
(named "SingleWordProductionDutch-iBIDS") of the dataset.


The notebook "Modeling.ipynb" will be used for the audio reconstruction, and is mainly based on the "reconstruction_minimal.py" script 
that Maxime Verwoert et al. used. 

<h2> Data Preparation and modeling </h2>

We obtained the desired feature and label vectors by running the "extract_features.py" script. 
Thus, for each 10 subjects, we got the following attributes stored as numpy arrays:
- The spectogram of the original audio (this is what we aim to reconstruct)
- The features transformed from the EEG data
- The featurename vector
- The words corresponding to the spectogram in each timestamp

From this, we are focusing mainly on the first two, that is, we want to reconstruct the spectogram from the EEG feature vectors.
For the one-speaker model, the method of reconstruction is the following for each subject:
- We divide the set of feature vectors into <i>k</i> equal parts (where the initial value for <i>k</i> is 10, but we would like to experiment with 
other options as well).
- We do <i>k</i> iterations. In each iteration, we label one of the parts as test set (a different, never previously used part in each iteration),
and another part as validation set, then we train the neural networks on the other <i>k-2</i> parts, and finally, we reconstruct that part of the spectrogram which corresponds 
to the test set.
- After <i>k</i> iteration, we completely reconstructed the spectrogram, so we compare it to the original.

For the speaker-free model, we chose 6 individuals to serve as test set, 2 other as validation set and the remaining 2 as test set. 
The distribution of the subject into sets were based sex and age:
- Train set subject: 5, 6, 7, 8, 9, 10 (3 male, 3 female, mean age: 31.83)
- Validation set subjects: 1, 2 (20 years old female, 43 years old male)
- Test set subjects: 3, 4  (24 years old male, 46 years old female)

The modeling for the speaker-indepenent system is straight-forward: we train the neural networks on the test set, validate them on the 
validation set, and test them on the test set.

  
