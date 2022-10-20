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
autoencoders and transformer models, while we are also trying to make a speaker-independent system.

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


The notebook "Modeling.ipynb" will be used for the audio reconstruction. 
