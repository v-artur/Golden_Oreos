<h1>Deep Learning Big Project</h1>

<b>Team name</b>: Golden Oreos

<b>Team members</b>:
- Köller Donát Ákos (D813GB)
- Vastag Emese (OTNB7G)
- Vlaszov Artúr (XKMPWF)

<b>Project</b>:
Our main objective is to reconstruct the spectral features of speech from intracranial EEG data. (BRAIN2SPEECH)<br>

This project is heavily reliant on the research conducted by Maxime Verwoert
et al. for which the article can be found <a href="https://www.nature.com/articles/s41597-022-01542-9" rel='nofollow'>here</a> and
the dataset that we use can be obtained from <a href="https://osf.io/download/g6q5m/" rel='nofollow'>here</a> (after downloading it, unzip it). We are aiming to replace the linear regression model described in the paper with recurrent and fully connected dense neural network models while we are also trying to implement a speaker-independent system.

The notebooks and scripts require Python >= 3.6 and the following packages:
- numpy (1.21.6)
- scipy (1.7.3)
- matplotlib (3.2.2)
- scikit-learn (1.0.2)
- pandas (1.3.5)
- pywnb (2.2.0)
- tensorflow (2.9.2)
- fastdtw (0.3.4)
- librosa (0.8.1) 
- pyworld (0.3.2) 
- pysptk (0.2.0) 
- gdown (4.4.0)
- (List will be updated as the project progresses)

<b>Important note</b>: To load in the required biga datasets, we mainly use the 'gdown' library to download the data from our Google Drive. However, sometimes Google denies every type of access to the files (however big they are). According to our experiences, this can occur randomly even if we don't try to access the data for days. If that were to happen, the download links to the files will be shown on the screen, so you can still obtain them.

The scripts used for data preparation and preprocessing are the same that Maxime Verwoert et al. 
<a href="https://github.com/neuralinterfacinglab/SingleWordProductionDutch">used</a> with minor to no changes. 
The function of each script:
- <b>MelFilterBank.py</b>: This is used to apply mel filter banks to the spectograms.
- <b>extract_features.py</b>: Reads in the iBIDS dataset and creates the features used for modeling.
- <b>reconstructWave.py</b>: Used for audio waveform synthesis. (Applies Fourier-transformations)

The first script serves as a module used by the other scripts, while the latter two of the scripts can be run from 
command line. The only requirement is that the directory which contains the scripts must also contain the main directory 
(named "SingleWordProductionDutch-iBIDS") of the dataset.


The notebooks <b>"Modeling1.ipynb"</b>  and <b>"speaker_indep.ipynb"</b> will be used for the spectrogram reconstruction, and is partially based on the "reconstruction_minimal.py" script 
that Maxime Verwoert et al. used. It also requires the "MelFilterBank.py" and "reconstructWave.py" scripts. The required data and scripts will be downloaded from Google Drive during the modeling.

The <b>"Important urls"</b> directory contains one Word documentum named "URLS.docx", which lists the references we have checked so far.

<h2> Data Preparation and Modeling </h2>

The dataset contains various information about the 10 test subjects (gender, age) and their recordings (coordinates of the implanted electrodes, raw data streams etc.) which are further described in detail in the article mentioned above. From the <i>_ieeg.nwb</i> files which contain the iEEG, Audio and Stimulus raw data streams we obtained the desired feature and label vectors by running the "extract_features.py" script (We uploaded the resulted files onto Google Drive and can be viewed <a href="https://drive.google.com/drive/folders/1pdc95RPUk-Zh0J8kaYo8cXz_ickSOwcB?usp=sharing">here</a> along with the original audiofiles). 
Thus, for each 10 subjects, we got the following attributes stored as numpy arrays:
- The mel spectrogram of the original audio (this is what we aim to reconstruct)
- The features transformed from the EEG data
- The featurename vector
- The words corresponding to the spectrogram in each timestep

To understand the data better, in the <b>"Data_visualization.ipynb"</b> notebook we plotted some example audio files. Using short time Fourier-transformation we created spectrograms from the wave files and plotted them on a short interval, and from these we reconstructed the wave form to see if it matches the original data. On another spectrogram we also showed the words that were pronounced in the record.

From now on, we will focus mainly on the first two attributes, that is, we want to reconstruct the mel spectrogram from the EEG feature vectors.
For the one-speaker model, the main method of reconstruction is the following for each subject:
- We divide the set of feature vectors and their corresponding labels into <i>k</i> equal parts (where the initial value for <i>k</i> is 10, but we would like to experiment with other options as well).
- We do <i>k</i> iterations. In each iteration, we label one of the parts as test set (a different, never previously used part in each iteration),
a portion of the remaining <i>k-1</i> parts as validation set, and the rest of the set as training set. We then train the models on the train set, validate them on the validation set, and finally reconstruct the part of the mel spectrogram that corresponds to the test set.
- After <i>k</i> iterations, we completely reconstructed the mel spectrogram, so we compare it to the original.

For the speaker-independent model, we chose 6 individuals to serve as train set, 2 other as validation set and the remaining 2 as test set. 
The distribution of the subject into sets were based on sex and age:
- Train set subject: 5, 6, 7, 8, 9, 10 (3 male, 3 female, mean age: 31.83)
- Validation set subjects: 1, 2 (20 years old female, 43 years old male)
- Test set subjects: 3, 4  (24 years old male, 46 years old female)

The modeling for the speaker-independent system is straight-forward: we train the neural networks on the train set, validate them on the 
validation set, and test them on the test set.

<h3> Early Results (2nd milestone) </h3>

(Quick disclaimer: When we finalized the results before the deadline, we did not have access to Google Colab's GPU, so we had to use CPU to run the notebooks. So when you run the notebook on GPU, you might get slightly different results from those shown below. Another thing to note is that if you run the notebook from Google Drive, the notebook tends to put "ipynb.checkpoint" files into specific folders, which could mess up the MCD calculation functions, so we advise you to run the notebook using the "Open in Colab" button.)

We experimented from two different angles: modeling for one speaker (which was the participant 1 in our case) and making a speaker-independent system. The modeling and evaluation for one speaker can be found in the <b>"Modeling1.ipynb"</b> notebook while the <b>"speaker_indep.ipynb"</b> notebook contains the results for the speaker-independent system. 

<b>One speaker model:</b>

The "Modeling1.ipynb" notebook is sectioned into 4 parts: the preparation part (where the different models and functions are defined), the "One Person Model" part, the "MCD score evaluation" part and the "Trying out the best configuration for every subject" part. You can run the whole notebook to train the models and get the results altough only the first three parts are relevant (the "Trying out the best configuration for every subject" part is still experimental and takes about 2-3 hours to run, but it is commented out for safety reasons). The first part downloads the datasets and scripts, imports the modules and defines the necessary functions. The second part called "One Person Model" is where the networks' training happens and some of the evaluation metrics are also calculated within this part. The "MCD score evaluation" is where we evaluate the MCD (Mel Cepstral Distortion) of the original and the synthsized audio files.

We tried out two different neural network models: a normal 3-layer FC-DNN and a 5-layer FC-DNN with a bottleneck layer. We trained each model from scratch during each iteration of the reconstruction with ADAM optimizer and we used MSE as loss. To lower the computation time in the iterations, we standardized and transformed the original feature vectors into a lower dimensional space with PCA (200 for the bottleneck model, 100 for the normal) and we only trained the models for 100 epochs. The number of iterations during the reconstruction were 10 for the DNN models.
For evaluation, we used two metrics: <b>mean Pearson correlation</b> and <b>MCD</b>. The results are the following:

<table>
<thead>
  <tr>
    <th>Model </th>
    <th>Mean Pearson correlation</th>
    <th>MCD</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Bottleneck FC-DNN</td>
    <td>0.5915</td>
    <td>7.7355</td>
  </tr>
  <tr>
    <td>Normal FC-DNN</td>
    <td>0.5459</td>
    <td>3.7586</td>
  </tr>
</tbody>
</table>

<b>Speaker-independent system:</b> <br>
You need to run the whole "speaker_indep.ipynb" notebook to train the model and get the results. For the speaker-independent model, we first concatenated the data of participants 5-10 to be used as our training data, and similarly the data of participants 1-2 and 3-4 for the validation and test data respectively. We applied normalization and PCA on the feature vectors to get 300 dimensional data. The model we used is a simple fully-connected network with 5 layers using ADAM optimizer and MSE as loss function. The Pearson correlation of the result is very low (0.1754 for participant 3 and 0.0973 for participant 4), we are planning on improving it. For participant 4, we also computed the MCD value, which was 5.349841.


<b>Further plans/goals</b> <br>
We intend to fully optimize the one-speaker models + trying out the best performing one on the other subjects. But as of now, the training of one model takes around 20 minutes, so we would like to lower the training time as well. We are also planning on implementing reccurent networks such as LSTM and GRU on the speaker-independent system in order to achieve better performance. If we have time, we would like to try out other speech synthesizer models (such as WaveGlow) as well.

