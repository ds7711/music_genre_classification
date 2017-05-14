# music_genre_classification
CNN achieves human-level accuracy in music genre classification
This is part of the course project for Rutgers Machine Learning class (2017 Spring). 

We used a simple convolutional neural network to perform music genre classification and achieve 70% accuracy, which is comparable to the human accuracy. 

To run the code, one has to either download the GTZAN dataset (http://marsyasweb.appspot.com/download/data_sets/) and convert them into mel-spectrogram (64 mel-filters), or download the already converted data (https://drive.google.com/file/d/0B3I2KG9W0zM2YmFWbk1oMFhkU1k/view?usp=sharing). Note: the preprocessed data (.npz) file contains the mel-spectrogram of the audio and the corresponding genre. Mel-spectrogram is converted using Librosa library in python. 

After downloading the data, simply run the music_classification_main.py to train the model. random_seed (0-29) determines how the data are split into training, validation, and testing set. The codes were tested under ubuntu 14.04 using Keras and tensorflow with a GTX-1070. On GPU, it roughly takes 10-30 minutes for the model to converge. 

For the analysis of the model, run the notebook files. 

A detailed explanation of the code is in the PDF. 
