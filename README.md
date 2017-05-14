# music_genre_classification
CNN achieves human-level accuracy in music genre classification
This is part of the course project for Rutgers Machine Learning class (2017 Spring). 

It uses the convolutional neural network to perform music genre classification and achieves 70% accuracy. 

To run the code, one has to either download the GTZAN dataset (http://marsyasweb.appspot.com/download/data_sets/) and convert them into mel-spectrogram (64 mel-filters), or download the already converted data.

After the data is ready, simply run the music_classification_main.py to train the model. random_seed (0-29) determines how the data are split into training, validation, and testing set. The codes were trained under ubuntu 14.04 using Keras and tensorflow with a GTX-1070. 

For the analysis of the model, run the notebook files. 

A detailed explanation of the code is in the PDF. 
