# store the function/object used in the project

# import modules
from __future__ import print_function
import numpy as np
import librosa
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from keras.utils import np_utils
from keras import regularizers
import time
from keras.engine.topology import Layer


# parameters
sr = 22050 # if sampling rate is different, resample it to this

# parameters for calculating spectrogram in mel scale
fmax = 10000 # maximum frequency considered
fft_window_points = 512
fft_window_dur = fft_window_points * 1.0 / sr # 23ms windows
hop_size = int(fft_window_points/ 2) # 50% overlap between consecutive frames
n_mels = 64

# segment duration
num_fft_windows = 256 # num fft windows per music segment
segment_in_points = num_fft_windows * 255 # number of data points that insure the spectrogram has size: 64 * 256
segment_dur = segment_in_points * 1.0 / sr

num_genres=10
input_shape=(64, 256, 1)


def split_data(T, split_idxes):
    """
    give the indexes of training, validation, and testing data
    :param T: label of all data
    :param split_idxes: splitting points of the data
    :return:
    """
    genres = np.unique(T)
    training_idxes = []
    validation_idxes = []
    testing_idxes = []
    for idx, music_genre in enumerate(genres):
        tmp_logidx = music_genre == T
        tmp_idx = np.flatnonzero(tmp_logidx)
        tmp_shuffled_idx = np.random.permutation(tmp_idx)
        tmp_num_examles = len(tmp_shuffled_idx)
        tmp_split_idxes = np.asarray(split_idxes * tmp_num_examles, dtype=np.int)
        training_idxes.append(tmp_shuffled_idx[tmp_split_idxes[0] : tmp_split_idxes[1]])
        validation_idxes.append(tmp_shuffled_idx[tmp_split_idxes[1] : tmp_split_idxes[2]])
        testing_idxes.append(tmp_shuffled_idx[tmp_split_idxes[2] : tmp_split_idxes[3]])
    return(np.concatenate(training_idxes), np.concatenate(validation_idxes), np.concatenate(testing_idxes))


def load_original_data():
    """
    load original audio files
    :return:
    """
    import os
    data_folder = "/home/md/Dropbox/Courses/2017_Spring_Machine_learning/projects/music_gen/genres"
    # genre_folders = [x[0] for x in os.walk(data_folder)]
    genre_folders = os.listdir(data_folder)
    X = []
    T = []
    SR = []
    min_length = 0
    for sub_folder in genre_folders:
        genre_path = data_folder + "/" + sub_folder
        print(genre_path)
        audio_files = os.listdir(genre_path)
        for audio_name in audio_files:
            audio_path = genre_path + "/" + audio_name
            x, sr = librosa.core.load(audio_path)
            if x.shape[0] < 30 * sr:
                x = np.append(x, np.zeros(30*sr - x.shape[0])) # insure all files are exactly the same length
                if min_length < x.shape[0]:
                    min_length = x.shape[0] # report the duration of the minimum audio clip
                    print("This audio last %f seconds, zeros are padded at the end." % (x.shape[0]*1.0/sr))
            X.append(x[:30*sr])
            SR.append(sr)
            T.append(sub_folder)
    return np.asarray(X), np.asarray(SR), np.asarray(T, dtype=str)

# calculate mel-spectrogram
def mel_spectrogram(ys, sr, n_mels=n_mels, hop_size=hop_size, fmax=fmax, pre_emphasis=False):
    """
    calculate the spectrogram in mel scale, refer to documentation of libriso and MFCC tutorial
    :param ys:
    :param sr:
    :param n_mels:
    :param hop_size:
    :param fmax:
    :param pre_emphasis:
    :return:
    """
    if pre_emphasis:
        ys = np.append(ys[0], ys[1:]-pre_emphasis*ys[:-1])
    return librosa.feature.melspectrogram(ys, sr,
                                          n_fft=fft_window_points,
                                          hop_length=hop_size, n_mels=n_mels,
                                          fmax=fmax)

# batch convert waveform into spectrogram in mel-scale
def batch_mel_spectrogram(X, SR):
    """
    convert all waveforms in R into time * 64 spectrogram in mel scale
    :param X:
    :param SR:
    :return:
    """
    melspec_list = []
    for idx in xrange(X.shape[0]):
        tmp_melspec = mel_spectrogram(X[idx], SR[idx])
        melspec_list.append(tmp_melspec)
    return np.asarray(melspec_list)


# def segment_spectrogram(input_spectrogram, num_fft_windows=num_fft_windows):
#     # given a spectrogram of a music that's longer than 3 seconds, segment it into relatively independent pieces
#     length_in_fft = input_spectrogram.shape[1]
#     num_segments = int(length_in_fft / num_fft_windows)
#     pass


def baseline_model_32(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)

def baseline_model_64(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)

def baseline_model_96(num_genres=num_genres, input_shape=input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(3, 3),
                     activation='relu', kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Conv2D(64, (3, 5), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02)))
    model.add(Dropout(0.2))
    model.add(Dense(num_genres, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(decay=1e-5),
                  metrics=['accuracy'])
    return(model)


class Music_Genre_CNN(object):

    def __init__(self, ann_model):
        self.model = ann_model()

    def load_model(self, model_path, custom_objects=None):
        self.model = load_model(model_path, custom_objects=custom_objects)

    def train_model(self, input_spectrograms, labels, cv=False,
                    validation_spectrograms=None, validation_labels=None,
                    small_batch_size=150, max_iteration=500, print_interval=1):
        """
        train the CNN model
        :param input_spectrograms: number of training examplex * num of mel bands * number of fft windows * 1
            type: 4D numpy array
        :param labels: vectorized class labels
            type:
        :param cv: whether do cross validation
        :param validation_spectrograms: data used for cross validation
            type: as input_spectrogram
        :param validation_labels: used for cross validation
        :param small_batch_size: size of each training batch
        :param max_iteration:
            maximum number of iterations allowed for one training
        :return:
            trained model
        """
        validation_accuracy_list = []
        for iii in xrange(max_iteration):

            st_time = time.time()

            # split training data into even batches
            num_training_data = len(input_spectrograms)
            batch_idx = np.random.permutation(num_training_data)
            num_batches = int(num_training_data / small_batch_size)

            for jjj in xrange(num_batches - 1):
                sample_idx = np.random.randint(input_spectrograms.shape[2] - num_fft_windows)
                training_idx = batch_idx[jjj * small_batch_size: (jjj + 1) * small_batch_size]
                training_data = input_spectrograms[training_idx, :, sample_idx:sample_idx+num_fft_windows, :]
                training_label = labels[training_idx]
                self.model.train_on_batch(training_data, training_label)
                training_accuracy = self.model.evaluate(training_data, training_label)
                # print("Training accuracy is: %f" % (training_accuracy))

            end_time = time.time()
            elapsed_time = end_time - st_time
            if cv:
                validation_accuracy = self.model.evaluate(validation_spectrograms[:, :, sample_idx:sample_idx+num_fft_windows, :], validation_labels)
                validation_accuracy_list.append(validation_accuracy[1])
            else:
                validation_accuracy = [-1.0, -1.0]

            if iii % print_interval == 0:
                print("\nTime elapsed: %f; Training accuracy: %f, Validation accuracy: %f\n" %
                      (elapsed_time, training_accuracy[1], validation_accuracy[1]))
        if cv:
            return np.asarray(validation_accuracy_list)


    def song_spectrogram_prediction(self, song_mel_spectrogram, overlap):
        """
        give the predicted_probability for each class and each segment
        :param song_mel_spectrogram:
            4D numpy array: num of time windows * mel bands * 1 (depth)
        :param overlap:
            overlap between segments, overlap = 0 means no overlap between segments
        :return:
            predictions: numpy array (number of segments * num classes)
        """
        # 1st segment spectrogram into sizes of 64 * 256
        largest_idx = song_mel_spectrogram.shape[1] - num_fft_windows - 1
        step_size = int((1 - overlap) * num_fft_windows)
        num_segments = int(largest_idx / step_size)
        segment_edges = np.arange(num_segments) * step_size
        segment_list = []
        for idx in segment_edges:
            segment = song_mel_spectrogram[:, idx : idx + num_fft_windows]
            segment_list.append(segment)
        segment_array = np.asarray(segment_list)[:, :, :, np.newaxis]
        predictions = self.model.predict_proba(segment_array, batch_size=len(segment_array), verbose=0)
        summarized_prediction = np.argmax(predictions.sum(axis=0))
        return(summarized_prediction, predictions)

    def test_model(self, test_X, test_T, overlap=0.5):
        # test the accuracy of the model using testing data
        num_sample = len(test_T)
        correct_labels = np.argmax(test_T, axis=1)
        predicted_labels = np.zeros(num_sample)
        for iii in xrange(len(test_X)):
            song_mel_spectrogram = test_X[iii].squeeze()
            predicted_labels[iii], _ = self.song_spectrogram_prediction(song_mel_spectrogram, overlap=overlap)
            # correct_labels[iii] = np.argmax(test_T[iii])
        confusion_data = np.vstack((predicted_labels, correct_labels)).T
        accuracy = np.sum(correct_labels == predicted_labels) * 1.0 / num_sample
        return(accuracy, confusion_data)

    def backup_model(self, model_bk_name=False):
        if not model_bk_name:
            year, month, day, hour, minute = time.strftime("%Y,%m,%d,%H,%M").split(',')
            model_bk_name = "mgcnn_" + month + day + hour + minute + ".h5"
        self.model.save(model_bk_name)

    def song_genre_prediction(self, song_waveform):
        # resample the song into single channel, 22050 sampling frequency

        # convert into mel-scale spectrogram

        # predict using trained model

        #

        pass


