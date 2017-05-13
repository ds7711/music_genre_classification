# import modules
import numpy as np
import librosa
import music_gen_lib as mgl
from keras.utils import np_utils
import time
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import keras
from sklearn.metrics import classification_report, confusion_matrix

def main(random_seed=None, visualize_label=False, ann_model=mgl.baseline_model_96):


    # determine the random seed so that results are reproducible
    # random_seed = 11 # also determines which shuffled index to use
    np.random.seed(random_seed)


    # 1st part: load all the data and organize data
    data_converted = True
    if data_converted == False:
        # load all the data
        X, SR, T = mgl.load_original_data()
        # data format:
        #       x: 1d numpy array
        #       t: 1d numpy array with numsic genre names (numeric arrays or multinomial vector?)

        # convert the data into mel-scale spectrogram
        st = time.time()
        newX = mgl.batch_mel_spectrogram(X, SR)
        print(time.time() - st)

        # save the data into npz
        np.savez_compressed("audio_sr_label.npz", X=newX, SR=SR, T=T)

    else:
        st = time.time()
        data = np.load("audio_sr_label.npz")
        X = data["X"]
        SR = data["SR"]
        T = data["T"]
        loading_time = time.time() - st
        print("Loading takes %f seconds." % (loading_time))


    # Use log transformation to preserve the order but shrink the range
    X = np.log(X + 1)
    X = X[:, :, :, np.newaxis]  # image channel should be the last dimension, check by using print K.image_data_format()



    # convert string type labels to vectors
    genres = np.unique(T)
    genres_dict = dict([[label, value] for value, label in enumerate(genres)])
    T_numeric = np.asarray([genres_dict[label] for label in T])
    T_vectorized = np_utils.to_categorical(T_numeric)


    # split data into training, cross-validation,  testing data
    # following is used to generate random see used to split the data into different sets
    # split_idxes = np.asarray([0, 0.5, 0.7, 1])
    # training_idxes_list, validation_idxes_list, testing_idxes_list = [], [], []
    # for idx in xrange(30):
    #     training_idxes, validation_idxes, testing_idxes = mgl.split_data(T, split_idxes)
    #     training_idxes_list.append(training_idxes)
    #     validation_idxes_list.append(validation_idxes)
    #     testing_idxes_list.append(testing_idxes)
    #
    # training_idxes_list = np.asarray(training_idxes_list)
    # validation_idxes_list = np.asarray(validation_idxes_list)
    # testing_idxes_list = np.asarray(testing_idxes_list)
    #
    # np.savez_compressed("shuffled_idx_list.npz", training_idxes_list=training_idxes_list,
    #                     validation_idxes_list=validation_idxes_list, testing_idxes_list=testing_idxes_list)


    ## load one fixed data shuffling indexes
    idxes_list = np.load("shuffled_idx_list.npz")
    training_idxes = idxes_list["training_idxes_list"][random_seed]
    validation_idxes = idxes_list["validation_idxes_list"][random_seed]
    testing_idxes = idxes_list["testing_idxes_list"][random_seed]


    # shuffled_idx = np.random.permutation(num_total_data) # shuffle or not
    # shuffled_idx_list = np.asarray([np.random.permutation(num_total_data) for x in xrange(30)])
    # np.savez_compressed("shuffled_idx_list.npz", shuffled_idx_list=shuffled_idx_list)


    training_X = X[training_idxes]
    validation_X = X[validation_idxes]
    testing_X = X[testing_idxes]

    training_T = T_vectorized[training_idxes]
    validation_T = T_vectorized[validation_idxes]
    testing_T = T_vectorized[testing_idxes]
    # testing_T_label = T[testing_idxes]


    # try to load pre-trained model
    saved_model_name = "mgcnn_rs_" + str(random_seed) + ".h5"
    # saved_model_name = "mgcnn_poisson_rs_" + str(random_seed) + ".h5"
    MGCNN = mgl.Music_Genre_CNN(ann_model)
    try:
        # MGCNN.load_model(saved_model_name, custom_objects={'PoissonLayer': ann_model})
        MGCNN.load_model(saved_model_name)
    except:
        print("The model hasn't been trained before.")
        # training the model
        training_flag = True
        max_iterations = 10
        while training_flag and max_iterations >= 0:
            validation_accuracies = MGCNN.train_model(training_X, training_T, cv=True,
                                                      validation_spectrograms=validation_X,
                                                      validation_labels=validation_T)

            diff = np.mean(validation_accuracies[-10:]) - np.mean(validation_accuracies[:10])
            MGCNN.backup_model()  # backup in case error occurred
            if np.abs(diff) < 0.01:
                training_flag = False
            max_iterations -= 1

        MGCNN.backup_model(saved_model_name)




    test_accuracy, confusion_data = MGCNN.test_model(testing_X, testing_T)
    print("\n ****** The final test accuracy is %f. ******\n" % (test_accuracy))

    with open("model_accuracy_log.txt", "a") as text_file:
        things2write = saved_model_name + "\t" + "accuracy: " + str(test_accuracy) + "\n"
        text_file.write(things2write)

    # analyze the confusion matrix
    cs = classification_summary = classification_report(confusion_data[:, 1], confusion_data[:, 0],
                                                        labels=genres_dict.values(), target_names=genres_dict.keys())
    cm = confusion_matrix(confusion_data[:, 1], confusion_data[:, 0]) / (len(testing_T) * 1.0 / len(genres))

    # visualize
    if visualize_label:
        import matplotlib.pylab as plt
        plt.matshow(cm); plt.colorbar()

    return cs, cm

main(random_seed=0, visualize_label=False, ann_model=mgl.baseline_model_64)