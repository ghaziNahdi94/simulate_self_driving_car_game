from Model.neural_network import NeuralNetwork
import os
import numpy as np
import pandas as pd
from config import *

validation_rate = 0.1


def getValidationDataAndLabels(images, labels):
    global validation_rate
    validation_length = int(len(images) * validation_rate)
    return np.array(images[-validation_length:]), np.array(labels[-validation_length:])


def getTrainingDataAndLabels(images, labels):
    global validation_rate
    training_length = len(images) - int(len(images) * validation_rate)
    return np.array(images[:training_length]), np.array(labels[:training_length])


if not os.path.isfile(train_data_filename):
    print("le fichier n'existe pas !!!")
    exit(1)

data = list(np.load(train_data_filename))
data_frame = pd.DataFrame(data, columns=('image', 'action'))
images = data_frame['image'].values.tolist()
labels = data_frame['action'].values.tolist()

test_data, test_labels = getValidationDataAndLabels(images, labels)
training_data, training_labels = getTrainingDataAndLabels(images, labels)

neural_network = NeuralNetwork()
neural_network.create_neural_network(input_shape=(image_height, image_width, image_channel),
                                     number_outputs=output_nb_actions)
neural_network.fit_training_data(training_data, training_labels, batch_size=30, epochs=10,
                                 is_shuffled=True, validation_split_rate=0.2)
neural_network.evaluate_model(test_data, test_labels)
neural_network.save_model()
