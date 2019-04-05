from Model.neural_network import NeuralNetwork
import os
import numpy as np
import pandas as pd

validation_rate = 0.1

def getValidationDataAndLabels(images, labels):
    global validation_rate
    validation_length = int(len(images) * validation_rate)
    return images[-validation_length:], labels[-validation_length:]

def getTrainingDataAndLabels(images, labels):
    global validation_rate
    training_length = len(images) - int(len(images) * validation_rate)
    return images[:training_length], labels[:training_length]


training_file_name = "train_data.npy"
if not os.path.isfile(training_file_name):
    print("le fichier n'existe pas !!!")
    exit(1)

data = list(np.load(training_file_name))
data_frame = pd.DataFrame(data, columns=('image', 'action'))
images = data_frame['image'].values
labels = data_frame['action'].values

validation_data, validation_labels = getValidationDataAndLabels(images, labels)
training_data, training_labels = getTrainingDataAndLabels(images, labels)

neural_network = NeuralNetwork()
neural_network.create_neural_network(input_shape=(80, 60, 1), number_outputs=5)
neural_network.fit_training_data(training_data, training_labels, batch_size=30, epochs=100,
                                 is_shuffled=True, validation_split_rate=0.2)
neural_network.evaluate_model(validation_data, validation_labels)
neural_network.save_model()
