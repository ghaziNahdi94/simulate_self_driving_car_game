from Model.neural_network import NeuralNetwork
import os
import numpy as np
import pandas as pd

validation_rate = 0.1

def getValidationDataAndLabels(images, labels):
    global validation_rate
    validation_length = int(len(images) * (validation_rate*100) / 100)
    print(validation_length)
    return images[-validation_length:], labels[-validation_length:]

def getTrainingDataAndLabels(images, labels):
    global validation_rate
    training_length = len(images) - int(len(images) * (validation_rate*100) / 100)
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

print('total length : {}'.format(len(images)))
print('training length : {}'.format(len(training_data)))
print('validation length : {}'.format(len(validation_data)))
neural_network = NeuralNetwork()
