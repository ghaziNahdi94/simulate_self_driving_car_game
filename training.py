from Model.neural_network import NeuralNetwork
import os
import numpy as np

training_file_name = "train_data.npy"
if not os.path.isfile(training_file_name):
    print("le fichier n'existe pas !!!")
    exit(1)

data = list(np.load(training_file_name))
neural_network = NeuralNetwork()
