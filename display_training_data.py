import cv2
import numpy as np

file_name = "train_data.npy"
training_data = list(np.load(file_name))

for data in training_data:
    image = data[0]
    output = data[1]

    image = cv2.resize(image, (150, 120))
    cv2.imshow('img', image)
    print(output)
    cv2.waitKey()
