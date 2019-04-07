import cv2
import numpy as np
from config import train_data_filename

training_data = list(np.load(train_data_filename))

for data in training_data:
    image = data[0]
    output = data[1]
    cv2.imshow('img', image)
    print(output)
    cv2.waitKey()
