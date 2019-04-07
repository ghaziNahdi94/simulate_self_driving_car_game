from pynput.keyboard import Key, Controller
from Model.neural_network import NeuralNetwork
import pyscreenshot
import numpy as np
from config import *
import cv2
import time

keyboard = Controller()
neural_network = NeuralNetwork()
neural_network.restore_model()

def press_key(key):
    global keyboard
    keyboard.press(key)
    # keyboard.release(key)

def press_2_keys(key1, key2):
    global keyboard
    keyboard.press(key1)
    keyboard.press(key2)
    # keyboard.release(key1)
    # keyboard.release(key2)

def make_decision(action):
    global keyboard
    index = np.argmax(action)
    if output_action_labels[index] == "Up+Left":
        press_2_keys(Key.up, Key.left)
    elif output_action_labels[index] == "Up+Right":
        press_2_keys(Key.up, Key.right)
    elif output_action_labels[index] == "Up":
        press_key(Key.up)
    elif output_action_labels[index] == "Left":
        press_key(Key.left)
    elif output_action_labels[index] == "Right":
        press_key(Key.right)

def wait_seconds_before_start(seconds=5):
    for i in range(seconds):
        print(seconds - i)
        time.sleep(1)

wait_seconds_before_start(seconds=5)
while True:
    image = pyscreenshot.grab((screen_grab_x1, screen_grab_y1, screen_grab_x2, screen_grab_y2))
    image = np.array(image)
    screen_image = cv2.resize(src=image, dsize=(image_width, image_height))
    action = neural_network.predict_action(screen_image)
    print(output_action_labels[np.argmax(action)])
    make_decision(action)
