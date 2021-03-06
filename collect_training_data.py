import pyscreenshot
import cv2
import numpy as np
import time
import pyxhook
import os
from config import *

training_data = []
key_combination = []

def init_data():
    global training_data
    if os.path.isfile(train_data_filename):
        training_data = list(np.load(train_data_filename))
    else:
        training_data = []

def wait_seconds_before_start(seconds=5):
    for i in range(seconds):
        print(seconds - i)
        time.sleep(1)

def on_key_press(event):
    global key_combination
    if len(key_combination) < 2:
        if event.Key not in key_combination:
            key_combination.append(event.Key)

def on_key_release(event):
    global key_combination
    try:
        key_combination.remove(event.Key)
    except:
        pass

def get_outpu_from_combination_keys():
    #output : [left,up,right,(up+left),(up+right)]
    output = [0,0,0,0,0]
    if ('Up' in key_combination) and ('Left' in key_combination):
        output = [0,0,0,1,0]
    elif ('Up' in key_combination) and ('Right' in key_combination):
        output = [0,0,0,0,1]
    elif 'Up' in key_combination:
        output = [0,1,0,0,0]
    elif 'Left' in key_combination:
        output = [1,0,0,0,0]
    elif 'Right' in key_combination:
        output = [0,0,1,0,0]
    else:
        output = [0, 0, 0, 0, 0]
    return output

def start_detecting_keys():
    hook_manager = pyxhook.HookManager()
    hook_manager.HookKeyboard()
    hook_manager.KeyDown = on_key_press
    hook_manager.KeyUp = on_key_release
    hook_manager.start()

def start_capture_frames():
    global training_data
    global key_pressed
    last_time = time.time()
    while True:
        image = pyscreenshot.grab((screen_grab_x1, screen_grab_y1, screen_grab_x2, screen_grab_y2))
        screen_image = cv2.resize(src=np.array(image), dsize=(120, 100))
        output = get_outpu_from_combination_keys()
        training_data.append([screen_image, output])
        key_pressed = 'Up'
        print('FPS : {}'.format(time.time() - last_time))
        last_time = time.time()
        if len(training_data) % 50 == 0:
            print(len(training_data))
            np.save(train_data_filename, training_data)

init_data()
wait_seconds_before_start(seconds=5)
start_detecting_keys()
start_capture_frames()