# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 19:56:36 2021

@author: j2609
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

model_dir = r"D:\Code\Face Keypoints\face_keypoints_model"
train_dir = os.path.join(r"D:\Kaggle\facial-keypoints-detection", r'training.csv')
model = tf.keras.models.load_model(model_dir)
model.summary()
train = pd.read_csv(train_dir)

def train_image(train):
    img = []
    for i in range(7049):
        image = train.loc[i, 'Image'].split(' ')
        image = [x for x in image]
        img.append(image)
    return img

def tran_list2np(img_list):
    img_data = np.array(img_list, dtype = 'float')
    img_data = img_data.reshape(-1,96,96,1)
    return img_data

image = train_image(train)
image_data = tran_list2np(image)
image_data /= 255


result_dir = os.path.join(r"D:\Kaggle\facial-keypoints-detection", r'Submission.csv')
result = pd.read_csv(result_dir)
for i in range(10):
  x = result.query(f"ImageId == {i+1}")[result.query(f'ImageId == {i+1}').FeatureName.str.match('(.*_x)')].Location
  y = result.query(f"ImageId == {i+1}")[result.query(f'ImageId == {i+1}').FeatureName.str.match('(.*_y)')].Location
  
x = list(x)
y = list(y)

x1, y1 = [x[0], x[1]], [y[0], y[1]]
plt.figure()
plt.imshow(image_data[0], cmap = 'gray')
plt.plot(x1, y1, marker = 'o')
plt.show()
