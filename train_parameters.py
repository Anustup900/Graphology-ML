# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:24:53 2020

@author: Anustup
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Depression")
  elif answer == 1:
    print("Label: Nodepression")
  

  return answer

daisy_t = 0
daisy_f = 0
rose_t = 0
rose_f = 0
sunflower_t = 0
sunflower_f = 0

for i, ret in enumerate(os.walk('./test-data/depression')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Daisy")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      daisy_t += 1
    else:
      daisy_f += 1

for i, ret in enumerate(os.walk('./test-data/nodepression')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Rose")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      rose_t += 1
    else:
      rose_f += 1




"""
Check metrics
"""
print("True Depression: ", daisy_t)
print("False Depression: ", daisy_f)
print("True nodepression: ", rose_t)
print("False nodepression: ", rose_f)
