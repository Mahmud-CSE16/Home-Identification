# Here's a codeblock just for fun. You should be able to upload an image here 
# and have it classified without crashing

import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf 
#from keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model('covid19.model')
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

# predicting images
path = 'NORMAL2-IM-1179-0001.jpeg'
img = image.load_img(path, target_size=(224,224))
# YOUR CODE HERE))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=8)
print(classes[0])

x=classes[0]

#result checking
# if x[0]>0.5:
#     print(" is a covid")
# else:
#     print(" is a normal")