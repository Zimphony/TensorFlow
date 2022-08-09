import tensorflow as tf
import urllib.request
import zipfile
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.utils import load_img
import cv2

weight_url = "http://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weight_file = "inception_v3.h5"
urllib.request.urlretrieve(weight_url,weight_file)

pre_trained_model = InceptionV3(input_shape = (150,150,3),
	include_top = False,
	weights = None)

pre_trained_model.load_weights(weight_file)

## print(pre_trained_model.summary())  ## to see the archetecture

for layer in pre_trained_model.layers:
	layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output
print('last layer output shape:', last_layer.output_shape)

# Flatten the output layer to 1 simension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1-24 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
# Add a final sigmoil layer for classification 
x = layers.Dense(1, activation ='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer = RMSprop(learning_rate = 0.0001),
	loss = 'binary_crossentropy',
	metrics = ['acc'])

# Extract zip train and validation

train_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
training_file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'
urllib.request.urlretrieve(train_url,training_file_name)
zip_ref = zipfile.ZipFile(training_file_name,'r')
zip_ref.extractall(training_dir)
zip_ref.close()

validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_file_name = "validation-horse-or-human.zip"
validation_dir = "horse-or-human/Validation/"
urllib.request.urlretrieve(validation_url,validation_file_name)
zip_ref = zipfile.ZipFile(validation_file_name,'r')
zip_ref.extractall(validation_dir)
zip_ref.close()

# Add data-augmentation parameters to ImageDataGenerator
training_datagen = ImageDataGenerator(rescale = 1./255,
										rotation_range = 40,
										width_shift_range = 0.2,
										height_shift_range = 0.2,
										shear_range = 0.2,
										zoom_range=0.2,
										horizontal_flip = True)
# Note that validation data should not be augmented
test_datagen = ImageDataGenerator(rescale = 1./255)

# Flow training images in batches of 20 using train_datagen
train_generator = test_datagen.flow_from_directory(training_dir,
													batch_size = 20,
													class_mode = 'binary',
													target_size = (150,150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(validation_dir,
														batch_size = 20,
														class_mode = 'binary',
														target_size = (150,150))


history = model.fit_generator(
			train_generator,
			validation_data = validation_generator,
			epochs = 20,
			verbose = 1) # verbose = 1 : show bar progress, verbose = 0 : nothing show

path = "D:/One Drive/OneDrive/รูปภาพ/Work pic/python/horses-g2b56624c0_1920.jpg"
pic_name = 'horses-g2b56624c0_1920.jpg'

img = tf.keras.preprocessing.image.load_img(path, target_size = (150,150))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis = 0)
	
image_tensor = np.vstack([x])
classes = model.predict(x)
print(classes)
print(classes[0])
if classes[0] > 0.5:
	print(pic_name + "is human")
else:
	print(pic_name +'is  a horse')