# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K 

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

img_width, img_height = 200, 200
  
if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height)
    chanDim = 1   
else: 
    input_shape = (img_width, img_height, 3) 
    chanDim = -1
  
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))


print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  
#train_datagen = ImageDataGenerator(rescale = 1. / 255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True) 
train_datagen = ImageDataGenerator(rescale = 1. / 255) 
test_datagen = ImageDataGenerator(rescale = 1. / 255) 
  
train_generator = train_datagen.flow_from_directory(
        '/home/xueguang/coding/data/dataset/train',
        batch_size=32,
        target_size=(img_width, img_height),
        classes = ('yes','no'))

validation_generator = test_datagen.flow_from_directory(
        '/home/xueguang/coding/data/dataset/validation',
        batch_size=32,
        target_size=(img_width, img_height),
        classes = ('yes','no'))

print(".....................................")
print(train_generator.class_indices)
print(validation_generator.class_indices)

model.fit_generator(
        train_generator,
        steps_per_epoch=2200,
        epochs=10,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=800)
  
model.save_weights('model_saved.h5') 
model.save('model.h5')
