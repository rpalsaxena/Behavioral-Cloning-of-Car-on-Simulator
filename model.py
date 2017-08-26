import csv
import cv2
import numpy as np
import matplotlib.image as mpimg


from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from random import shuffle
from keras.callbacks import EarlyStopping, ModelCheckpoint


samples = []
with open('driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for row in batch_samples:
                steering_center = float(row[3])
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                angles.append(steering_center)
                angles.append(steering_center * -1.0)
                angles.append(steering_left)
                angles.append(steering_left * -1.0)
                angles.append(steering_right)
                angles.append(steering_right * -1.0)
                
                images.append(cv2.imread(row[0]))
                images.append(cv2.flip(mpimg.imread(row[0]), 1))
                images.append(mpimg.imread(row[1]))
                images.append(cv2.flip(mpimg.imread(row[1]), 1))
                images.append(mpimg.imread(row[2]))
                images.append(cv2.flip(mpimg.imread(row[2]), 1))
                
                
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)


model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(88, 3, 3, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(320))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss= 'mse', optimizer='adam')

batch_size = 64

samples_per_epoch =(len(train_samples)//batch_size)*batch_size

callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint('model.h5',
            monitor='val_loss', save_best_only = True, verbose = 0)]
							 
model.fit_generator(train_generator, samples_per_epoch = samples_per_epoch, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=3, verbose=1, callbacks=callbacks)		