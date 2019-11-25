import csv
from random import shuffle
from scipy import ndimage
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D


data_paths = ['./origin_data/driving_log.csv']

samples = []
for path in data_paths:
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    print(len(samples))

samples.pop(0)

def image_process(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#     img = img[70:135, :]
#     img = cv2.resize(img, (200, 66))
    return img


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                file_name = './origin_data/IMG/' + batch_sample[0].split('/')[-1]
                # file_name = batch_sample[0]
                center_image = ndimage.imread(file_name)
                # center_image = image_process(center_image)  # image process
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            # train image to only see section with toad
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# Set out batch size
batch_size = 128
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model = Sequential()
# Preprocess incoming origin_data, centered around zero with small standard deviation
# model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66, 200, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                    steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                    validation_data=validation_generator,
                    validation_steps=np.ceil(len(validation_samples)/batch_size),
                    epochs=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
