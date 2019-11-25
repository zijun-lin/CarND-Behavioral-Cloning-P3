import csv
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, Dropout

data_paths = ['./data/driving_log.csv']  # workspace data
# data_paths = ['./data/driving_log.csv', './data/my_driving_log.csv']  # workspace data and my training data

samples = []
for path in data_paths:
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    print(len(samples))

samples.pop(0)  # remove the title line


def image_process(img):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # img = img[70:135, :]
    # img = cv2.resize(img, (200, 66))
    return img


def generator(samples, batch_size=32, is_use_all_images=False, correction=0.2):
    num_samples = len(samples)
    while True:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # the center camera image
                file_name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = ndimage.imread(file_name)
                center_image = image_process(center_image)  # image process
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # Filpping the image horizontally (left right)
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)

                if is_use_all_images:
                    # the left and right camera images
                    left_file_name = './data/IMG/' + batch_sample[1].split('/')[-1]
                    right_file_name = './data/IMG/' + batch_sample[2].split('/')[-1]
                    left_image = ndimage.imread(left_file_name)
                    right_image = ndimage.imread(right_file_name)
                    left_image = image_process(left_image)  # image process
                    right_image = image_process(right_image)  # image process
                    left_angle = center_angle + correction
                    right_angle = center_angle - correction
                    images.append(left_image)
                    angles.append(left_angle)
                    images.append(right_image)
                    angles.append(right_angle)
                    # Filpping the image horizontally
                    images.append(cv2.flip(left_image, 1))
                    angles.append(left_angle * -1.0)
                    images.append(cv2.flip(right_image, 1))
                    angles.append(right_angle * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


# Randomly shuffle the origin_data
samples = shuffle(samples)
# split the samples into training and validation sets.
train_samples, validation_samples = train_test_split(samples, test_size=0.2)


# Set out batch size
batch_size = 128
# Is use all images for training the model
is_use_all_images = False
# create adjusted steering measurements for the side camera images
correction = 0.2

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size, is_use_all_images, correction)
validation_generator = generator(validation_samples, batch_size, is_use_all_images, correction)

# Image shape
# row, col, ch = 66, 200, 3
row, col, ch = 160, 320, 3
model = Sequential()
# Preprocess incoming origin_data, centered around zero with small standard deviation
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(Flatten())
# model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     steps_per_epoch=np.ceil(len(train_samples) / batch_size),
                                     validation_data=validation_generator,
                                     validation_steps=np.ceil(len(validation_samples)/batch_size),
                                     epochs=5, verbose=1)

model.save('model.h5')

# print the keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
