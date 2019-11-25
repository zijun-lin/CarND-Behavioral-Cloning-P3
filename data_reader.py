from __future__ import print_function
import numpy as np
import cv2
import sklearn
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Data(object):
    def __init__(self, batch_size):
        self.samples = self.read_csv(['origin_data/driving_log.csv', 'my-origin_data/driving_log.csv','my-origin_data-2/driving_log.csv',
                                      'my-origin_data-3/driving_log.csv', 'my-origin_data-4/driving_log.csv'])
        self.train, self.val = self.split_dataset(self.samples)
        self.train_generator = self.generator(self.train, is_training=True, batch_size=batch_size)
        self.validation_generator = self.generator(self.val, is_training=False, batch_size=batch_size)

    def read_csv(self, paths):
        samples = []
        for item in paths:
            with open(item, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    samples.append(line)
        return samples

    def split_dataset(self, samples, test_size=0.1):
        samples = shuffle(samples)
        train_samples, validation_samples = train_test_split(samples, test_size=test_size)
        return train_samples, validation_samples

    def data_augmentation(self, img):
        """
        method for adding random distortion to dataset images, including random brightness adjust, and a random
        vertical shift of the horizon position
        """
        new_img = img.astype(float)
        # random brightness - the mask bit keeps values from going beyond (0,255)
        value = np.random.randint(-28, 28)
        if value > 0:
            mask = (new_img[:, :, 0] + value) > 255
        if value <= 0:
            mask = (new_img[:, :, 0] + value) < 0
        new_img[:, :, 0] += np.where(mask, 0, value)
        # random shadow - full height, random left/right side, random darkening
        h, w = new_img.shape[0:2]
        mid = np.random.randint(0, w)
        factor = np.random.uniform(0.6, 0.8)
        if np.random.rand() > .5:
            new_img[:, 0:mid, 0] *= factor
        else:
            new_img[:, mid:w, 0] *= factor
        return (new_img.astype(np.uint8))


    def preprocessing(self, img):
        # # change the color space to YUV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        # crop the ROI
        img = img[50:140, :]
        # resize the image to (66, 200, 3) since we use the Nvidia Net
        img = cv2.resize(img, (200, 66))
        return img

    def generator(self, samples, is_training, batch_size=64):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:
                    file_name = batch_sample[0]

                    center_image = cv2.imread(file_name)
                    center_image = self.preprocessing(center_image)
                    center_angle = float(batch_sample[3])

                    # the origin_data augmentation is only performed during training
                    if is_training:
                        prob = np.random.rand()
                        if prob < 0.5:
                            center_image = self.data_augmentation(center_image)
                    images.append(center_image)
                    angles.append(center_angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)


# origin_data = Data(128)
#
# for i in range(3):
#     imgs, labels = origin_data.train_generator.next()
#     print(imgs.shape)
