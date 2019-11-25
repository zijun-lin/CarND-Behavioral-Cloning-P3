# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/p42.svg "Model Visualization"
[image2]: ./examples/Nvidia.png "Grayscaling"
[image3]: ./examples/center.jpg "Grayscaling"
[image4]: ./examples/side.gif "Recovery Image"
[image5]: ./examples/original_flipped.jpeg "Recovery Image"
[image6]: ./examples/loss.jpeg "Recovery Image"
[image7]: ./examples/original_flipped.jpeg "Normal Image"
[image8]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video01.mp4 and video02.mp4 recording the vehicel in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use the Nvidia's model convolutional neural network as the self-driving cars model and make some changes to the model. The model consists of a normalization layer and 5 convolutional layers and 4 fully connected layers. The architecture of the model is show in below:

```python
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
```

| Layer         |     Description               |
|:-------------:|------------------------------:|
| Input         | 160x320x3 RGB image           |
| Lambda        | Image normalization           |
| Cropping2D    | (50, 20), (0, 0)              |
| Conv2D        | 24 filters, 5x5 kernel size, 2x2 stride, `RELU` activetion|
| Conv2D        | 36 filters, 5x5 kernel size, 2x2 stride, `RELU` activetion|
| Conv2D        | 48 filters, 5x5 kernel size, 2x2 stride, `RELU` activetion|
| Conv2D        | 64 filters, 3x3 kernel size, 1x1 stride, `RELU` activetion|
| Conv2D        | 64 filters, 3x3 kernel size, 1x1 stride, `RELU` activetion|
| Flatten       |                               |
| Dense         | 100, `RELU` activetion        |
| Dense         | 50,  `RELU` activetion        |
| Dense         | 10,  `RELU` activetion        |
| Dense         | 1                             |

The first three convolution layers with 5x5 filter sizes and stride 2, the rest two convolutionlayers with 3x3 filter sizes and stride 3. After the five convolution layers is four full connected layers with the 100, 50, 10 and 1 neurons. The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
For Training the model, I drive the the car counter-clockwise in two laps to create my dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to reduce the loss of training dataset and validation dataset.

My first step was to use a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because this model make great success in MNIST handwritten digit database and trafic sign classifier.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model and some dropout layers after every fully connected layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I drive the the car counter-clockwise in two laps to create my dataset.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of 5 convolution neural network with the following 4 full connected layers.
Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]
![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to what to do when it’s off on the side of the road. These images show what a recovery looks like starting from right side:

![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the dataset, I also flipped images and angles thinking that this would augment the data quickly, For example, here is an image that has then been flipped:

![alt text][image5]

After the collection process, I had `11794` number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by training loss and validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image6]