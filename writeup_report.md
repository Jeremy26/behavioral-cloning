# **Welcome to Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/random_shadows.png "Random Shadows"
[image2]: ./examples/mse_loss.png "MSE Loss"
[image3]: ./examples/architecture.png "Architecture"


#### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

##### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

##### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

##### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### 1. Data Collect

I did split my data into training and validation. The test data is the autonomous mode in the simulator.

In order to get good data, I did the following :
* A. Data collection
* B. Use of three cameras with a correction 
* C. Data augmentation with flipping 
* D. Data augmentation with augment_brightness function
* E. Data augmentation with random shadows

##### A. Data Collection
I did a first collect of the training Data by driving on the center for 3 laps. I then added data by driving one lap counter-clockwise and one by driving on the border of the road. I did few records of driving on the curves also to help the model learning on how to take a curve.

I end up with a collection of 10 043 samples of data, which is a good starting point.

##### B. Use of three cameras with a correction
On track 1, the biggest difficuty was to drive on the curves and not get out of the road. I did drive on curves but it wasn't enough. I decided to follow David Silver's tutorial to use the three cameras and not only the center camera.
```
 correction = 0.22
images = []
            measurements = []
            for batch_sample in batch_samples:
for i in range(3):
                    current_path = '/home/carnd/Data/IMG/'+batch_sample[i].split('/')[-1]
                    image = ndimage.imread(current_path)
                    if (i == 0):
                        measurement = float(batch_sample[3])
                    if (i == 1):
                        measurement = float(batch_sample[3]) + correction
                    if (i == 2):
                        measurement = float(batch_sample[3]) - correction
                    images.append(image)
                    measurements.append(measurement)
```
This code means that I use all the three cameras in a loop and that I add a 0.22 correction for side cameras. If I have a center camera driving with a 0.0 angle on the center of the road, my left camera will need to a little bit to the right, and my right camera a little bit to the left.

##### C. Data augmentation with flipping
A better alternative to driving counter-clockwise is to flip every image so that it doesn't learn too much to drive to the left (the track 1 field does steer to the left more ofter than to the right).
To do that, I did use another list : augmented_images and augmented_measurements. I used ```cv2.flip``` function with flipping every images and multiplying every corresponding measurement by -1.
```
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
```

##### D. Data augmentation with augment_brightness function
I didn't augment brightness myself but used a function presented in an article : https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
This article also inspired my next part.
##### E. Data augmentation with random shadows

For D and E, I doubled every image I had with an image that had more brightness or an image that had random shadows.
Here is an example of an image with random shadows.
![alt text][image1]

We can see that the image now has a shadow covering the screen. The point of this is multiple, first it is more data to train. Second, it helps generalize the mode to track 2 that has a lot of shadows.

At the end of data augmentation, I have a sample of 192 816 images for training and 48 216 for validation, so 167 040 total data, from 10 043 data initially.

### 2. Model Architecture

##### Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia Team. This model is not really complex and is the model used by the Nvidia Team of a real self-driving car so I thought it would be good. I however had questions regarding this model. Why this number of layers ? What makes this model better than the same model with one more/less convolution ?

##### Final Model Architecture
![alt text][image3]

I did add a dropout after Layer 2 because experimenting a 0.5 Dropout gave me a lower loss. It could help with early overfitting and helped my model generalize a little bit.

### 3. Training
My process for training is quite simple.
I use an adam optimizer and I tune the learning rate to 0.0001. I then train on three epochs so it doesn't take too long.
```
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
history_object = model.fit_generator(train_generator, samples_per_epoch= 192816, 
                                     validation_data=validation_generator, 
                                     nb_val_samples=48216, 
                                     nb_epoch=3, 
                                     verbose=1)
```
I could notice I did use a generator so i don't have to load the data and store it on a drive. There are almost 200k data used for training, I used a batch size of 64. After tuning the batch size a lot, I don't feel like it has any consequences on the loss. Here is the corresponding plot.

![alt text][image2]

### 4. Video
Finally, here is the video corresponding to my training.
https://github.com/Jeremy26/behavioral-cloning/blob/master/run1.mp4

Thank you for reading.
