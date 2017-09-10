#Traffic Sign Recognition Writeup - M. Lapinskas 9 Sept 2017

Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:

Load the data set (see below for links to the project data set)
Explore, summarize and visualize the data set
Design, train and test a model architecture
Use the model to make predictions on new images
Analyze the softmax probabilities of the new images
Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization1.jpg "Visualization - Simple"
[image1a]: ./examples/visualization2.jpg "Visualization - Full Set"
[image2]: ./examples/visualization1.jpg "Grayscaling - Before"
[image2a]: ./examples/grayscale.jpg "Grayscaling - After"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/10_new_images.png "New Test Images"
[image5]: ./examples/bad_new_images.png "Bad New Test Images"

Rubric Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! Here is a link to my project code: TODO from Github

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the NumPy library to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Traffic sign image data shape = (32, 32, 3)
The number of unique classes/labels = 43


####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Initially, I displayed a single random image from the data set with classification label:

![alt text][image1]

Later, I added a display of one image from each classification:

![alt text][image1a]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale to reduce memory consumption, make image normalization easy and while color helps humans understand the meaning of signs, the CNN classification performance of traffice signs is not reliant on sign color.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image2a]

As a last step, I normalized the image data because without consistent image input, the varience from image to image would like overwhelm the learning rate corrections by either under or overshooting.

I decided not to generate additional data because after doing an initial pass the accuracty was very near 94% and I am very late with this project and need to catch up.  I understand that the training set has an inconsistent number of images for each classification and after completing the exercise, I can see the value of equalizing the number of images while adding variation in the images with slight rotation and distortion.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was the 'classic' LeNet model and consisted of the following layers:

| Layer                 | Description                                   | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 Grayscale image                       | 
| L1:Convolution 5x5    | 1x1 stride, 'valid' padding, outputs 28x28x6  |
| L1:RELU               | Activation                                    |
| L1:Max pooling        | 2x2 stride, 'valid' padding, outputs 14x14x6  |
| L2:Convolution 5x5    | 1x1 stride, 'valid' padding, outputs 10x10x16 |
| L2:RELU               | Activation                                    |
| L2:Max pooling        | 2x2 stride, 'valid' padding, outputs 5x5x16   |
| L2:Flatten            | Outputs 400                                   |
| L3:Fully connected    | Outputs 120                                   |
| L3:RELU               | Activation                                    |
| L4:Fully connected    | Outputs 84                                    |
| L4:RELU               | Activation                                    |
| L5:Fully connected    | Outputs 43                                    |
| Softmax               | Used for training                             |

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a workflow that included TensorFlow implementations of Softmax Cross Entropy calculation, Reduce Mean for loss and optimization withAdam Optimizer.

Here are the tuning parameters used:
* Epochs = 100
* Batch size = 128
* Learning rate = 0.001
* mu = 0
* sigma = 0.1


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
 
* Train Set Accuracy = 1.000
* Validation Set Accuracy = 0.934 (EPOCH 100)
* Test Set Accuracy = 0.916

* What architecture was chosen?
A) I chose the LeNet architecture initially setup to classify MINST numbers.

* Why did you believe it would be relevant to the traffic sign application?
A) Classifying traffic signs is very similar to classifying numbers, so the LeNet model should perform well by increasing the number of classifications from 10 to the total number of signs to classify.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
A) Training set accuracy was 100% which validates that training considered all cases in the training set.  The validation set accuracy for my last pass was at 93.4% and varied as high as 94.1%.  Test data was 91.6%.  As mentioned earlier, I believe that augmenting the training data set to include variations of image and provide additional samples for classifications with less data would increase both the validation and training data set accuracy.  Variations in validation set accuracy over many 'runs' was due to the image shuffling.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I used 10 German traffic signs that I found on the web:

![alt text][image4]

These images all get classified correctly but I did find many images that were difficult to classify:

![alt text][image5]

These images might be difficult to classify because of a number of reasons given my model.  Sign is: not centered, skewed, too modified or not in the list of classified signs.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Index | Image                   | Prediction                          | 
|:-----:|:-----------------------:|:-----------------------------------:| 
|   0   | 18,General caution      | 18,General caution                  | 
|   1   | 27,Pedestrians          | 27,Pedestrians                      |
|   2   | 18,General caution      | 18,General caution                  | 
|   3   | 25,Road work            | 25,Road work                        | 
|   4   | 25,Road work            | 25,Road work                        | 
|   5   | 17,No entry             | 17,No entry                         | 
|   6   | 28,Children crossing    | 28,Children crossing                | 
|   7   | 11,Right-of-way next    | 11,Right-of-way next intersection   | 
|   8   | 40,Roundabout mandatory | 40,Roundabout mandatory             | 
|   9   | 1,Speed limit (30km/h)  | 1,Speed limit (30km/h)              | 

My original set of 10 signs had only 3 of 10 matching (30%).  As I figured out that signs needed to be square and centered, with the replacement new signs the model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 91.6%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the near the bottom of the Ipython notebook after the label: "Top 5 Predictions for Each New Image".

For these well healed images, the predictions are 100% except for image 6 where the prediction is 94% for 28,Children crossing but has a 6% prediction for 29,Bicycles crossing. This makes sense given the relative nature of the graphics. Squint when you look at them and you can make out the 2 wheels versus the 2 people shapes.

| Index | Probability (Prediction)                                           | 
|:-----:|-------------------------------------------------------------------:| 
|   0   | 1.00(18)      0.00(27)      0.00(11)      0.00(30)      0.00( 1)   | 
|   0   | 1.00(18)      0.00(27)      0.00(11)      0.00(30)      0.00( 1)   | 
|   1   | 1.00(27)      0.00(11)      0.00(18)      0.00(40)      0.00(21)   | 
|   2   | 1.00(18)      0.00(27)      0.00(11)      0.00(26)      0.00( 0)   | 
|   3   | 1.00(25)      0.00( 5)      0.00(23)      0.00(20)      0.00(31)   | 
|   4   | 1.00(25)      0.00(20)      0.00(38)      0.00( 5)      0.00(12)   | 
|   5   | 1.00(17)      0.00(33)      0.00(14)      0.00(35)      0.00( 4)   | 
|   6   | 0.94(28)      0.06(29)      0.00(23)      0.00(12)      0.00(38)   | 
|   7   | 1.00(11)      0.00(21)      0.00( 0)      0.00( 1)      0.00( 2)   | 
|   8   | 1.00(40)      0.00( 7)      0.00(10)      0.00(11)      0.00(37)   | 
|   9   | 1.00( 1)      0.00( 0)      0.00( 5)      0.00( 2)      0.00( 3)   | 

(Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

I appreciate the opportunity to dive deeper, but unfortunatly, I need to be moving onto the next section of the course at this time.

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?