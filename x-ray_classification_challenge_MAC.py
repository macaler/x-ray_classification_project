###############################################################################
#                               Basic Neural Network Image Classification Model Building:                                        #
#                                                       A Codecademy Pro Project                                                                    #
###############################################################################
'''
This project was completed to fulfill the requirements of Codecademy's "Build Deep Learning Models
with TensorFlow" skill path. The project's goal was to build a deep learning model of x-ray lung scans
to determine if a patient has pneumonia, COVID-19, or no pulmonary illness. More specifically, I built
a classification model using TensorFlow and Keras which produces a diagnosis based on a patient's
chest x-ray scan.

The data used in the project come from:
https://www.kaggle.com/pranavraikokte/covid19-image-dataset
The Kaggler acknowledges the University of Montreal for releasing the images.
It is a relatively small image data set, containing just 251 images in the "train" directory and 66 images
in the "test" directory. Following Codecademy's lead, I will be using the images in the "test" directory
as the validation data set. Given such a small data set, and my naive assumption that the x-ray scans
of COVID-19 patients will likely be similar to the x-ray scans of pneumonia patients, I do not foresee
being able to build a tremendously accurate model. But the task at hand is to try my hand at building
one, so I shall build an image classifier.

Below is the code I wrote to build, train, and test my image classification deep learning model. Some
commentary is included in the comments.
'''

###############################################################################
#                                      Import classes and functions from various libraries                                              #
###############################################################################

''' I noticed that I was getting several warnings when compiling TensorFlow regarding appropriate
compiler flags; apparently, I'm not the only one, as evidenced by the following question from
stackoverflow:
https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
I have chosen to implement the commenter's solution.'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''Import several classes and functions from TensorFlow and Keras:'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy, AUC

'''Note that, since I am working with image data, I need to have the pillow library installed, and I do in
fact have it installed on my machine. I found that my code was able to work without an explicit call to
import pillow/PIL.'''

''' import random, Numpy and Pyplot:'''
import random
import numpy
import matplotlib.pyplot as plt

'''import model evaluation functions from scikit-learn:'''
from sklearn.metrics import classification_report, confusion_matrix

###############################################################################
#                                                         Define global variables                                                                        #
###############################################################################
'''Define some variables which describe the images' properties. The images are all 256x256 pixels and
are in grayscale, so the maximum value a pixel can have is 255 (the minimum, of course, is 0).'''
thetargetsize = (256,256)
rescale_factor = 1.0/255.0

'''Some of the model hyperparameters I fiddled with the most were batch size, learning rate, and
the number of epochs to train over. I chose to gather these all into one convenient place to make
it easier to change them during the optimization process.'''
thebatchsize = 24
thelearningrate = 0.006
numofepochs = 500

''' The following two lines of code were taken from
https://gist.github.com/CircleOnCircles/87a6dc0b8884affb2a546ca6036cad04 '''

''' The below is necessary for starting core Python generated random numbers in a well-defined state.'''
random.seed(26)

'''The below set_seed() will make random number generation in the TensorFlow backend have a
well-defined initial state. For further details, see:
https://www.tensorflow.org/api_docs/python/tf/random/set_seed'''
tf.random.set_seed(26)

'''To ensure the above random seed settings work properly, this code should be run with the following
command:
PYTHONHASHSEED=0 python challenge-starter.py
I recommend running from the IDLE environment.'''

###############################################################################
#                                    Load the training data using an ImageDataGenerator() object                               #
###############################################################################
print("Training Data:")

''' Instantiate an ImageDataGenerator() object with several data augmentation techniques. The data
augmentation techniques utilized include horizontal_flip (which randomly flips images horizontally),
width_shift_range (which randomly shifts an image along its width by up to a set percentage),
height_shift_range (which randomly shifts an image along its height by up to a set percentage),
zoom_range (which randomly increases or decreases an image's size by up to a set percentage),
and shear_range (which randomly increases or decreses shear intensity by up to a set percentage).
The data will also be scaled such that pixel values all lie between 0 and 1. '''
train_imagegener = ImageDataGenerator(rescale = rescale_factor, horizontal_flip = True, \
                                 width_shift_range = 0.05, height_shift_range = 0.05, \
                                 zoom_range = 0.1, shear_range = 0.2)

'''Load and batch the data into an iterator object. Images are categorized according to the sub-directory
they were found in. Image property variables were defined above; the images are all in grayscale, so
that was specified in the method call.'''
train_data_iter = train_imagegener.flow_from_directory("Covid19-dataset/train",\
                             class_mode="categorical", color_mode="grayscale", \
                             target_size=thetargetsize, batch_size = thebatchsize)

''' I borrowed the below code from Codecademy. It prints out the batch shape and label shape, to make
sure nothing blatantly obvious has gone wrong.'''
train_batch_input, train_batch_labels  = train_data_iter.next()
print(train_batch_input.shape, train_batch_labels.shape)

###############################################################################
#                              Load in the validation data using an ImageDataGenerator() object                              #
###############################################################################
print("Validation Data:")

''' Instantiate an ImageDataGenerator() object. Because the validation data will be loaded using this
object, no data augmentation techniques are specified. The data is, however, scaled such that pixel
values all lie between 0 and 1.'''
valid_imagegener = ImageDataGenerator(rescale = rescale_factor)

'''Load and batch the data into an iterator object. Images are categorized according to the sub-directory
they were found in. Image property variables were defined above; the images are all in grayscale, so
that was specified in the method call.'''
valid_data_iter = valid_imagegener.flow_from_directory("Covid19-dataset/test",\
                            class_mode="categorical", color_mode="grayscale", \
                            target_size=thetargetsize, batch_size = thebatchsize)

''' I borrowed the below code from Codecademy. It prints out the batch shape and label shape, to make
sure nothing blatantly obvious has gone wrong.'''
valid_batch_input, valid_batch_labels  = valid_data_iter.next()
print(valid_batch_input.shape, valid_batch_labels.shape)

###############################################################################
#                               Build and compile a Sequential() neural network model                                             #
###############################################################################
''' NOTE:
My original model is included below as an epilogue. After reading
https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/,
I realized that I shouldn't be using the softmax activation function between hidden layers of my neural
network, and that the ReLU activation function is a better choice from an algorithmic and conceptual
standpoint. I have updated my model accordingly.'''

''' Instantiate a Sequential() object:'''
model = Sequential()
''' Add the input layer. Images are 256x256 in grayscale, meaning that there is only 1 colour channel.'''
model.add(Input(shape=(256, 256, 1)))
''' Add a Conv2D() layer featuring 12 5x5 filters with a stride of 3 and the default padding (i.e., valid).
I have chosen to use the ReLU activation function for this layer.'''
model.add(Conv2D(12, 5, strides=3, activation="relu"))
''' Add a MaxPooling2D() layer with a window size of 3x3 and a stride of 3, using the default padding
(i.e., valid):'''
model.add(MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
''' Add a Conv2D layer featuring 10 4x4 filters with a stride of 2 and the default padding. I have once
again chosen to use  the ReLU activation function for this layer.'''
model.add(Conv2D(10, 4, strides=2, activation="relu"))
''' Add a MaxPooling2D() layer with a window size of 2x2 and a stride of 2, using the default padding:'''
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
''' Add a Conv2D() layer featuring 8 3x3 filters with a stride of 1 and the default padding. I will again use
 the ReLU activation function for this layer.'''
model.add(Conv2D(8, 3, strides=1, activation="relu"))
''' Add a MaxPooling2D() layer with a window size of 2x2 and a stride of 2, using the default padding:'''
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
''' Add a Conv2D() layer featuring 6 2x2 filters with a stride of 1 and the default padding. Once again,
the ReLU activation function is used for this layer.'''
model.add(Conv2D(6, 2, strides=1, activation="relu"))
''' Add a Flatten() layer to translate the input images into a single vector:'''
model.add(Flatten())
''' Finally, add the output layer as a Dense() layer with 3 perceptrons and a softmax activation function:'''
model.add(Dense(3,activation="softmax"))

''' Choose an optimization algorithm and specify its learning rate. I have chosen to use the Adam
optimization algorithm; its learning rate is specified by one of the global variables defined above:'''
le_optimizer = Adam(learning_rate = thelearningrate)

print("Compiling model ...")
''' Compile the model using categorical cross-entropy as the loss function and choosing categorical
accuracy and AUC (i.e., area under the ROC curve) as metrics to be tracked as the model is trained:'''
model.compile(optimizer=le_optimizer, loss='CategoricalCrossentropy', \
                          metrics=['CategoricalAccuracy','AUC'])

''' To help prevent overfitting, instantiate an EarlyStopping() object which will stop model training if the
validation loss (here, categorical cross-entropy) plateaus or starts to increase after having reached a
minimum value. Monitor the validation loss to decide whether or not to trigger early stopping, but be
patient about it: once a condition for early stopping has been triggered, keep training for 40 epochs to
see if a plateau has started to increase or decrease. And do say if early stopping has been implemented.'''
le_earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)

###############################################################################
#                        Train the Sequential() neural network model using the training data                                 #
###############################################################################
print("Training model ...")

''' Train the model compiled above using the training data iterator. Since I will be keeping track of
validation loss in order to decide whether or not to stop training before the specified number of epochs
has been reached, pass model.fit() the validation data iterator as well. The number of epochs to train
over was specified above as a global variable. Please do not show the training details of every epoch,
but do use the EarlyStopping() object to decide if training needs to be stopped early.'''
model_history = model.fit(train_data_iter, epochs=numofepochs, verbose=0, \
                          steps_per_epoch=train_data_iter.samples/thebatchsize, \
                          validation_data=valid_data_iter, \
                          validation_steps=valid_data_iter.samples/thebatchsize, \
                          callbacks=[le_earlystop])

###############################################################################
#                                See how well the model performs on the validation data                                            #
###############################################################################
print("Plotting figures ...")

'''Meta note: I don't usually assign plot objects to variable names, but this seems to be common
practice so I thought I would give it a try here.'''

'''Set up a figure with a large enough size to see three plots at once: '''
fig = plt.figure(figsize=(6,24))
'''Make sure there is enough space between sub-plots: '''
fig.subplots_adjust(hspace=0.5)

'''First, look at the validation loss (i.e., categorical cross-entropy) to see how its value changed over
the training epochs. Make sure it has reached a minimum and that the final categorical cross-entropy
for the training data and validation data aren't too far off from each other. '''
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(model_history.history['loss'])
ax1.plot(model_history.history['val_loss'])
ax1.set_title('Model Categorical Cross-Entropy')
ax1.set_ylabel('Categorical Cross-Entropy')
ax1.set_xlabel('Epoch')
ax1.legend(['Training Data', 'Validation Data'], loc='upper right')

'''Next, look at the categorical accuracy of both the training data and the validation data as a function
of epoch. Again, make sure that the final accuracies of the training and validation data aren't too far
off from each other.'''
ax2 = fig.add_subplot(3, 1, 2)
ax2.plot(model_history.history['categorical_accuracy'])
ax2.plot(model_history.history['val_categorical_accuracy'])
ax2.set_title('Model Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend(['Training Data', 'Validation Data'], loc='lower right')

'''Finally, look at the area under the FOC curve for both the training and validation data as a function of
epoch. Make sure that the final area under curve (AOC) values of the training and validation data aren't
too far off from each other.'''
ax3 = fig.add_subplot(3, 1, 3)
ax3.plot(model_history.history['auc'])
ax3.plot(model_history.history['val_auc'])
ax3.set_title('Model AUC')
ax3.set_ylabel('Area Under FOC Curve')
ax3.set_xlabel('Epoch')
ax3.legend(['Training Data', 'Validation Data'], loc='lower right')

''' Show the figure:'''
fig.show()

'''Meta note: Having tried assigning variable names to plot objects, I find that I prefer my default method
of leaving things un-assigned. However, it may be a company's coding standard to assign variable names
to plot objects, so I am glad I got a bit of experience doing so.'''


print("Making predictions with the trained model ...")

''' Use the model to predict classifications for the images in the validation data: '''
model_predictions = model.predict(valid_data_iter, steps=valid_data_iter.samples/thebatchsize)

''' I borrowed the following three lines of code from Codecademy. The first of those lines of code uses
NumPy's argmax function to, for each image in the validation data set, find the category with the highest
classification probability and stuff it into an array. The second line plucks off the true classifications of
each image in the validation data set from the validation data set iterator. The third line uses the
validation data set iterator to identify the names of the categories the validation data was originally
classified into.'''
predicted_classes = numpy.argmax(model_predictions, axis=1)
true_classes = valid_data_iter.classes
class_labels = list(valid_data_iter.class_indices.keys())

''' Print the classification report for the validation data.'''
print("Classification Report:")
print( classification_report(true_classes, predicted_classes, target_names=class_labels))   

'''Print the confusion matrix for the validation data. Its format is:
   predict_COVID_true_COVID      |     predict_normal_true_COVID     |     predict_pneumonia_true_COVID
  predict_COVID_true_normal     |  predict_normal_true_normal       |    predict_pneumonia_true_normal
predict_COVID_true_pneumonia | predict_normal_true_pneumonia | predict_pneumonia_true_pneumonia
'''

print("Confusion Matrix:")
print(confusion_matrix(true_classes,predicted_classes, labels = [0,1,2]))

###############################################################################
#                                                       Notes on model performance                                                                #
###############################################################################
'''
Even with initial random seeds set, I found that my results were rather variable based on which precise
images in the training set were flipped horizontally, etc. I have noted validation accuracies (reported in
the classification report) as high as 36% and as low as 30%. So at least some of the time, my model
outperforms random guessing. Sometimes it does not. When comparing my results to those
distributed by Codecademy as their solution to the classification challenge, I see that Codecademy's
"validation" accuracy is also variable, to about the same the degree that mine is. That is to say,
neither my model nor Codecademy's model regularly outperforms random.
(In my first version of this project, I used a softmax activation function between hidden layers, and
noticed more variability in the validation accuracies my models achieved. The current version of the
project uses the relu activation function between hidden layers, and while the validation accuracies do
vary from run to run their range is smaller than it was in the first version of the project.)

I also note that Codecademy's code uses the training data as the validation data; that is to say, they call
"Covid19-dataset/train" in the calls to .flow_from_directory() for both the training and the validation data.
I do not know if this is intentional or if it was an oversight on the part of the developer who wrote it. It
seems to me bad practice to use the full training set for both training and validation purposes.

I promise that I did not look at Codecademy's solution to the challenge until my own model-building code
was complete. Indeed, the model they built is significantly different from mine: it has half the number of
hidden layers that mine does and it implements Dropout() layers (whereas my model does not).
Codecademy's call to ImageDataGenerator() also uses random image rotation to augment the data,
whereas mine relies on horizontal flipping and shear intensity shifting. Both codes use height and width
shifting as well as zoom intensity shifting to augment the data.

The variability in validation accuracy that I am seeing does not surprise me given the relatively
small number of images in the training set (70 pneumonia images, 70 normal images, and
111 COVID-19 images). Indeed, in order to preserve as much training information as possible, I chose
not to do any regularization via Dropout layers, so it is quite possible that my model suffers from a fair
amount of overfitting. Had I implemented Dropout layers, the validation accuracy variability that I am
seeing would likely have been reduced. One of the reasons behind the overfitting is likely that my model
is too complicated for the small amount of data at hand. Were I to do the project over again, I would
probably choose fewer hidden layers and fewer filters in each Conv2D() layer. My goal in building my
model withas many hidden layers as I did above was to increase model accuracy, but in my attempts to
do so I likely made it too complicated and prone to overfit the data.

The above having been said, I understand why Codecademy chose this particular data set for this
particular application: it is timely, it is small, and even my complicated model did not take long to
train. Given that Codecademy gave an option to complete this project on their site, and that they
would not want their bandwidth eaten up by learners making complicated deep learning models on
big image data sets, it's no wonder that they chose such a small image data set to work with. While
that may not make for incredibly accurate results, it does make for a practical first project in
image classification with deep learning.
'''

###############################################################################
#                                                                      Epilogue                                                                                  #
###############################################################################

''' Included below is the first convolutional neural network I built to perform the image classification task
detailed above. However, after reading
https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
I realized that my choice of a softmax activation function between hidden layers probably wasn't the best,
and definitely is not the industry standard. So I decided to try again with a ReLU activation function to see
if I could achieve comparable, if not better, performance.'''
#model = Sequential()
#model.add(Input(shape=(256, 256, 1)))
#model.add(Conv2D(12, 5, strides=3, activation="softmax"))
#model.add(MaxPooling2D(pool_size=(3, 3), strides=(3,3)))
#model.add(Conv2D(10, 4, strides=2, activation="softmax"))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(Conv2D(8, 3, strides=1, activation="softmax"))
#model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(Conv2D(6, 2, strides=1, activation="softmax"))
#model.add(Flatten())
#model.add(Dense(3,activation="softmax"))
