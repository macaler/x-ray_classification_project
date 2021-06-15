# x-ray_classification_project

This project was completed to fulfill the requirements of Codecademy's "Build Deep Learning Models with TensorFlow" skill path. The project's goal was to build a deep learning model of x-ray lung scans to determine if a patient has pneumonia, COVID-19, or no pulmonary illness. More specifically, I built a classification model using TensorFlow and Keras which produces a diagnosis based on a patient's chest x-ray scan.

I have included in this repository the code used to build, train, and evaluate my deep learning model as "x-ray_classification_challenge_MAC.py." The code was written in Python 3.8. In writing my code, I used the most recent version (to the best of my knowledge) of the libraries os, TensorFlow, random, NumPy, Matplotlib, and scikit-learn. Under the hood, I also used the pillow library, though I found that I did not need to explicitly import PIL into my code. I attempted to initialize random seed settings, and in order to do so my code should be run with the following command:
PYTHONHASHSEED=0 python x-ray_classification_challenge_MAC.py
I recommend running from the IDLE environment.

The data used in the project come from Codecademy; they, in turn, downloaded the image files from:
https://www.kaggle.com/pranavraikokte/covid19-image-dataset <br>
I have included the image files with this repository. I have not made any changes to the files themselves, nor have I changed any of the directory sub-structures. <br>
The dataset was released by user pranavraikokte under the CC BY-SA 4.0 license as found at https://creativecommons.org/licenses/by-sa/4.0/. I did not make any changes to the files found in the data set. However, I did use the data to build my deep learning model. I am not clear as to whether or not using the data to build a deep learning model counts as "build upon the material" (as detailed at https://creativecommons.org/licenses/by-sa/4.0/), but I suppose it is better to be safe than sorry, so I will be releasing my work under the same license. <br>
Kaggle user pranavraikokte acknowledges the University of Montreal for releasing the x-ray images. 

So long as my code and the image files as downloaded are in the same directory, you should be able to run my code.
