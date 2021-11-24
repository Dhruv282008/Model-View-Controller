import numpy as np #Performs complex mathematical/list operations
import pandas as pd #To treat the data as a dataframe
from sklearn.datasets import fetch_openml #This function allows us to retrieve a dataset by name from OpenML, a public repository for machine learning data and experiments.
from sklearn.model_selection import train_test_split #Helps us to test and train the data
from sklearn.linear_model import LogisticRegression #Used to create a logistic regression classifier
from PIL import Image
import PIL.ImageOps

X,y = fetch_openml('mnist_784', version = 1, return_X_y = True)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 2500, train_size = 7500, random_state = 9)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver = 'saga',  multi_class = 'multinomial').fit(X_train_scaled, Y_train)

def getPrediction(image):
    im_PIL = Image.open(image)
    im_bw = im_PIL.convert('L')
    im_bw_resized = im_bw.resized((28, 28), Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(im_bw_resized, pixel_filter)
    im_bw_resized_scaled = np.clip(im_bw_resized - min_pixel, 0, 255)
    max_pixel = np.max(im_bw_resized)
    im_bw_resized_scaled_array = np.asarray(im_bw_resized_scaled)/max_pixel
    test_sample = np.array(im_bw_resized_scaled_array).reshape(1, 784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]
    