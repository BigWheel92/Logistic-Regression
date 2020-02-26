# LogisticRegression
a vectorized binary logistic regression implementation in python.


The following functions are supported:

1. fit(self, train_X, train_Y, learningRate=0.01, numOfIterations=2000, validation_X=None, validation_Y=None): fit function is passed as parameters training dataset (train_X), training dataset labels (train_Y), learningRate, numOfIterations, validation dataset and validation dataset labels. This funtion then learns weights.

2. predict(self, test_X): predict function is passed as parameter the test set (test_X). It then predicts the labels of each item in the test set and returns the labels in an array.

3. sigmoid(self, Z): sigmoid function (activation function) is used by above two functions.

Note:

-> The input shape for training set, validation set, and test set must be (m, nx) where m is the number of items in the set and nx is the number of features.

-> The shape of array containing labels for training set, test set and validation set must be (m, 1) where m is the number of items.

-> The model has been tested in main.py on dataset containg cat images (dataset has been taken from coursera deep learning course assignment). The model gives 72% test accuracy.
