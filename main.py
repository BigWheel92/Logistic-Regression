#The dataset is taken from the coursera assignment 1 of deep learning specialization course 1

import h5py


def load_dataset():
    train_dataset = h5py.File('./dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('./dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


import model
import numpy as np

#load dataset
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes= load_dataset()


#reshape and normalize dataset
train_set_x=np.reshape( train_set_x_orig, (train_set_x_orig.shape[0], -1))
train_set_x=train_set_x/255.0

test_set_x=np.reshape( test_set_x_orig, (test_set_x_orig.shape[0], -1))
test_set_x=test_set_x/255.0

train_set_y=np.reshape(train_set_y_orig, (train_set_y_orig.shape[1], 1))
test_set_y=np.reshape(test_set_y_orig, (test_set_y_orig.shape[1], 1))

#declare the model
lg= model.BinaryLogisticRegression()

#train the model
lg.fit(train_X=train_set_x, train_Y=train_set_y, learningRate=0.005, numOfIterations=1500)


#evaluate the model
predictions=lg.predict(test_X=test_set_x)

#print test accuracy
print("Test Accuracy: {}%".format(100 - np.mean(np.abs(predictions - test_set_y)) * 100))