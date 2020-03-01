import numpy as np

class BinaryLogisticRegression:

    def __init__(self):
        pass

    def fit(self, train_X, train_Y, learningRate=0.01, numOfIterations=2000, validation_X=None, validation_Y=None):

        self.W = np.zeros((train_X.shape[1], 1))
        self.b = 0
        m = train_X.shape[0]

        for i in range(numOfIterations):
            # forward propagate
            Z = np.dot(train_X, self.W) + self.b
            self.A = self.sigmoid(Z)

            # computing cost
            cost = -(1 / m) * np.sum((train_Y * np.log(self.A) + (1 - train_Y) * np.log(1 - self.A)), axis=0)
            cost = np.squeeze(cost)

            #convert computed values in A to absolute 0 or 1 (for measuring accuracy)
            predictions=np.round(self.A)

            print("Iteration= ", i + 1, ". Cost= ", cost, ". Train Accuracy= {}%".format(100 - np.mean(np.abs(predictions - train_Y)) * 100), '. ', sep='', end='')

            # calculating gradients
            dz = self.A - train_Y
            dw = (1 / m) * np.dot(train_X.T, dz)
            db = (1 / m) * np.sum(dz, axis=0)

            # updating weights
            self.W = self.W - learningRate * dw
            self.b = self.b - learningRate * db

            # printing validation accuracy if validation set is passed as parameter
            if validation_X is None:
                print('')
                continue
            else:
                validation_Z = np.dot(validation_X, self.W) + self.b
                validation_predictions = self.sigmoid(validation_Z)
                for i in range(validation_predictions.shape[0]):
                    validation_predictions[i, 0] = 1 if validation_predictions[i, 0] > 0.5 else 0
                print("Validation Accuracy: {}%".format(100 - np.mean(np.abs(validation_predictions - validation_Y)) * 100))

    # predict function predicts labels of the test set and returns them in an array.
    def predict(self, test_X):
        Z = np.dot(test_X, self.W) + self.b
        predictions = self.sigmoid(Z)

        for i in range(predictions.shape[0]):
            predictions[i, 0] = 1 if predictions[i, 0] > 0.5 else 0

        return predictions

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
