import os
import json
import time
import numpy as np


class FCNN_BP(object):
    '''
            Description: Class to define a Fully Connected Neural Network (FCNN)
                         with backpropagation (BP) as learning algorithm
    '''

    def __init__(self, sizes, save_dir):
        '''
        Description: initialize the biases and weights using a Gaussian
        distribution with mean 0, and variance 1.
        Params:
            - sizes: a list of size L; where L is the number of layers
                     in the deep neural network and each element of list contains
                     the number of neuron in that layer.
                     first and last elements of the list corresponds to the input
                     layer and output layer respectively
                     intermediate layers are hidden layers.
            - save_dir: the directory where all the data of experiment will be saved
        '''
        self.num_layers = len(sizes)
        self.save_dir = save_dir
        # setting appropriate dimensions for weights and biases
        self.biases = [np.sqrt(1. / (x + y)) * np.random.randn(y, 1)
                       for x, y in zip(sizes[:-1], sizes[1:])]
        self.weights = [np.sqrt(1. / (x + y)) * np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        # define the variables to save data in during training and testing
        self.data = {}

    def print_and_log(self, log_str):
        '''
        Description: Print and log messages during experiments
        Params:
            - log_str: the string to log
        '''
        print(log_str)
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f_:
            f_.write(log_str + '\n')

    def sigmoid(self, out):
        '''
        Description: the sigmoid activation function
        Params:
            - out: a list or a matrix to perform the activation on
        Outputs: the sigmoid activated list or a matrix
        '''
        return 1.0 / (1.0 + np.exp(-out))

    def delta_sigmoid(self, out):
        '''
        Description: the derivative of sigmoid activation function
        Params:
            - out: a list or a matrix to perform the activation on
        Outputs: the sigmoid prime activated list or matrix
        '''
        return self.sigmoid(out) * (1 - self.sigmoid(out))

    def SigmoidCrossEntropyLoss(self, a, y):
        """
        Description: the cross entropy loss
        Params:
            - a: the last layer activation
            - y: the target one hot vector
        Outputs: a loss value
        """
        return np.mean(np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)), axis=0))

    def feedforward(self, x):
        '''
        Description: Forward Passes an image feature matrix through the Deep Neural
                                 Network Architecture.
        Params:
            - x: the input signal
        Outputs: 2 lists which stores outputs and activations at every layer:
                 the 1st list is non-activated and 2nd list is activated
        '''
        activation = x
        activations = [x]  # list to store activations for every layer
        outs = []          # list to store out vectors for every layer
        for b, w in zip(self.biases, self.weights):
            out = np.matmul(w, activation) + b
            outs.append(out)
            activation = self.sigmoid(out)
            activations.append(activation)

        return outs, activations

    def get_batch(self, X, y, batch_size):
        '''
        Description: A data iterator for batching of input signals and labels
        Params::
            - X, y: lists of input signals and its corresponding labels
            - batch_size: size of the batch
        Outputs: a batch of input signals and labels of size equal to batch_size
        '''
        for batch_idx in range(0, X.shape[0], batch_size):
            batch = (X[batch_idx:batch_idx + batch_size].T,
                     y[batch_idx:batch_idx + batch_size].T)
            yield batch

    def train(self, X_train, y_train, X_test, y_test, batch_size, learning_rate, epochs, test_frequency):
        '''
        Description: Batch-wise trains image features against corresponding labels.
                     The weights and biases of the neural network are updated through
                     backpropagation on batches using SGD
                     The variables del_b and del_w are of same size as all the weights and biases
                     of all the layers. The variables del_b and del_w contains the gradients which
                     are used to update weights and biases.

        Params:
            - X_train, y_train: lists of training features and corresponding labels
            - X_test, y_test: lists of testing features and corresponding labels
            - batch_size: size of the batch
            - learning_rate: eta which controls the size of changes in weights & biases
            - epochs: no. of times to iterate over the whole data
            - test_frequency: the frequency of the evaluation on the test data
        '''
        n_batches = int(X_train.shape[0] / batch_size)

        for j in range(epochs):
            # initialize the epoch field in the data to store
            self.data['epoch{}'.format(j)] = {}

            start = time.time()
            epoch_loss = []
            batch_iter = self.get_batch(X_train, y_train, batch_size)

            for i in range(n_batches):
                (batch_X, batch_y) = next(batch_iter)
                batch_loss, del_b, del_w = self.backpropagate(batch_X, batch_y)
                epoch_loss.append(batch_loss)
                # update weight and biases
                self.weights = [w - (learning_rate / batch_size)
                                * delw for w, delw in zip(self.weights, del_w)]
                self.biases = [b - (learning_rate / batch_size)
                               * delb for b, delb in zip(self.biases, del_b)]
            epoch_loss = np.mean(epoch_loss)
            self.data['epoch{}'.format(j)]['loss'] = epoch_loss

            # Log the loss
            log_str = "\nEpoch {} completed in {:.3f}s, loss: {:.3f}".format(j, time.time() - start, epoch_loss)
            self.print_and_log(log_str)

            # Evaluate on test set
            test_accuracy = self.eval(X_test, y_test)
            log_str = "Test accuracy: {}%".format(test_accuracy)
            self.print_and_log(log_str)
            self.data['epoch{}'.format(j)]['test_accuracy'] = test_accuracy

            # save results as a json file
            with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
                json.dump(self.data, f)

    def backpropagate(self, x, y):
        '''
        Description: Based on the derivative(delta) of cost function the gradients(rate of change
                     of cost function with respect to weights and biases) of weights and biases are calculated.
                     The variables del_b and del_w are of same size as all the weights and biases
                     of all the layers. The variables del_b and del_w contains the gradients which
                     are used to update weights and biases.
        Params:
            - x, y: training feature and corresponding label
        Outputs: del_b: gradient of bias
                 del_w: gradient of weight
        '''
        del_b = [np.zeros(b.shape) for b in self.biases]
        del_w = [np.zeros(w.shape) for w in self.weights]

        outs, activations = self.feedforward(x)

        # Cost function
        loss = self.SigmoidCrossEntropyLoss(activations[-1], y)

        # calculate derivative of cost Sigmoid Cross entropy which is to be minimized
        delta_cost = activations[-1] - y
        # backward pass to reduce cost gradients at output layers
        delta = delta_cost
        del_b[-1] = np.expand_dims(np.mean(delta, axis=1), axis=1)
        del_w[-1] = np.matmul(delta, activations[-2].T)

        # updating gradients of each layer using reverse or negative indexing, by propagating
        # gradients of previous layers to current layer so that gradients of weights and biases
        # at each layer can be calculated
        for l in range(2, self.num_layers):
            out = outs[-l]
            delta_activation = self.delta_sigmoid(out)
            delta = np.matmul(self.weights[-l + 1].T, delta) * delta_activation
            del_b[-l] = np.expand_dims(np.mean(delta, axis=1), axis=1)
            del_w[-l] = np.dot(delta, activations[-l - 1].T)

        return loss, del_b, del_w

    def eval(self, X, y):
        '''
        Description: Based on trained(updated) weights and biases, predict a batch of labels and compare
                     them with the original labels and calculate accuracy
        Params:
            - X: test input signals
            - y: test labels
        Outputs: accuracy of prediction
        '''
        outs, activations = self.feedforward(X.T)
        # count the number of times the postion of the maximum value is the predicted label
        count = np.sum(np.argmax(activations[-1], axis=0) == np.argmax(y.T, axis=0))
        test_accuracy = 100. * count / X.shape[0]
        return test_accuracy
