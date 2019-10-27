import os
import json
import time
import numpy as np

from .FCNN_FA import FCNN_FA


class FCNN_WM(FCNN_FA):
    '''
            Description: Class to define a Fully Connected Neural Network (FCNN)
                         with weight mirrors (WM)
    '''

    def __init__(self, sizes, save_dir):
        '''
        Description: initialize the biases, forward weights and backward weights using
        a Gaussian distribution with mean 0, and variance 1.
        Params:
            - sizes: a list of size L; where L is the number of layers
                     in the deep neural network and each element of list contains
                     the number of neuron in that layer.
                     first and last elements of the list corresponds to the input
                     layer and output layer respectively
                     intermediate layers are hidden layers.
            - save_dir: the directory where all the data of experiment will be saved
        '''
        super(FCNN_WM, self).__init__(sizes, save_dir)

    def mirror(self, batch_size, X_shape=None, mirror_learning_rate=0.01, noise_amplitude=0.1):
        '''
        Description: weight mirroring by feeding an iid Gaussian noise through *each* layer of the network
                     If the iid Gaussian noise is generated once and get forward-propagated,
                     the iid property is lost for hidden layers.
        Params:
            - batch_size: size of the mirroring batch
            - X_shape: the shape of the noise matrix
            - mirror_learning_rate: eta which controls the size of changes in backward weights
            - noise_amplitude: the amplitude of the iid Gaussian noise
        '''
        if not X_shape is None:
            n_batches = int(X_shape[0] / batch_size)
        else:
            n_batches = 1

        for i in range(n_batches):
            for layer, (b, w, back_w) in enumerate(zip(self.biases, self.weights, self.backward_weights)):
                noise_x = noise_amplitude * (np.random.rand(w.shape[1], batch_size) - 0.5)
                noise_y = self.sigmoid(np.matmul(w, noise_x) + b)
                # update the backward weight matrices using the equation 7 of the paper manuscript
                back_w += mirror_learning_rate * np.matmul(noise_x, noise_y.T)

        # Prevent feedback weights growing too large
        for layer, (b, w, back_w) in enumerate(zip(self.biases, self.weights, self.backward_weights)):
            x = np.random.rand(back_w.shape[1], batch_size)
            y = np.matmul(back_w, x)
            y_std = np.mean(np.std(y, axis=0))
            back_w = 0.5 * back_w / y_std

    def train(self, X_train, y_train, X_test, y_test, batch_size, learning_rate, epochs, test_frequency):
        '''
        Description: Batch-wise trains image features against corresponding labels.
                     The forward and backward weights and biases of the neural network are updated through
                     the Kolen-Pollack algorithm on batches using SGD
                     del_b and del_w are of same size as all the forward weights and biases
                     of all the layers. del_b and del_w contains the gradients which
                     are used to update forward weights and biases

        Params:
            - X_train, y_train: lists of training features and corresponding labels
            - X_test, y_test: lists of testing features and corresponding labels
            - batch_size: size of the batch
            - learning_rate: eta which controls the size of changes in weights & biases
            - epochs: no. of times to iterate over the whole data
            - test_frequency: the frequency of the evaluation on the test data
        '''
        n_batches = int(X_train.shape[0] / batch_size)

        # Start with an initial update of the backward matrices with weight mirroring
        self.mirror(batch_size, X_train.shape)

        for j in range(epochs):
            # initialize the epoch field in the data to store
            self.data['epoch_{}'.format(j)] = {}
            start = time.time()
            epoch_loss = []
            batch_iter = self.get_batch(X_train, y_train, batch_size)

            for i in range(n_batches):
                (batch_X, batch_y) = next(batch_iter)
                batch_loss, delta_del_b, delta_del_w = self.backpropagate(batch_X, batch_y)
                epoch_loss.append(batch_loss)
                del_b = delta_del_b
                del_w = delta_del_w
                # update weight and biases
                self.weights = [w - (learning_rate / batch_size)
                                * delw for w, delw in zip(self.weights, del_w)]
                self.biases = [b - (learning_rate / batch_size)
                               * delb for b, delb in zip(self.biases, del_b)]
                # update the backward matrices with weight mirroring
                self.mirror(batch_size=batch_size)

            epoch_loss = np.mean(epoch_loss)
            self.data['epoch_{}'.format(j)]['loss'] = epoch_loss

            # Log the loss
            log_str = "\nEpoch {} completed in {:.3f}s, loss: {:.3f}".format(j, time.time() - start, epoch_loss)
            self.print_and_log(log_str)

            # Evaluate on test set
            test_accuracy = self.eval(X_test, y_test)
            log_str = "Test accuracy: {}%".format(test_accuracy)
            self.print_and_log(log_str)
            self.data['epoch_{}'.format(j)]['test_accuracy'] = test_accuracy

            # Compute angles between both weights and deltas
            deltas_angles, weights_angles = self.evaluate_angles(X_train, y_train)
            self.data['epoch_{}'.format(j)]['delta_angles'] = deltas_angles
            self.data['epoch_{}'.format(j)]['weight_angles'] = weights_angles

            # save results as a json file
            with open(os.path.join(self.save_dir, 'results.json'), 'w') as f:
                json.dump(self.data, f)
