import os
import json
import time
import numpy as np

from .FCNN_BP import FCNN_BP


class FCNN_FA(FCNN_BP):
    '''
            Description: Class to define a Fully Connected Neural Network (FCNN)
                         with feedback alignment (FA) as learning algorithm
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
        super(FCNN_FA, self).__init__(sizes, save_dir)
        # setting backward matrices
        self.backward_weights = [np.sqrt(1. / (x + y)) * np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

    def train(self, X_train, y_train, X_test, y_test, batch_size, learning_rate, epochs, test_frequency):
        '''
        Description: Batch-wise trains image features against corresponding labels.
                     The forward weights and biases of the neural network are updated through
                     feedback alignment on batches using SGD
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

        for j in range(epochs):
            # initialize the epoch field in the data to store
            self.data['epoch_{}'.format(j)] = {}

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

    def backpropagate(self, x, y, eval_delta_angle=False):
        '''
        Description: Based on the derivative(delta) of cost function the gradients(rate of change
                     of cost function with respect to weights and biases) of weights and biases are calculated.
                     The variables del_b and del_w are of same size as all the forward weights and biases
                     of all the layers. The variables del_b and del_w contains the gradients which
                     are used to update the forward weights and biases.
        Params:
            - x, y: training feature and corresponding label
            - eval_delta_angle: a boolean to determine if the angle between deltas should be computed
        Outputs:
            - del_b: gradient of bias
            - del_w: gradient of weight
        '''
        # Set a variable to store angle during evaluation only
        if eval_delta_angle:
            deltas_angles = {}

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
            if eval_delta_angle:
                # compute both FA and BP deltas and the angle between them
                delta_bp = np.matmul(self.weights[-l + 1].T, delta) * delta_activation
                delta = np.matmul(self.backward_weights[-l + 1], delta) * delta_activation
                deltas_angles['layer_{}'.format(self.num_layers - l)] = self.angle_between(delta_bp, delta)
            else:
                delta = np.matmul(self.backward_weights[-l + 1], delta) * delta_activation
            del_b[-l] = np.expand_dims(np.mean(delta, axis=1), axis=1)
            del_w[-l] = np.dot(delta, activations[-l - 1].T)
        if eval_delta_angle:
            return deltas_angles
        else:
            return loss, del_b, del_w

    def angle_between(self, A, B):
        '''
        Description: computes the angle between two matrices A and B
        Params:
            - A: a first matrix
            - B: a second matrix
        Outputs:
            - angle: the angle between the two vectors resulting from vectorizing and normalizing A and B
        '''
        flat_A = np.reshape(A, (-1))
        normalized_flat_A = flat_A / np.linalg.norm(flat_A)

        flat_B = np.reshape(B, (-1))
        normalized_flat_B = flat_B / np.linalg.norm(flat_B)

        angle = (180.0 / np.pi) * np.arccos(np.clip(np.dot(normalized_flat_A, normalized_flat_B), -1.0, 1.0))
        return angle

    def evaluate_angles(self, X_train, y_train):
        '''
        Description: computes the angle between both:
                        - the forward and backwards matrices
                        - the delta signals
        Params:
            - X_train, y_train: training feature and corresponding label
        Outputs:
            - deltas_angles: the angle between the delta signal and the backpropagation delta signal
            - weights_angles: the angle between the forward and backwards matrices
        '''

        # Evaluate angles between matrices
        weights_angles = {}
        for layer, (w, back_w) in enumerate(zip(self.weights, self.backward_weights)):
            matrix_angle = self.angle_between(w.T, back_w)
            weights_angles['layer_{}'.format(layer)] = matrix_angle
            log_str = 'In layer {} angle between matrices: {}'.format(self.num_layers - layer, matrix_angle)
            self.print_and_log(log_str)

        # Evaluate angles between delta signals
        [sample_x, sample_y] = list(next(self.get_batch(X_train, y_train, batch_size=1)))
        deltas_angles = self.backpropagate(sample_x, sample_y, eval_delta_angle=True)
        log_str = 'Angle between deltas: {}'.format(deltas_angles)
        self.print_and_log(log_str)
        return deltas_angles, weights_angles
