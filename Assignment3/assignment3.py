"""
Neural Networks for Machine Learning

Programming Assignment 3: Optimization and generalization

In this assignment, you're going to train a simple Neural Network, for recognizing
handwritten digits. You'll be programming, looking into efficient optimization, and
looking into effective regularization.
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class Data:
    def __init__(self, datas):
        self.training = Group(datas['training'][0,0]['inputs'], datas['training'][0,0]['targets'])
        self.validation = Group(datas['validation'][0,0]['inputs'], datas['validation'][0,0]['targets'])
        self.test = Group(datas['test'][0,0]['inputs'], datas['test'][0,0]['targets'])


class Group:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets


class Model:
    def __init__(self, input_to_hid, hid_to_class):
        self.input_to_hid = input_to_hid
        self.hid_to_class = hid_to_class


def a3(wd_coefficient, n_hid, n_iters, learning_rate, momentum_multiplier, do_early_stopping, mini_batch_size):
    model = initial_model(n_hid)
    datas = Data(sio.loadmat('data.mat')['data'][0,0])
    n_training_cases = datas.training.inputs.shape[1]

    if n_iters:
        test_gradient(model, datas.training, wd_coefficient)

    # Optimization.
    theta = model_to_theta(model)
    momentum_speed = theta * 0
    training_data_losses = []
    validation_data_losses = []
    best_so_far = {}

    if do_early_stopping:
        best_so_far['theta'] = -1  # this will be overwritten soon
        best_so_far['validation_loss'] = np.inf
        best_so_far['after_n_iters'] = -1
    
    for optimization_iteration_i in range(1, n_iters+1):
        # Pull training batch from data
        batch_start = (optimization_iteration_i - 1) * mini_batch_size % n_training_cases
        batch_end = batch_start + mini_batch_size
        training_batch_inputs = datas.training.inputs[:, batch_start:batch_end]
        training_batch_targets = datas.training.targets[:, batch_start:batch_end]
        training_batch = Group(training_batch_inputs, training_batch_targets)
        # Compute batch gradient, update momentum, update model
        gradient = model_to_theta(d_loss_by_d_model(model, training_batch, wd_coefficient))
        momentum_speed = momentum_speed * momentum_multiplier - gradient
        theta += momentum_speed * learning_rate

        model = theta_to_model(theta)
        # Store losses
        training_data_losses.append(loss(model, datas.training, wd_coefficient))
        validation_data_losses.append(loss(model, datas.validation, wd_coefficient))
        
        # Remember best model
        if do_early_stopping and (validation_data_losses[-1] < best_so_far['validation_loss']):
            best_so_far['theta'] = theta.copy()
            best_so_far['validation_loss'] = validation_data_losses[-1]
            best_so_far['after_n_iters'] = optimization_iteration_i
        
        if not(optimization_iteration_i % (n_iters // 10)):
            print("{:4} iterations: training data loss is {:.5f}, validation data loss is {:.5f}".\
                format(optimization_iteration_i, training_data_losses[-1], validation_data_losses[-1]))
    
    if n_iters:
        # check again, this time with more typical parameters
        test_gradient(model, datas.training, wd_coefficient)
    
    if do_early_stopping:
        print("Early stopping: validation loss was lowest after {} iterations. We chose the model "
            "that we had then.".format(best_so_far['after_n_iters']))
        theta = best_so_far['theta']
        model = theta_to_model(theta)
    
    # The optimization is finished. Now do some reporting.
    if n_iters:
        fig = plt.figure()
        tdl, = plt.plot(training_data_losses, 'b')
        vdl, = plt.plot(validation_data_losses, 'r')
        plt.legend([tdl, vdl], ['training', 'validation'])
        plt.ylabel('loss')
        plt.xlabel('iteration number')
        plt.show()
        plt.close(fig)
    
    datas2 = [datas.training, datas.validation, datas.test]
    data_names = ['training', 'validation', 'test']
    
    for data_i in range(3):
        data = datas2[data_i]
        data_name = data_names[data_i]
        print("\nThe loss on the {} data is {}".format(data_name, loss(model, data, wd_coefficient)))
        
        if wd_coefficient:
            print("The classification loss (i.e. without weight decay) on the {} data is {}"\
                .format(data_name, loss(model, data, 0)))
        
        print("The classification error rate on the {} data is {}".format(data_name,
            classification_performance(model, data)))


def test_gradient(model, data, wd_coefficient):
    base_theta = model_to_theta(model)
    h = 1e-2
    correctness_threshold = 1e-5
    analytic_gradient = model_to_theta(d_loss_by_d_model(model, data, wd_coefficient))
    
    # Test the gradient not for every element of theta, because that's a lot of work.
    # Test for only a few elements.
    for i in range(1, 101):
        # 1299721 is prime and thus ensures a somewhat random-like selection of indices
        test_index = i * 1299721 % base_theta.shape[0]
        analytic_here = analytic_gradient[test_index]
        theta_step = base_theta * 0
        theta_step[test_index] = h
        contribution_distances = [-4, -3, -2, -1, 1, 2, 3, 4]
        contribution_weights = [1/280, -4/105, 1/5, -4/5, 4/5, -1/5, 4/105, -1/280]
        temp = 0
        
        for contribution_index in range(8):
            m = theta_to_model(base_theta + theta_step * contribution_distances[contribution_index])
            temp += loss(m, data, wd_coefficient) * contribution_weights[contribution_index]
        
        fd_here = temp / h
        diff = abs(analytic_here - fd_here)
        # print("{} {:e} {:e} {:e} {:e}".format(test_index, base_theta[test_index], diff, fd_here, analytic_here))
        
        if diff < correctness_threshold:
            continue
        
        if diff / (np.abs(analytic_here) + np.abs(fd_here)) < correctness_threshold:
            continue
        
        err_msg = "Theta element #{}, with value {:e}, has finite difference gradient {:e} but " \
            "analytic gradient {:e}.".format(test_index, base_theta[test_index], fd_here, analytic_here)
        raise(ValueError(err_msg))
    
    # print("Gradient test passed. That means that the gradient that your code computed is within "
    #     "0.001%% of the gradient that the finite difference approximation computed, so the "
    #     "gradient calculation procedure is probably correct (not certainly, but probably).")


def logistic(input):
    return 1 / (1 + np.exp(-input))


def log_sum_exp_over_rows(a):
    # This computes log(sum(exp(a), axis=0)) in a numerically stable way
    maxs_small = np.amax(a, axis=0)
    maxs_big = np.tile(maxs_small, (a.shape[0], 1))
    return np.log(np.sum(np.exp(a - maxs_big), axis=0)) + maxs_small


def loss(model, data, wd_coefficient):
    """Returns value of a cost function augmented by weight decay term.
    
    model.input_to_hid  Matrix of size <number of hidden units> by <number of inputs i.e. 256>.
                        It contains the weights from the input units to the hidden units.

    model.hid_to_class  Matrix of size <number of classes i.e. 10> by <number of hidden units>.
                        It contains the weights from the hidden units to the softmax units.

    data.inputs         Matrix of size <number of inputs i.e. 256> by <number of data cases>.
                        Each column describes a different data case. 

    data.targets        Matrix of size <number of classes i.e. 10> by <number of data cases>.
                        Each column describes a different data case. It contains a one-of-N
                        encoding of the class, i.e. one element in every column is 1 and the
                        others are 0."""
    
    # Before we can calculate the loss, we need to calculate a variety of intermediate values,
    # like the state of the hidden units.
    
    # Input to the hidden units, i.e. before the logistic.
    # size: <number of hidden units> by <number of data cases>
    hid_input = np.dot(model.input_to_hid, data.inputs)
    # Output of the hidden units, i.e. after the logistic.
    # size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)
    # Input to the components of the softmax.
    # size: <number of classes, i.e. 10> by <number of data cases>
    class_input = np.dot(model.hid_to_class, hid_output)

    # The following three lines of code implement the softmax.
    
    # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities.
    # size: <1> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input)
    # log of probability of each class.
    # size: <number of classes, i.e. 10> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, (class_input.shape[0], 1))
    # Probability of each class. Each column (i.e. each case) sums to 1.
    # size: <number of classes, i.e. 10> by <number of data cases>
    class_prob = np.exp(log_class_prob)

    # Select the right log class probability using that sum; then take the mean over all data cases.
    classification_loss = -np.mean(np.sum(log_class_prob * data.targets, axis=0))
    # Weight decay loss. very straightforward: E = 1/2 * wd_coeffecient * theta^2
    wd_loss = np.sum(model_to_theta(model)**2) / 2 * wd_coefficient
    return classification_loss + wd_loss


def d_loss_by_d_model(model, data, wd_coefficient):
    """Returns the gradient of the loss function with respect to model weights.
    
    model.input_to_hid  Matrix of size <number of hidden units> by <number of inputs i.e. 256>
    
    model.hid_to_class  Matrix of size <number of classes i.e. 10> by <number of hidden units>
    
    data.inputs         Matrix of size <number of inputs i.e. 256> by <number of data cases>.
                        Each column describes a different data case. 
    
    data.targets        Matrix of size <number of classes i.e. 10> by <number of data cases>.
                        Each column describes a different data case. It contains a one-of-N
                        encoding of the class, i.e. one element in every column is 1 and the
                        others are 0.
    
    The returned object is supposed to be exactly like parameter <model>, i.e. it has fields
    ret.input_to_hid and ret.hid_to_class. However, the contents of those matrices are gradients
    (d loss by d model parameter), instead of model parameters.
    
    This is the only function that you're expected to change. Right now, it just returns a lot of
    zeros, which is obviously not the correct output. Your job is to replace that by a correct
    computation."""
    input_to_hid_gradient = model.input_to_hid * 0
    hid_to_class_gradient = model.hid_to_class * 0
    return Model(input_to_hid_gradient, hid_to_class_gradient)


def model_to_theta(model):
    """This function takes a model (or gradient in model form), and turns it into one long vector.
    See also theta_to_model."""
    return np.concatenate((model.input_to_hid.flatten(), model.hid_to_class.flatten()))


def theta_to_model(theta):
    """This function takes a model (or gradient) in the form of one long vector (maybe produced by
    model_to_theta), and restores it to the structure format, i.e. with fields .input_to_hid and
    .hid_to_class, both matrices."""
    n_hid = theta.shape[0] // (256 + 10)
    input_to_hid = np.reshape(theta[ : 256*n_hid], (n_hid, 256))
    hid_to_class = np.reshape(theta[256*n_hid : ], (10, n_hid))
    return Model(input_to_hid, hid_to_class)


def initial_model(n_hid):
    n_params = (256 + 10) * n_hid
    as_row_vector = np.cos(list(range(n_params)))
    return theta_to_model(as_row_vector * 0.1)


def classification_performance(model, data):
    """Returns the fraction of data cases that is incorrectly classified by the model."""
    # Input to the hidden units, i.e. before the logistic.
    # size: <number of hidden units> by <number of data cases>
    hid_input = np.dot(model.input_to_hid, data.inputs)
    # Output of the hidden units, i.e. after the logistic.
    # size: <number of hidden units> by <number of data cases>
    hid_output = logistic(hid_input)
    # Input to the components of the softmax.
    # size: <number of classes, i.e. 10> by <number of data cases>
    class_input = np.dot(model.hid_to_class, hid_output)

    choices = np.argmax(class_input, axis=0)  # choices is integer: the chosen class, plus 1
    targets = np.argmax(data.targets, axis=0) # targets is integer: the target class, plus 1
    return np.mean(np.double(choices != targets))

