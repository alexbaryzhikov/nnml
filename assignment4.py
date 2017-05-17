"""
Neural Networks for Machine Learning

Programming Assignment 4: Restricted Boltzmann Machines

This assignment is about Restricted Boltzmann Machines (RBMs). We'll first make a few basic
functions for dealing with RBMs, and then we'll train an RBM. We'll use it as the visible-to-hidden
layer in a network exactly like the one we made in programming assignment 3 (PA3).
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


class Data:
    def __init__(self, datas):
        self.training   = Group(datas['training'][0,0]['inputs'], datas['training'][0,0]['targets'])
        self.validation = Group(datas['validation'][0,0]['inputs'], datas['validation'][0,0]['targets'])
        self.test       = Group(datas['test'][0,0]['inputs'], datas['test'][0,0]['targets'])


class Group:
    def __init__(self, inputs, targets):
        self.inputs  = inputs
        self.targets = targets


# -----------------------------------------------------------------------------
# Initialization

randomness_source = None
data_sets = None
report_calls_to_sample_bernoulli = None


def a4_init():
    # Load array of pseudo random numbers
    global randomness_source
    randomness_source = sio.loadmat('a4_randomness_source.mat')['randomness_source']
    
    # Load data: arrays of 16x16 images of greyscale hand-written digits
    global data_sets
    data_sets = Data(sio.loadmat('data_set.mat')['data'][0,0])  # same as in PA3
    
    global report_calls_to_sample_bernoulli
    report_calls_to_sample_bernoulli = False

    test_rbm_w       = a4_rand([100, 256], 0) * 2 - 1
    small_test_rbm_w = a4_rand([ 10, 256], 0) * 2 - 1

    temp          = extract_mini_batch(data_sets.training,   0,  1)
    data_1_case   = sample_bernoulli(temp.inputs)
    temp          = extract_mini_batch(data_sets.training,  99, 10)
    data_10_cases = sample_bernoulli(temp.inputs)
    temp          = extract_mini_batch(data_sets.training, 199, 37)
    data_37_cases = sample_bernoulli(temp.inputs)

    test_hidden_state_1_case   = sample_bernoulli(a4_rand([100,  1], 0))
    test_hidden_state_10_cases = sample_bernoulli(a4_rand([100, 10], 1))
    test_hidden_state_37_cases = sample_bernoulli(a4_rand([100, 37], 2))

    report_calls_to_sample_bernoulli = True

    del temp


def a4_rand(requested_size, seed):
    global randomness_source
    start_i = int(round(seed) % round(randomness_source.shape[1] / 10))
    end_i   = start_i + np.prod(requested_size)
    if end_i >= randomness_source.shape[1]:
        raise ValueError('failed to generate an array of that size (too big)')
    return np.reshape(randomness_source[:, start_i : end_i], requested_size, order='F')


def extract_mini_batch(data_set, start_i, n_cases):
    inputs  = data_set.inputs[:, start_i : start_i + n_cases]
    targets = data_set.targets[:, start_i : start_i + n_cases]
    return Group(inputs, targets)


def sample_bernoulli(probabilities):
    global report_calls_to_sample_bernoulli
    if report_calls_to_sample_bernoulli:
        print('sample_bernoulli() was called with a matrix of size {} by {}.'.format(*probabilities.shape))
    seed = np.sum(probabilities)
    # the "1*" is to avoid the "logical" data type, which just confuses things.
    binary = 1 * (probabilities > a4_rand(probabilities.shape, seed))
    return binary


def describe_matrix(matrix):
    print('Describing a matrix of size {} by {}. The mean of the elements is {}.'
          'The sum of the elements is {}'.format(matrix.shape[0], matrix.shape[1], \
          np.mean(matrix), np.sum(matrix)))


# -----------------------------------------------------------------------------
# Main

def a4_main(n_hid, lr_rbm, lr_classification, n_iterations):
    # first, train the rbm
    global report_calls_to_sample_bernoulli
    report_calls_to_sample_bernoulli = False
    global data_sets
    if data_sets is None:
        raise BaseException('You must run a4_init before you do anything else.')

    rbm_w = optimize([n_hid, 256],
                     lambda rbm_w, data: cd1(rbm_w, data.inputs),  # discard labels
                     data_sets.training,
                     lr_rbm,
                     n_iterations)

    # rbm_w is now a weight matrix of <n_hid> by <number of visible units, i.e. 256>
    show_rbm(rbm_w)
    input_to_hid = rbm_w
    # calculate the hidden layer representation of the labeled data
    hidden_representation = logistic(np.dot(input_to_hid, data_sets.training.inputs))
    # train hid_to_class
    inputs  = hidden_representation
    targets = data_sets.training.targets
    data_2  = Group(inputs, targets)

    hid_to_class = optimize([10, n_hid],
                            lambda model, data: classification_phi_gradient(model, data),
                            data_2,
                            lr_classification,
                            n_iterations)

    # report results
    data_items = (('training',   data_sets.training),
                  ('validation', data_sets.validation),
                  ('test',       data_sets.test))
    
    for data_details in data_items:
        data_name = data_details[0]
        data = data_details[1]
        # size: <number of hidden units> by <number of data cases>
        hid_input = np.dot(input_to_hid, data.inputs)
        # size: <number of hidden units> by <number of data cases>
        hid_output = logistic(hid_input)
        # size: <number of classes> by <number of data cases>
        class_input = np.dot(hid_to_class, hid_output)
        # log(sum(exp of class_input)) is what we subtract to get properly normalized log class probabilities.
        # size: <1> by <number of data cases>
        class_normalizer = log_sum_exp_over_rows(class_input)
        # log of probability of each class.
        # size: <number of classes, i.e. 10> by <number of data cases>
        log_class_prob = class_input - np.tile(class_normalizer, [class_input.shape[0], 1])
        # scalar
        error_rate = np.mean(argmax_over_rows(class_input) != argmax_over_rows(data.targets))
        # scalar. Select the right log class probability using that sum, then take the mean over all data cases.
        loss = -np.mean(np.sum(log_class_prob * data.targets, axis=0))
        print('For the {} data:\n'
              '    classification cross-entropy loss is {}\n'
              '    classification error rate (i.e. the misclassification rate) is {}\n' \
              .format(data_name, loss, error_rate))

    report_calls_to_sample_bernoulli = True


def classification_phi_gradient(input_to_class, data):
    # This is about a very simple model: there's an input layer, and a softmax output layer. There
    # are no hidden layers, and no biases.
    # This returns the gradient of phi (a.k.a. negative the loss) for the <input_to_class> matrix.
    # <input_to_class> is a matrix of size <number of classes> by <number of input units>.
    # <data> has fields 'inputs' (matrix of size <number of input units> by <number of data cases>)
    # and 'targets' (matrix of size <number of classes> by <number of data cases>).
    
    # FORWARD PASS
    # input to the components of the softmax.
    # size: <number of classes> by <number of data cases>
    class_input = np.dot(input_to_class, data.inputs)
    # log(sum(exp)) is what we subtract to get normalized log class probabilities.
    # size: <1> by <number of data cases>
    class_normalizer = log_sum_exp_over_rows(class_input)
    # log of probability of each class.
    # size: <number of classes> by <number of data cases>
    log_class_prob = class_input - np.tile(class_normalizer, [class_input.shape[0], 1])
    # probability of each class. Each column (i.e. each case) sums to 1.
    # size: <number of classes> by <number of data cases>
    class_prob = np.exp(log_class_prob)

    # GRADIENT COMPUTATION
    # size: <number of classes> by <number of data cases>
    d_loss_by_d_class_input = -(data.targets - class_prob) / data.inputs.shape[1]
    # size: <number of classes> by <number of input units>
    d_loss_by_d_input_to_class = np.dot(d_loss_by_d_class_input, data.inputs.T)
    d_phi_by_d_input_to_class = -d_loss_by_d_input_to_class
    return d_phi_by_d_input_to_class


def argmax_over_rows(matrix):
    return np.argmax(matrix, axis=0)


def log_sum_exp_over_rows(matrix):
    # This computes log(sum(exp(a), 1)) in a numerically stable way
    maxs_small = np.amax(matrix, axis=0)
    maxs_big = np.tile(maxs_small, [matrix.shape[0], 1])
    return np.log(np.sum(np.exp(matrix - maxs_big), axis=0)) + maxs_small


def logistic(input):
    return 1 / (1 + np.exp(-input))


def show_rbm(rbm_w):
    n_hid = rbm_w.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_hid)))
    blank_lines = 4
    distance = 16 + blank_lines
    to_show = np.zeros([n_rows * distance + blank_lines, n_rows * distance + blank_lines])

    for i in range(n_hid):
        row_i = int(np.floor(i / n_rows))
        col_i = i % n_rows
        pixels = np.reshape(rbm_w[i, :], [16, 16])
        row_base = row_i * distance + blank_lines
        col_base = col_i * distance + blank_lines
        to_show[row_base : row_base + 16, col_base : col_base + 16] = pixels

    plt.imshow(to_show, cmap='gray')
    plt.title('hidden units of the RBM')
    plt.show()


# -----------------------------------------------------------------------------
# Optimization

def optimize(model_shape, gradient_function, training_data, learning_rate, n_iterations):
    # This trains a model that's defined by a single matrix of weights.
    # <model_shape> is the shape of the array of weights.
    # <gradient_function> is a function that takes parameters <model> and <data> and returns the
    # gradient (or approximate gradient in the case of CD-1) of the function that we're maximizing.
    # Note the contrast with the loss function that we saw in PA3, which we were minimizing. The
    # returned gradient is an array of the same shape as the provided <model> parameter.
    # This uses mini-batches of size 100, momentum of 0.9, no weight decay, and no early stopping.
    # This returns the matrix of weights of the trained model.
    model = (a4_rand(model_shape, np.prod(model_shape)) * 2 - 1) * 0.1
    momentum_speed = np.zeros(model_shape)
    mini_batch_size = 100
    mini_batch_start = 0

    for iteration_number in range(n_iterations):
        mini_batch = extract_mini_batch(training_data, mini_batch_start, mini_batch_size)
        mini_batch_start = (mini_batch_start + mini_batch_size) % training_data.inputs.shape[1]
        gradient = gradient_function(model, mini_batch)
        momentum_speed = 0.9 * momentum_speed + gradient
        model = model + momentum_speed * learning_rate

    return model


# -----------------------------------------------------------------------------
# TODO

def cd1(rbm_w, visible_data):
    # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    # <visible_data> is a (possibly but not necessarily binary) matrix of size
    #   <number of visible units> by <number of data cases>
    # The returned value is the gradient approximation produced by CD-1.
    # It's of the same shape as <rbm_w>.
    raise NotImplementedError()
    return ret


def configuration_goodness(rbm_w, visible_state, hidden_state):
    # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    # <visible_state> is a binary matrix of size
    #   <number of visible units> by <number of configurations that we're handling in parallel>.
    # <hidden_state> is a binary matrix of size
    #   <number of hidden units> by <number of configurations that we're handling in parallel>.
    # This returns a scalar: the mean over cases of the goodness (negative energy) of the described
    # configurations.
    raise NotImplementedError()
    return G


def configuration_goodness_gradient(visible_state, hidden_state):
    # <visible_state> is a binary matrix of size
    #   <number of visible units> by <number of configurations that we're handling in parallel>.
    # <hidden_state> is a (possibly but not necessarily binary) matrix of size
    #   <number of hidden units> by <number of configurations that we're handling in parallel>.
    # You don't need the model parameters for this computation.
    # This returns the gradient of the mean configuration goodness (negative energy, as computed by
    # function configuration_goodness) with respect to the model parameters. Thus, the returned
    # value is of the same shape as the model parameters, which by the way are not provided to this
    # function. Notice that we're talking about the mean over data cases (as opposed to the sum
    # over data cases).
    raise NotImplementedError()
    return d_G_by_rbm_w


def hidden_state_to_visible_probabilities(rbm_w, hidden_state):
    # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    # <hidden_state> is a binary matrix of size
    #   <number of hidden units> by <number of configurations that we're handling in parallel>.
    # The returned value is a matrix of size
    #   <number of visible units> by <number of configurations that we're handling in parallel>.
    # This takes in the (binary) states of the hidden units, and returns the activation
    # probabilities of the visible units, conditional on those states.
    raise NotImplementedError()
    return visible_probability


def visible_state_to_hidden_probabilities(rbm_w, visible_state):
    # <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
    # <visible_state> is a binary matrix of size
    #   <number of visible units> by <number of configurations that we're handling in parallel>.
    # The returned value is a matrix of size
    #   <number of hidden units> by <number of configurations that we're handling in parallel>.
    # This takes in the (binary) states of the visible units, and returns the activation
    # probabilities of the hidden units conditional on those states.
    raise NotImplementedError()
    return hidden_probability

