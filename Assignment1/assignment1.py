"""
Neural Networks for Machine Learning

Programming Assignment 1: The perceptron learning algorithm

In this assignment you will take the provided starter code and fill in the missing details in order
to create a working perceptron implementation.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

DATA_FILE = 'dataset1.mat'

# -----------------------------------------------------------------------------
# Learn perceptron

def learn_perceptron():
    """Learns the weights of a perceptron for a 2-dimensional dataset and plots
    the perceptron at each iteration where an iteration is defined as one
    full pass through the data. If a generously feasible weight vector
    is provided then the visualization will also show the distance
    of the learned weight vectors to the generously feasible weight vector.

    Returns:
      w - The learned weight vector."""

    # Read data
    data = scipy.io.loadmat(DATA_FILE)
    # The num_neg_examples x 2 matrix for the examples with target 0.
    # num_neg_examples is the number of examples for the negative class.
    neg_examples_nobias = data['neg_examples_nobias']
    # The num_pos_examples x 2 matrix for the examples with target 1.
    # num_pos_examples is the number of examples for the positive class.
    pos_examples_nobias = data['pos_examples_nobias']
    # A 3-dimensional initial weight vector. The last element is the bias.
    w_init = data['w_init']
    # A generously feasible weight vector.
    w_gen_feas = data['w_gen_feas']

    # Bookkeeping
    num_neg_examples = neg_examples_nobias.shape[0]
    num_pos_examples = pos_examples_nobias.shape[0]
    num_err_history = []
    w_dist_history = []

    # Here we add a column of ones to the examples in order to allow us to learn
    # bias parameters.
    neg_examples = np.append(neg_examples_nobias, np.ones((num_neg_examples, 1)), axis=1)
    pos_examples = np.append(pos_examples_nobias, np.ones((num_pos_examples, 1)), axis=1)

    # If weight vectors have not been provided, initialize them appropriately.
    if w_init.size:
        w = w_init
    else:
        w = np.random.randn(3,1)

    # Find the data points that the perceptron has incorrectly classified
    # and record the number of errors it makes.
    iter = 0
    mistakes0, mistakes1 = eval_perceptron(neg_examples, pos_examples, w)
    num_errs = len(mistakes0) + len(mistakes1)
    num_err_history.append(num_errs)
    print("Number of errors in iteration {}:\t{}".format(iter, num_errs))
    print("weights:\t", *(float(i) for i in w))
    plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, \
                    w, w_dist_history)
    key = input('Press enter to continue, q to quit. >> ')
    if 'q' in key:
        return w
    
    # If a generously feasible weight vector exists, record the distance
    # to it from the initial weight vector.
    if w_gen_feas.size:
        w_dist_history.append(np.linalg.norm(w - w_gen_feas))

    # Iterate until the perceptron has correctly classified all points.
    while num_errs > 0:
        iter += 1

        # Update the weights of the perceptron.
        w = update_weights(neg_examples, pos_examples, w)

        # If a generously feasible weight vector exists, record the distance
        # to it from the initial weight vector.
        if w_gen_feas.size:
            w_dist_history.append(np.linalg.norm(w - w_gen_feas))

        # Find the data points that the perceptron has incorrectly classified.
        # and record the number of errors it makes.
        mistakes0, mistakes1 = eval_perceptron(neg_examples, pos_examples, w)
        num_errs = len(mistakes0) + len(mistakes1)
        num_err_history.append(num_errs)

        print("Number of errors in iteration {}:\t{}".format(iter, num_errs))
        print("weights:\t", *(float(i) for i in w))
        plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, num_err_history, \
                        w, w_dist_history)
        key = input('Press enter to continue, q to quit. >> ')
        if 'q' in key:
            break

    return w


# WRITE THE CODE TO COMPLETE THIS FUNCTION
def update_weights(neg_examples, pos_examples, w_current):
    """Updates the weights of the perceptron for incorrectly classified points
    using the perceptron update algorithm. This function makes one sweep
    over the dataset.
    
    Inputs:
      neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples - The num_pos_examples x 3 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      w_current    - A 3-dimensional weight vector, the last element is the bias.
    
    Returns:
      w - The weight vector after one pass through the dataset using the perceptron
          learning rule."""

    w = w_current
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]

    for i in range(num_neg_examples):
        this_case = np.array(neg_examples[i], ndmin=2)
        x = this_case.T # Hint
        activation = np.dot(this_case, w)
        if activation >= 0:
            # YOUR CODE HERE

    for i in range(num_pos_examples):
        this_case = np.array(pos_examples[i], ndmin=2)
        x = this_case.T # Hint
        activation = np.dot(this_case, w)
        if activation < 0:
            # YOUR CODE HERE

    return w

def eval_perceptron(neg_examples, pos_examples, w):
    """Evaluates the perceptron using a given weight vector. Here, evaluation
    refers to finding the data points that the perceptron incorrectly classifies.

    Inputs:
      neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples - The num_pos_examples x 3 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      w            - A 3-dimensional weight vector, the last element is the bias.

    Returns:
      mistakes0 - A vector containing the indices of the negative examples that have been
          incorrectly classified as positive.
      mistakes1 - A vector containing the indices of the positive examples that have been
          incorrectly classified as negative."""

    mistakes0, mistakes1 = [], []
    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]

    for i in range(num_neg_examples):
        x = np.array(neg_examples[i], ndmin=2)
        activation = np.dot(x, w)
        if activation >= 0:
            mistakes0.append(i)

    for i in range(num_pos_examples):
        x = np.array(pos_examples[i], ndmin=2)
        activation = np.dot(x, w)
        if activation < 0:
            mistakes1.append(i)

    return mistakes0, mistakes1


# -----------------------------------------------------------------------------
# Plot perceptron

def plot_perceptron(neg_examples, pos_examples, mistakes0, mistakes1, \
                    num_err_history, w, w_dist_history):
    """Plots information about a perceptron classifier on a 2-dimensional dataset.
    
    The top-left plot shows the dataset and the classification boundary given by
    the weights of the perceptron. The negative examples are shown as circles
    while the positive examples are shown as squares. If an example is colored
    green then it means that the example has been correctly classified by the
    provided weights. If it is colored red then it has been incorrectly classified.
    The top-right plot shows the number of mistakes the perceptron algorithm has
    made in each iteration so far.
    The bottom-left plot shows the distance to some generously feasible weight
    vector if one has been provided (note, there can be an infinite number of these).
    Points that the classifier has made a mistake on are shown in red,
    while points that are correctly classified are shown in green.
    The goal is for all of the points to be green (if it is possible to do so).
    
    Inputs:
      neg_examples - The num_neg_examples x 3 matrix for the examples with target 0.
          num_neg_examples is the number of examples for the negative class.
      pos_examples- The num_pos_examples x 3 matrix for the examples with target 1.
          num_pos_examples is the number of examples for the positive class.
      mistakes0 - A vector containing the indices of the datapoints from class 0 incorrectly
          classified by the perceptron. This is a subset of neg_examples.
      mistakes1 - A vector containing the indices of the datapoints from class 1 incorrectly
          classified by the perceptron. This is a subset of pos_examples.
      num_err_history - A vector containing the number of mistakes for each
          iteration of learning so far.
      w - A 3-dimensional vector corresponding to the current weights of the
          perceptron. The last element is the bias.
      w_dist_history - A vector containing the L2-distance to a generously
          feasible weight vector for each iteration of learning so far.
          Empty if one has not been provided."""

    f = plt.figure(figsize=(15,10))
    f.clf()

    num_neg_examples = neg_examples.shape[0]
    num_pos_examples = pos_examples.shape[0]
    neg_correct_ind = np.setdiff1d(np.arange(num_neg_examples), mistakes0)
    pos_correct_ind = np.setdiff1d(np.arange(num_pos_examples), mistakes1)

    sp1 = f.add_subplot(221)
    if neg_examples.size:
        sp1.plot(neg_examples[neg_correct_ind, 0], neg_examples[neg_correct_ind, 1], "og", markersize=20)
    if pos_examples.size:
        sp1.plot(pos_examples[pos_correct_ind, 0], pos_examples[pos_correct_ind, 1], "sg", markersize=20)
    if len(mistakes0):
        sp1.plot(neg_examples[mistakes0, 0], neg_examples[mistakes0, 1], "or", markersize=20)
    if len(mistakes1):
        sp1.plot(pos_examples[mistakes1, 0], pos_examples[mistakes1, 1], "sr", markersize=20)
    sp1.set_title("Classifier")

    # In order to plot the decision line, we just need to get two points.
    sp1.plot([-5, 5], [(- w[-1] + 5 * w[0]) / w[1], (- w[-1] - 5 * w[0]) / w[1]], 'k')
    sp1.set_xlim(-1, 1);
    sp1.set_ylim(-1, 1);

    sp2 = f.add_subplot(222)
    sp2.plot(range(len(num_err_history)), num_err_history)
    sp2.set_xlim(-1, max(15, len(num_err_history)))
    sp2.set_ylim(0, num_neg_examples + num_pos_examples + 1)
    sp2.set_title("Number of errors")
    sp2.set_xlabel("Iteration")
    sp2.set_ylabel("Number of errors")

    sp3 = f.add_subplot(223)
    sp3.plot(range(len(w_dist_history)), w_dist_history)
    sp3.set_xlim(-1, max(15, len(w_dist_history)))
    if len(w_dist_history):
        sp3.set_ylim(0, max(w_dist_history) + 1)
    else:
        sp3.set_ylim(0, 15)
    sp3.set_title("Distance")
    sp3.set_xlabel("Iteration")
    sp3.set_ylabel("Distance")

    f.savefig("learning_plot")
    plt.close(f)


# -----------------------------------------------------------------------------
# Example usage

# learn_perceptron()
