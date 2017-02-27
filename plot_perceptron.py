import matplotlib.pyplot as plt
import numpy as np

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
