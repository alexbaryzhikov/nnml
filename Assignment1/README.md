### Neural Networks for Machine Learning
### Programming Assignment 1 : The perceptron learning algorithm

In this assignment you will take the provided starter code and fill in the missing details in order to create a working perceptron implementation.

To start, download the following code files:
```
assignment1.py
```

And the following datasets:
```
dataset1.mat
dataset2.mat
dataset3.mat
dataset4.mat
```

You can run the algorithm by entering the following at the Python console:
```
w = learn_perceptron()
```

This will start the algorithm and plot the results as it proceeds. Until the algorithm converges you can keep pressing enter to run the next iteration. Pressing 'q' will terminate the program. At each iteration it should produce a plot that looks something like this.

The top left plot shows the data points. The circles represent one class while the squares represent the other. The line shows the decision boundary of the perceptron using the current set of weights. The green examples are those that are correctly classified while the red are incorrectly classified. The top-right plot will show the number of mistakes made by the perceptron. If a generously feasible weight vector is provided (and not empty), then the bottom left plot will show the distance of the learned weight vectors to the generously feasible weight vector.

Currently, the code doesn't do any learning. It is your job to fill this part in. Specifically, you need to fill in the lines  marked %YOUR CODE HERE. When you are finished, use this program to help you answer the questions below.

DISCLAIMER: Before beginning the actual quiz portion of the assignment, please read the corresponding Reading.

1) Which of the provided datasets are linearly separable?

> Dataset 1  
> Dataset 2  
> Dataset 3  
> Dataset 4  

2) True or false: if the dataset is not linearly separable, then it is possible for the number of classification errors to increase during learning.

> True  
> False  

3) True or false: If a generously feasible region exists, then the distance between the current weight vector and a weight vector in the generously feasible region will monotonically decrease as the learning proceeds.

> True  
> False  

4) The perceptron algorithm as implemented and described in class implicitly uses a learning rate of 1. We can modify the algorithm to use a different learning rate α so that the update rule for an input x and target t becomes:  
```
w^t ← w^(t−1) + α(t−prediction)x
```
   where prediction is the decision made by the perceptron using the current weight vector w(t−1), given by:  
```
prediction = { 1 if w^T, x≥00 otherwise }
```
   True or false: if we use a learning rate of 0.5, then the perceptron algorithm will always converge to a solution for linearly separable datasets.  

> True  
> False  

5) According to the code, how many iterations does it take for the perceptron to converge to a solution on dataset 1 using the provided initial weight vector w_init?

   Note: the program will output `Number of errors in iteration x:	0`  
   You simply need to report x.

> 5  
> It doesn't converge.  
> 1  
> 3  
