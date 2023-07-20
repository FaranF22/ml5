import numpy as np
from scipy.optimize import fmin_cg, minimize
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict
from sklearn.metrics import accuracy_score

## Machine Learning Course - Assignment 7: Neural Networks

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     sigmoidGradient.py
#     nnCostFunction.py
#     predict.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization

## Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)


## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = np.array([[float(x) for x in l.split(',')] for l in open('../data/dataset.csv', 'r').readlines()])
X = data[:, 0:-1]
y = data[:, -1]

m = X.shape[0]

# Randomly select 100 data points to display
sel = np.random.choice(m, 100, replace=False)

displayData(X[sel, :])

# input("Program paused. Press enter to continue.\n")


## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
Theta1 = np.array([[float(x) for x in l.split(',')] for l in open('../data/sampleTheta1.csv', 'r').readlines()])
Theta2 = np.array([[float(x) for x in l.split(',')] for l in open('../data/sampleTheta2.csv', 'r').readlines()])

# Unroll parameters 
nn_params = np.concatenate((Theta1.ravel(), Theta2.ravel()), axis=0)

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.py to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lambda_par = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par)

print('Cost at parameters (loaded from sampleTheta1 and sampleTheta2):', J, '\n(this value should be about 0.287629)\n')

# input("Program paused. Press enter to continue.\n")


## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function with Regularization ... \n')

# Weight regularization parameter (we set this to 1 here).
lambda_par = 1.

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par)

print('Cost at parameters (loaded from sampleTheta1 and sampleTheta2):', J, '\n(this value should be about 0.383770)\n')

# input("Program paused. Press enter to continue.\n")


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.py file.
#

print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print(g)
print('\n\n')

# input("Program paused. Press enter to continue.\n")


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.py)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.ravel(), initial_Theta2.ravel()), axis=0)

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.py to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients()

# input("Program paused. Press enter to continue.\n")


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('\nChecking Backpropagation with Regularization ... \n')

#  Check gradients by running checkNNGradients
lambda_par = 3.
checkNNGradients(lambda_par)

# Also output the costFunction debugging values
debug_J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par)

print('\n\nCost at (fixed) debugging parameters (with lambda = 3): ', debug_J,
      '\n(this value should be about 0.576051)\n\n')

# input("Program paused. Press enter to continue.\n")


## =================== Part 9: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
# options = optimset('MaxIter', 50)

#  You should also try different values of lambda
lambda_par = 1.

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par)[0]
gradientFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par)[1]

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
Nfeval = 1


def callbackF(Xi):
    global Nfeval
    print(Nfeval)
    Nfeval += 1


nn_params = fmin_cg(costFunction, initial_nn_params, fprime=gradientFunction, maxiter=50, disp=True, callback=callbackF)

# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[range(hidden_layer_size * (input_layer_size + 1))],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)))

# input("Program paused. Press enter to continue.\n")


## ================= Part 10: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by
#  displaying the hidden units to see what features they are capturing in
#  the data.

print('\nVisualizing Neural Network... \n')

displayData(Theta1[:, 1:])

# input("Program paused. Press enter to continue.\n")


## ================= Part 11: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print('\nTraining Set Accuracy:', accuracy_score(y, pred))
