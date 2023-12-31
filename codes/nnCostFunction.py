import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_par):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    #   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    #   X, y, lambda_par) computes the cost and gradient of the neural network. The
    #   parameters for the neural network are "unrolled" into the vector
    #   nn_params and need to be converted back into the weight matrices.
    #
    #   The returned parameter grad should be a "unrolled" vector of the
    #   partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in main.py

    # Input Layer
    b_input = np.ones((m, 1))
    a1 = np.concatenate((b_input, X), axis=1)
    # Hidden Layer
    z2 = np.dot(a1, Theta1.T)
    a2 = sigmoid(z2)
    b_hidden_layer = np.ones((a2.shape[0], 1))
    a2 = np.concatenate((b_hidden_layer, a2), axis=1)
    # Output Layer
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)  # = h

    # one hot y (numbers) column
    y = y.astype(int)
    y_one_hot = np.zeros((m, num_labels))
    for i in range(1, m):
        y_one_hot[i, y[i] - 1] = 1

    J = (1 / m) * np.sum(-y_one_hot * np.log(a3) - (1 - y_one_hot) * np.log(1 - a3))

    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.

    for t in range(m):
        # 1
        # Input Layer
        b_input = np.ones((1, 1))
        a1 = np.concatenate((b_input, X[t].reshape((1, -1))), axis=1)  # reshape X[t] from one dimension to two dimension after concat
        # Hidden Layer
        z2 = np.dot(a1, Theta1.T)
        a2 = sigmoid(z2)
        b_hidden_layer = np.ones((a2.shape[0], 1))
        a2 = np.concatenate((b_hidden_layer, a2), axis=1)
        # Output Layer
        z3 = np.dot(a2, Theta2.T)
        a3 = sigmoid(z3)

        # 2
        delta3 = a3 - y_one_hot[t]

        # 3
        delta2 = np.dot(delta3, Theta2[:, 1:]) * sigmoidGradient(z2)

        # 4
        Theta1_grad += np.dot(delta2.T, a1)
        Theta2_grad += np.dot(delta3.T, a2)

    # 5
    Theta1_grad /= m
    Theta2_grad /= m

    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.

    J += (lambda_par / (2 * m)) * (np.sum(np.square(Theta1[:, 1:])) + np.sum(np.square(Theta2[:, 1:])))

    theta1_zero_col = np.zeros((Theta1.shape[0], 1))  # zero col to replace bias col in theta
    Theta1_grad += (lambda_par / m) * np.concatenate((theta1_zero_col, Theta1[:, 1:]), axis=1)  # remove bias col so regularization does not affect it
    theta2_zero_col = np.zeros((Theta2.shape[0], 1))  # zero col to replace bias col in theta
    Theta2_grad += (lambda_par / m) * np.concatenate((theta2_zero_col, Theta2[:, 1:]), axis=1)  # remove bias col so regularization does not affect it

    # =========================================================================

    # Unroll gradients
    grad = np.concatenate((Theta1_grad.ravel(), Theta2_grad.ravel()), axis=0)

    return J, grad
