from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]  # The number of classes in the dataset
    num_train = X.shape[0]  # The number of images in the training batch X
    scores = np.zeros((num_train, num_classes))
    
    # Insert few lines of code here to calculate the scores as we usually do with linear classifiers
    for ii in range(num_train):
        scores[ii] = X[ii].dot(W)
        
    for i in range(num_train):
        f = scores[i]-np.max(scores[i]) #shift the scores so the highest value inside f is 0 for numeric stability (according to cs231n notes)
        softmax = np.exp(f) / np.sum(np.exp(f))  # Use the formula for softmax given in the lecture             
        loss += -np.log(softmax[y[i]]) # Only the -log of the softmax of the correct class adds to the softmax loss
        for j in range(num_classes):
            dW[:,j] += X[i] * softmax[j] #gradient for all classes
        dW[:, y[i]] -= X[i] #gradient for the correct class gets adjusted

    # We now have the total loss & gradient, however we need the average over the whole training set
    loss /= num_train
    dW /= num_train

    # We also need to add regularization to the loss and the gradients
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]  # Number of training images
    scores = X.dot(W) # Calculate the scores as we are used to doing it
    maxscores = np.amax(scores, axis=1)
    #maxscores.reshape((500, 1))            #Virker ikke..
    maxscores = maxscores[:, np.newaxis]    #Numpy broadcasting magi til at den gider trække maksværdien fra alle rækker..
    scores = scores-maxscores               #Shift the scores so the highest value is 0 for numeric stability
    
    # Calculate the scores after the softmax function as shown in the lecture (you may need multiple lines of code)
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    loss = -np.sum(np.log(softmax[np.arange(num_train),y])) # Only the -log of the softmax of the correct class adds to the softmax loss

    #Gradient calculations
    softmax[np.arange(num_train),y] -= 1 #subtract 1 from all the softmax scores of the correct class
    dW = X.T.dot(softmax) #dot prod. between the transposed input array X and the softmax scores

    # We now have the total loss & gradient, however we need the average over the whole training set
    loss /= num_train
    dW /= num_train

    # We also need to add regularization to the loss and the gradients
    loss += reg * np.sum(W * W)
    
    # Adding regularization to the gradients
    dW = dW + reg*2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
