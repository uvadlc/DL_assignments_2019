"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################


    self.params = {'weight': 0.0001 * np.random.randn(out_features, in_features), 'bias': np.zeros((out_features, 1))}
    self.grads = {'weight': np.zeros((out_features, in_features)), 'bias': np.zeros((out_features, 1))}



    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # has dim_out x dim_in
    W = self.params['weight']
    # has 1 x dim_out
    b = self.params['bias']

    # dimension batch_size x dim_out
    out = (W.dot(x.T) + b).T
    #print(dim(x.T))

    # is of dimension batch_size x dim_in
    self.x = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    W = self.params['weight']
    x = self.x

    # size (batch_size x n_l-1 ) * (batch_size x n_l ) -> ( n_l x n_l-1 )
    dL_dW = np.einsum('ij, ik -> kj', x, dout)
     # is like np.diag(a). np.einsum('ij,jh->ih', a, b) directly specifies the order of the output subscript labels     #and therefore returns matrix multiplication, unlike the example above in implicit mode..
    # getting bias gradient of shape n_l x 1 by taking mean over all batch elements

    tmp = np.mean(dout, axis=0)
    dL_db = tmp.reshape(len(tmp), 1)

    dx = dout.dot(W)

    # store the gradients again
    self.grads['weight'] = dL_dW
    self.grads['bias'] = dL_db

    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################



    self.neg_slope = neg_slope



    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # x and out are of dim batch_size x dim_out
    self.indicator_x_tilde = (x > 0)
    self.indicator_x_tilde_op = (x <= 0)
    y1 = (self.indicator_x_tilde * x)
    y2 = (self.indicator_x_tilde_op * x * self.neg_slope)

    out = y1 + y2
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # return alpha if x < 0 else 1

    # return np.array([1 if i >= 0 else alpha for i in x])

    dx_pos = dout * self.indicator_x_tilde  # +indicator_neg_slope
    dx_neg = dout * self.indicator_x_tilde_op * self.neg_slope
    # print(dx_neg)
    dx = dx_pos + dx_neg

    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # take maximum of input and reshape to dimension batch_size x 1
    x_max = np.max(x, axis=1)
    x_max = x_max.reshape(len(x_max), 1)

    e_x = np.exp(x - x_max)

    # take sum and reshape to dimension batch_size x 1
    sum_e_x = e_x.sum(axis=1)
    sum_e_x = sum_e_x.reshape(len(sum_e_x), 1)

    # store x_N in module
    out = e_x / sum_e_x
    self.x_N = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x_N = self.x_N

    # see https://stackoverflow.com/questions/26511401/numpy-fastest-way-of-computing-diagonal-for-each-row-of-a-2d-array
    batch_size = x_N.shape[0]
    n_classes = x_N.shape[1]

    # create matrix of shape batch_size x n_classes x n_classes
    diagonal_matrix = np.zeros((batch_size, n_classes, n_classes))
    diag = np.arange(n_classes)
    diagonal_matrix[:, diag, diag] = x_N

    # result is of shape batch_size x n_classes x n_classes
    dx_dx_tilde = diagonal_matrix - np.einsum('ij, ik -> ijk', x_N, x_N)

    # (batchs_size x n_classes) * (batch_size x n_classes x n_classes) - > (batchs_size x n_classes)
    dx = np.einsum('ij, ijk -> ik', dout, dx_dx_tilde)
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    eps = 1e-6  # numerical stability

    # result is of size batch_size x 1
    loss_per_datapoint = - np.sum(y * np.log(x + eps), axis=1)

    # average over datapoints in batch
    out = np.mean(loss_per_datapoint)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    eps = 1e-6  # numerical stability

    dx = -np.divide(y, x + eps) / len(y)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx