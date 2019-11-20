"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    '''
       Functions that we will use for CNN implementation
       nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
       nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
       nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
       '''
    # as super class
    super(ConvNet, self).__init__()

    # define parameters
    out_channels = {
      "conv1": 64,
      "conv2": 128,
      "conv3": 256,
      "conv4": 512,
      "conv5": 512,
      "avg_pool": 512
    }

    kernel_size = 3  # for conv and max pool layers
    kernel_size_avg = 1  # only for last avg pooling layer
    padding = 1  # for all conv and max pool layer
    padding_avg = 0  # for last avg pooling layer
    stride = 1
    stride_max = 2

    # define network via sequential method
    self.network = nn.Sequential(

      # Block 1
      nn.Conv2d(in_channels=n_channels, out_channels=out_channels['conv1'], kernel_size=kernel_size, stride=stride,
                padding=padding),
      nn.BatchNorm2d(num_features=out_channels['conv1']),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride_max, padding=padding),
      nn.ReLU(),

      # Block 2
      nn.Conv2d(in_channels=out_channels['conv1'], out_channels=out_channels['conv2'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels['conv2']),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride_max, padding=padding),
      nn.ReLU(),

      # Block 3
      nn.Conv2d(in_channels=out_channels['conv2'], out_channels=out_channels['conv3'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.Conv2d(in_channels=out_channels['conv3'], out_channels=out_channels['conv3'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels['conv3']),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride_max, padding=padding),
      nn.ReLU(),

      # Block 4
      nn.Conv2d(in_channels=out_channels['conv3'], out_channels=out_channels['conv4'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.Conv2d(in_channels=out_channels['conv4'], out_channels=out_channels['conv4'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels['conv4']),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride_max, padding=padding),
      nn.ReLU(),

      # Block 5
      nn.Conv2d(in_channels=out_channels['conv4'], out_channels=out_channels['conv5'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.Conv2d(in_channels=out_channels['conv5'], out_channels=out_channels['conv5'], kernel_size=kernel_size,
                stride=stride, padding=padding),
      nn.BatchNorm2d(num_features=out_channels['conv5']),
      nn.MaxPool2d(kernel_size=kernel_size, stride=stride_max, padding=padding),
      nn.ReLU(),

      # avg pooling
      nn.AvgPool2d(kernel_size=kernel_size_avg, stride=stride, padding=padding_avg),
    )

    # fully connected layer
    self.linear_layer = nn.Linear(in_features=out_channels['conv5'], out_features=n_classes)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # forward pass for all conv layers
    X_N_minus_1 = self.network(x)

    # X_N-1 is of dimension
    # torch.Size([BATCH_SIZE, 512, 1, 1])
    # = batch_size X output_channels last layer

    # reshape for fully connected layer
    BATCH_SIZE = X_N_minus_1.shape[0]
    X_N_minus_1 = X_N_minus_1.reshape(BATCH_SIZE, -1)

    # forward through linear layer
    out = self.linear_layer(X_N_minus_1)
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
