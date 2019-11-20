"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import sys
import pandas as pd

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '500,500,500'  #'512,512,512,512'
LEARNING_RATE_DEFAULT = 3e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 250
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.01
OPTIMIZER_CHOICE='Adam'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################

  predictions = predictions.argmax(dim=1)
  targets = targets.argmax(dim=1)

  # check if predicted and actual label are equal
  correct_predictions = (predictions == targets).numpy()

  # calculate accuracy
  accuracy = np.mean(correct_predictions)

  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  
  ########################
  # PUT YOUR CODE HERE  #
  #######################
  LEARNING_RATE_DEFAULT = FLAGS.learning_rate
  MAX_STEPS_DEFAULT = FLAGS.max_steps
  BATCH_SIZE_DEFAULT = FLAGS.batch_size
  EVAL_FREQ_DEFAULT = FLAGS.eval_freq
  NEG_SLOPE_DEFAULT = FLAGS.neg_slope
  OPTIMIZER_CHOICE=FLAGS.optimizer

  # load the data
  cifar10 = cifar10_utils.get_cifar10()

  # load test set
  ## load test data // we need it anyway for evaluation
  x, y = cifar10['test'].images, cifar10['test'].labels
  # make numpy arrays pytorch tensors
  x_test = torch.tensor(x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
  y_test = torch.tensor(y, dtype=torch.long)

  # load train set
  ## load test data // we need it anyway for evaluation
  x, y = cifar10['train'].images, cifar10['train'].labels
  # make numpy arrays pytorch tensors
  x_train_whole = torch.tensor(x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
  y_train_whole = torch.tensor(y, dtype=torch.long)

  # get parameters of network
  n_classes = y_test.shape[1]
  n_inputs = x_test.shape[1]

  # initialize network
  net = MLP(n_inputs, dnn_hidden_units, n_classes, neg_slope)

  # initialize loss // already contains softmax layer
  loss_fn = torch.nn.CrossEntropyLoss()

  # initiliaze optimizier
  if OPTIMIZER_CHOICE == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT) #, weight_decay=REGULARIZER
  elif OPTIMIZER_CHOICE == 'SDG':
    optimizer=optim.SGD(net.parameters(), lr=LEARNING_RATE_DEFAULT)
  else:
    print('Not a valid optimizer')

  # list for evalation metrics
  acc_train_list = []
  acc_test_list = []
  loss_train_list = []
  loss_test_list = []
  # print(MAX_STEPS_DEFAULT)
  # print(EVAL_FREQ_DEFAULT)
  # loop over whole training set several times
  for step in range(MAX_STEPS_DEFAULT):

    # load current batch
    x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

    # reshape and make it pytorch tensor
    x_train = torch.tensor(x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    y_train = torch.tensor(y, dtype=torch.long)

    # zero the gradients
    optimizer.zero_grad()

    # do one forward pass
    x_N_train = net.forward(x_train)

    # calculate the loss
    loss = loss_fn(x_N_train, y_train.argmax(dim=1))

    # get gradients with respect to that loss
    loss.backward()

    # actual optimizing step
    optimizer.step()

    if (step + 1) % EVAL_FREQ_DEFAULT == 0:
      print("Step", step + 1)
      # loss and accuracy for test set
      x_N_test = net.forward(x_test)
      acc_test = accuracy(x_N_test, y_test)
      loss_test = loss_fn(x_N_test, y_test.argmax(dim=1)).item()
      print('Accuracy', acc_test)

      # loss and accuracy of train set
      x_N_train_whole = net.forward(x_train_whole)
      acc_train = accuracy(x_N_train_whole, y_train_whole)
      loss_train = loss_fn(x_N_train_whole, y_train_whole.argmax(dim=1)).item()

      acc_train_list.append(acc_train)
      acc_test_list.append(acc_test)
      loss_train_list.append(loss_train)
      loss_test_list.append(loss_test)
      print('Loss', loss_test)

  folder = "./pytorch_results/"
  print("Save results")
  np.save(folder + "loss_train", loss_train_list)
  np.save(folder + "acc_train", acc_train_list)
  np.save(folder + "loss_test", loss_test_list)
  np.save(folder + "acc_test", acc_test_list)

  activation='Leaky Relu'
  loss_test=round(loss_test,3)
  acc_test=round(acc_test, 4)

  data = [[activation, DNN_HIDDEN_UNITS_DEFAULT, LEARNING_RATE_DEFAULT, NEG_SLOPE_DEFAULT, BATCH_SIZE_DEFAULT, OPTIMIZER_CHOICE, acc_test, loss_test]]

  df = pd.DataFrame(data, columns=['Activation','DNN_hidden_units', 'Learning Rate', 'Neg_Slope',
                                   'Batch_size', 'Optimizer', 'Accuracy', 'Loss'])

  f = 'results_pytorch.txt'
  df.to_csv(folder + f, header=None, index=None, mode='a', sep=' ')


  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  parser.add_argument('--optimizer', type=str, default= OPTIMIZER_CHOICE, help= 'Optimizer')
  FLAGS, unparsed = parser.parse_known_args()

  main()