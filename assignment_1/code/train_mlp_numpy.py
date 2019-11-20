"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

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
  # get's index of greatest highest value; so the actual predicted class
  predictions = np.argmax(predictions, axis=1)
  targets = np.argmax(targets, axis=1)

  # check if predicted and actual label are equal
  correct_predictions = (predictions == targets)

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
  cifar10 = cifar10_utils.get_cifar10()

  ## load test data // we need it anyway for evaluation
  x, y = cifar10['test'].images, cifar10['test'].labels
  # reshape images to (batch_size x input_size)
  x_test = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
  y_test = y

  ## load whole train set // we need it anyway for evaluation
  x, y = cifar10['train'].images, cifar10['train'].labels
  # reshape images to (batch_size x input_size)
  x_train_whole = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
  y_train_whole = y

  n_classes = y_test.shape[1]
  n_inputs = x_test.shape[1]

  # initialize numpy MLP and loss
  numpy_MLP = MLP(n_inputs=n_inputs, n_hidden=dnn_hidden_units, n_classes=n_classes, neg_slope=neg_slope)
  cross_entropy_loss = CrossEntropyModule()

  # list for evalation metrics
  acc_train_list = []
  acc_test_list = []
  loss_train_list = []
  loss_test_list = []

  for step in range(MAX_STEPS_DEFAULT):
    # load train set
    x, y_train = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
    x_train = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

    # forward through everything except loss
    x_N = numpy_MLP.forward(x_train)

    # forward and backward through loss
    loss_train = cross_entropy_loss.forward(x_N, y_train)
    dL_dx = cross_entropy_loss.backward(x_N, y_train)

    # backward through the rest
    numpy_MLP.backward(dL_dx)

    for layer in numpy_MLP.layers:

      if hasattr(layer, 'params'):
        # update weight matrix
        layer.params["weight"] = layer.params["weight"] - LEARNING_RATE_DEFAULT * layer.grads["weight"]
        layer.params["bias"] = layer.params["bias"] - LEARNING_RATE_DEFAULT * layer.grads["bias"]

    if (step + 1) % EVAL_FREQ_DEFAULT == 0:
      print("Step", step + 1)

      # accuracy and loss on test set
      x_N_test = numpy_MLP.forward(x_test)
      acc_test = accuracy(x_N_test, y_test)
      loss_test = cross_entropy_loss.forward(x_N_test, y_test)
      print('Accuracy', acc_test)

      # a accuracy and loss one whole train set
      x_N_train_whole = numpy_MLP.forward(x_train_whole)
      loss_train = cross_entropy_loss.forward(x_N_train_whole, y_train_whole)
      acc_train = accuracy(x_N_train_whole, y_train_whole)

      acc_train_list.append(acc_train)
      acc_test_list.append(acc_test)
      loss_train_list.append(loss_train)
      loss_test_list.append(loss_test)


  folder = "./np_results/"
  print("Save results")
  np.save(folder + "loss_train", loss_train_list)
  np.save(folder + "acc_train", acc_train_list)
  np.save(folder + "loss_test", loss_test_list)
  np.save(folder + "acc_test", acc_test_list)

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
  FLAGS, unparsed = parser.parse_known_args()

  main()