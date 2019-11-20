"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import torch
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
REGULARIZER = 0

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
  print(predictions)

  # for the train
  targets = targets.argmax(dim=1)
  print(targets)

  # check if predicted and actual label are equal and calculate accuracy
  accuracy = (predictions == targets).float().mean().data.item()
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  # load the data
  cifar10 = cifar10_utils.get_cifar10()

  # at the evaluation step we will go iteratively through the train and test set
  nr_required_test_iterations = int(np.ceil(cifar10['test']._num_examples) / BATCH_SIZE_DEFAULT)
  nr_required_train_iterations = int(np.ceil(cifar10['train']._num_examples) / BATCH_SIZE_DEFAULT)

  # get parameters of network
  n_classes = 10  # y_test.shape[1]
  n_channels = 3  # x_test.shape[1] # this is here indeed just the number of channels; so 3

  # initialize network
  net = ConvNet(n_channels=n_channels, n_classes=n_classes).to(device)

  # initialize loss // already contains softmax layer
  loss_fn = torch.nn.CrossEntropyLoss()

  # initiliaze optimizier
  if OPTIMIZER_DEFAULT == 'ADAM':
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE_DEFAULT)

  # list for evalation metrics
  acc_train_list = []
  acc_test_list = []
  loss_train_list = []
  loss_test_list = []

  # for step in range(1):
  for step in range(MAX_STEPS_DEFAULT):

    # load current batch
    x, y = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)

    # make it pytorch tensor ; no reshape
    x_train = torch.tensor(x).to(device)
    y_train = torch.tensor(y).to(device)

    # index of actual label
    y_train = y_train.argmax(dim=1)

    # zero the gradients
    optimizer.zero_grad()

    # do one forward pass
    x_N_train = net.forward(x_train)

    # calculate the loss
    loss = loss_fn(x_N_train, y_train)
    # print('Loss',loss.item())

    # get gradients with respect to that loss
    loss.backward()

    # actual optimizing step
    optimizer.step()

    # to save memory
    x_N_train.detach()
    y_train.detach()

    if (step + 1) % EVAL_FREQ_DEFAULT == 0:

      print("Step", step + 1)
      # intermediate result lists
      acc_train_list_interm = []
      acc_test_list_interm = []
      loss_train_list_interm = []
      loss_test_list_interm = []

      # evaluate test set by going via batches through the whole set
      # instead of evaluating the whole test set at once

      for intermediate_step_test in range(nr_required_test_iterations):
        # if (intermediate_step_test % 50 ==0):
        #     print('test',intermediate_step_test)

        x_test, y_test = cifar10['test'].next_batch(BATCH_SIZE_DEFAULT)
        x_test = torch.tensor(x_test, requires_grad=False).to(device)
        y_test = torch.tensor(y_test, requires_grad=False).to(device)

        # calculate for each test set batch loss and accuracy
        x_N_test = net.forward(x_test)
        acc_test = accuracy(x_N_test, y_test)
        loss_test = loss_fn(x_N_test, y_test.argmax(dim=1)).item()

        # append intermediate results to list for each batch
        acc_test_list_interm.append(acc_test)
        loss_test_list_interm.append(loss_test)

        x_N_test.detach()
        x_test.detach()
        y_test.detach()

      # average accuracies and loss over all batches to get
      # acc and loss for the whole training set
      acc_test_list.append(np.mean(acc_test_list_interm))
      loss_test_list.append(np.mean(loss_test_list_interm))

      print('Accuracy test', acc_test_list[-1])
      # print('Loss test', loss_test_list[-1])

      # now to the same for the train set, that means
      # go through the whole train set via batches
      for intermediate_step_train in range(nr_required_train_iterations):
        # if (intermediate_step_train % 50 ==0):
        #     print('train',intermediate_step_train)

        x_train, y_train = cifar10['train'].next_batch(BATCH_SIZE_DEFAULT)
        x_train = torch.tensor(x_train, requires_grad=False).to(device)
        y_train = torch.tensor(y_train, requires_grad=False).to(device)

        # store loss and accuracy on train set
        x_N_train = net.forward(x_train)
        acc_train = accuracy(x_N_train, y_train)
        loss_train = loss_fn(x_N_train, y_train.argmax(dim=1)).item()

        # append intermediate results to list for each batch
        acc_train_list_interm.append(acc_train)
        loss_train_list_interm.append(loss_train)

        # save memory
        x_N_train.detach()
        x_train.detach()
        y_train.detach()

      # get test metrics on whole train set for current step
      acc_train_list.append(np.mean(acc_train_list_interm))
      loss_train_list.append(np.mean(loss_train_list_interm))

      # print('Accuracy train', acc_train_list[-1])
      # print('Loss train', loss_train_list[-1])

  folder = "./cnn_results/"
  print("Saving results")
  np.save(folder + "loss_train", loss_train_list)
  np.save(folder + "acc_train", acc_train_list)
  np.save(folder + "loss_test", loss_test_list)
  np.save(folder + "acc_test", acc_test_list)

  # loss_test = round(loss_test, 3)
  # acc_test = round(acc_test, 4)
  #
  # data = [[activation, DNN_HIDDEN_UNITS_DEFAULT, LEARNING_RATE_DEFAULT, NEG_SLOPE_DEFAULT, BATCH_SIZE_DEFAULT,
  #          OPTIMIZER_CHOICE, acc_test, loss_test]]
  #
  # df = pd.DataFrame(data, columns=['Activation', 'DNN_hidden_units', 'Learning Rate', 'Neg_Slope',
  #                                  'Batch_size', 'Optimizer', 'Accuracy', 'Loss'])
  #
  # f = 'results_pytorch.txt'
  # df.to_csv(folder + f, header=None, index=None, mode='a', sep=' ')

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


  FLAGS, unparsed = parser.parse_known_args()

  main()