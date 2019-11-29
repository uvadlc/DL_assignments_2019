################################################################################
# MIT License
# 
# Copyright (c) 2019
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# import part1.grads_over_time

# from matplotlib import pyplot as plt

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

################################################################################

grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad

    return hook


def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = "cpu"

    # Initialize the model that we are going to use
    # add option for RNN or lstm
    # config.input_length=5

    input_dim = config.input_dim
    num_hidden = config.num_hidden
    num_classes = config.num_classes

    try_palin_lengths = [5, 10, 15, 20, 30, 50, 100]

    # in these lists we store the average training or test accuracies and losses for each palindrome length
    total_av_train_loss = []
    total_av_train_acc = []
    total_av_test_acc = []
    total_av_test_loss = []

    for palin_len in try_palin_lengths:
        print("palindrom length is " + str(palin_len))

        # overwrite default input length
        config.input_length = palin_len
        seq_length = config.input_length

        if config.model_type == 'RNN':
            model = VanillaRNN(seq_length, input_dim, num_hidden, num_classes, device='cpu')  # fixme

        elif config.model_type == 'LSTM':
            model = LSTM(seq_length, input_dim, num_hidden, num_classes, device='cpu')
            print("LSTM used")

        else:
            print('model type should be either RNN or LSTM')

        # Initialize the dataset and data loader (note the +1)
        dataset_train = PalindromeDataset(config.input_length + 1)
        data_loader_train = DataLoader(dataset_train, config.batch_size, num_workers=1)
        # testing data
        dataset_test = PalindromeDataset(config.input_length + 1)
        data_loader_test = DataLoader(dataset_test, config.batch_size, num_workers=1)

        # Setup the loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()  # fixme
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)  # fixme

        # iniitialise lists to keep the training performace stored (over each training step)
        acc_list_train = []
        loss_list_train = []

        loss_test_per_eval = []
        acc_test_per_eval = []

        h_gradients = []
        for step, (batch_inputs, batch_targets) in enumerate(
                data_loader_train):  # enumerate is for looping through with automted counter

            # Only for time measurement of step through network
            t1 = time.time()
            x, y = batch_inputs, batch_targets

            # zero the gradients
            optimizer.zero_grad()

            # do one forward pass
            x_N_train, h = model.forward(x=x)  # h

            # h.retain_grad()
            #
            # h[:, 1].sum().backward()  # this is towards the end of all time stpes
            # print(h.grad)  #empty here bc no backprop yet

            # calculate the loss
            loss = criterion(x_N_train, y)

            # get gradients with respect to that loss
            loss.backward()

            # print(h.grad.sum())  #now it is printing the gradient.
            # h_gradients.append(h.grad.sum())

            # actual optimizing step
            optimizer.step()

            ############################################################################
            # QUESTION: what happens here and why?
            ############################################################################
            # torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            # The norm is computed over all gradients together, as if they wereconcatenated into a single vector. Gradients are modified in-place.
            # clip gradient, like rescale to deal with exploding gradients.

            ############################################################################

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / max(float(t2 - t1), 1)  # it's going to quick?

            # determine trainin accuracy and loss
            if step % 10 == 0:

                # evaluate training performance
                predicted = torch.argmax(x_N_train.data,
                                         1)  # max is the identity operation for that one element  - all zero predictions are converted to 1s, all others are rounded to full int because 1 is int

                total_predictions = y.size(0)  # used to be +=
                correct_predictions = predicted.eq(y.data).sum().item()  # used to be +=
                accuracy_current = 100 * correct_predictions / total_predictions

                loss_current = loss.item()

                accuracy = accuracy_current
                loss = loss_current

                # store accuracies of training evaluation
                acc_list_train.append(accuracy_current)
                loss_list_train.append(loss_current)

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                ))

                # here the test accuracies and losses are stored for each time the model is tested

                acc_test_eval = 0.0
                loss_test_eval = 0.0
                acc_test_cum = 0.0
                loss_test_cum = 0.0

                no_evaluations = 1000
                if step % 500 == 0 and step != 0:

                    # we wil make an average of the acc and the loss for the testing time point.
                    acc_test_cum = 0.0
                    loss_test_cum = 0.0

                    with torch.no_grad():
                        for step_test, (batch_inputs_test, batch_targets_test) in enumerate(
                                data_loader_test):  # enumerate is for looping through with automted counte
                            # this stops the for loop when we have sampled 1000 times from the test dataset
                            if step_test >= no_evaluations:
                                break

                            # # load current batch
                            x, y = batch_inputs_test, batch_targets_test

                            # do one forward pass
                            x_N_test, h = model.forward(x=x)

                            # calculate the loss
                            loss = criterion(x_N_test, y)

                            predicted_test = torch.argmax(x_N_test.data, 1)

                            # get performance of current testing step
                            total_predictions = predicted_test.size(0)
                            correct_predictions = predicted.eq(y.data).sum().item()
                            accuracy_current_test = 100 * correct_predictions / total_predictions
                            loss_current_test = loss.item()

                            # add performance to all testing steps (at this evaluation)
                            acc_test_cum += accuracy_current_test
                            loss_test_cum += loss_current_test

                    #get the accuracy of the test evaluation
                    acc_test_eval = acc_test_cum / no_evaluations
                    loss_test_eval = loss_test_cum / no_evaluations
                    print("After %s time steps " % step, "Accuracy is ", acc_test_eval, "and Loss is ", loss_test_eval)
                    acc_test_per_eval.append(acc_test_eval)
                    loss_test_per_eval.append(loss_test_eval)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                print('Done training.')

                break

        num_evals_total = len(acc_test_per_eval)
        acc_test_per_palin_len = sum(acc_test_per_eval[0:num_evals_total]) / num_evals_total
        loss_test_per_palin_len = sum(loss_test_per_eval[0:num_evals_total]) / num_evals_total

        # get the overall testing result for model and sequence length
        total_av_test_acc.append(acc_test_per_palin_len)
        total_av_test_loss.append(loss_test_per_palin_len)
        # print(total_av_test_loss)

        # this is kinda unnecessary i guess
        len_steps = len(acc_list_train)
        last_batches = 5
        final_accuracy = sum(acc_list_train[len_steps - last_batches: len_steps]) / last_batches
        final_loss = sum(loss_list_train[len_steps - last_batches:len_steps]) / last_batches
        # print(final_accuracy)
        # print(final_loss)

        folder = "./results/"

        # save training performance
        filename_acc_train = folder + config.model_type + "_acc_train"  # acc_list)
        filename_loss_train = folder + config.model_type + "_loss_train"  # loss_list)
        np.save(filename_acc_train + "_palindrome_len_" + str(seq_length), acc_list_train)
        np.save(filename_loss_train + "_palindrome_len_" + str(seq_length), loss_list_train)

        # save test performance
        filename_acc_test = folder + config.model_type + "_acc_test"  # acc_list)
        filename_loss_test = folder + config.model_type + "_loss_test"  # loss_list)
        np.save(filename_acc_test + "_palindrome_len_" + str(seq_length), acc_test_per_palin_len)
        np.save(filename_loss_test + "_palindrome_len_" + str(seq_length), loss_test_per_palin_len)

        # append the overall trainging results for each step
        total_av_train_loss.append(final_loss)
        total_av_train_acc.append(final_accuracy)
        # save model
        model_name = config.model_type + "_seqlen_" + str(palin_len) + "_palin_model.pt"
        torch.save(model, model_name)

        # save test results, these are the important results
        filename_acc_test = folder + config.model_type + "_acc_test"  # acc_list)
        filename_loss_test = folder + config.model_type + "_loss_test"  # loss_list)
        np.save(filename_acc_test + "_palindrome_len", total_av_test_acc)
        np.save(filename_loss_test + "_palindrome_len", total_av_test_loss)
        np.save(folder + "palins_tried", try_palin_lengths)

        # save training results
        filename_acc_train = folder + config.model_type + "_acc_train"  # acc_list)
        filename_loss_train = folder + config.model_type + "_loss_train"  # loss_list)
        np.save(filename_acc_train + "_palindrome_len", total_av_train_acc)
        np.save(filename_loss_train + "_palindrome_len", total_av_train_loss)

        print("results saved")

    # return seq_length, final_accuracy, final_loss

    ################################################################################
    ################################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=6000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    train(config)
