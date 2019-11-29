# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#from part2.dataset import TextDataset
from model import TextGenerationModel

from dataset import TextDataset


################################################################################
def one_hot(batch_data, vocab_size):
    # add vocabulary to input dimension size
    one_hot_dim = list(batch_data.shape)  #64, 31
    one_hot_dim.append(vocab_size) # result ist of shape batchsize x seq_length x vocab_Size
    one_hot_encodings = torch.zeros(one_hot_dim, device=batch_data.device)
    u=batch_data.unsqueeze(-1)  # adds the last dimension to the batch x batch_size- vocab_size x 1
    one_hot_encodings.scatter_(2, u, 1)   #dim- where we want to modify or input the values., then input- thats the 3d tensor we make by unsqueezing i.e. adding a useless dimension, source: the value at that index that will be written to self. basically it means that in the one hot encoding there will be 1s if true

    return one_hot_encodings

def get_next_character(input, temperature):

    if temperature == 0:  #greedy
        next_character = input.squeeze().argmax() #character with highest prob
    # otherwise lower the peaks before applying softmax
    # and sample according to softmax function
    else:
        distribution = torch.softmax(input.squeeze()/temperature, dim=0)
        next_character = torch.multinomial(distribution, 1)  #Returns a tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of tensor input

    return next_character.item()



def text_generator(model, sentence_length, dataset, device, temperature, random=True, finish_sentence=None):
    with torch.no_grad():
        # generate random character to start with
        if (finish_sentence == None):
            first_char_no = torch.randint(dataset.vocab_size, (1, 1), dtype=torch.long, device=device)
        else:
            first_char_no = finish_sentence
        #first char is now a tensor, thne convert to list (with just the character as entry)

        sentence = first_char_no.view(-1).tolist()

        # get one hot encoding of character and get next_character
        first_char_no = one_hot(first_char_no, dataset.vocab_size)

        out, (h_prev, c_prev) = model(first_char_no)

        next_char = get_next_character(out[:, -1, :], temperature)

        # and append to sentence list
        sentence.append(next_char)

        for t in range(sentence_length - 1):
            input = one_hot(torch.tensor(next_char, dtype=torch.long, device=device).view(1, -1),
                            dataset.vocab_size)
            out, (h_prev, c_prev) = model(input, (h_prev, c_prev))
            next_char = get_next_character(out, temperature)

            sentence.append(next_char)

        text = dataset.convert_to_string(sentence)

        return text



def train(config):
    device= config.device
    if (device == 'cuda'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    seq_length = config.seq_length

    # Initialize the dataset and data loader (note the +1)
    filename =  "./grim.txt"
    txt_file=filename
    dataset = TextDataset(filename, seq_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    vocabulary_size=TextDataset(filename, seq_length+1).vocab_size


    # Initialize the model that we are going to use
    batch_size=config.batch_size
    lstm_num_hidden = config.lstm_num_hidden
    lstm_num_layers = config.lstm_num_layers
    model = TextGenerationModel(batch_size=batch_size, seq_length=seq_length, vocabulary_size=vocabulary_size, lstm_num_hidden=lstm_num_hidden, lstm_num_layers=lstm_num_layers, device=device)  # fixme
    #model=model.to(device)  #not necessary bc parsing device to model

    # Setup the loss and optimizer
    criterion =torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    acc_list = []
    loss_list = []

    batches_dataset= (len(dataset)/batch_size)
    print(batches_dataset)

    no_epochs= int(config.train_steps /batches_dataset)
    print("total no of epochs", no_epochs)

    step_counter=1
    for epoch in range(no_epochs):
        print("Epoch ", epoch)
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()
            # # load current batch
            x, y = batch_inputs, batch_targets
            x_one_hot = one_hot(x, dataset.vocab_size)
            y_one_hot= one_hot(y, dataset.vocab_size)
            #forward pass
            y_hat_train, (h_prev, c_prev) = model.forward(x_one_hot)
            # calculate the loss
            #print(batch_targets.shape)  #torch.Size([64, 30])
            y_hat_train_t = y_hat_train.transpose(2, 1)
            # print(y_hat_train_t.shape)   #torch.Size([64, 87, 30])
            loss = criterion(y_hat_train_t, batch_targets)
            # zero the gradients
            optimizer.zero_grad()
            # get gradients with respect to that loss
            loss.backward()
            # actual optimizing step
            optimizer.step()

            #######################################################
            # Add more code here ..
            #######################################################

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/t2 #float(t2-t1)

            total_train=0
            correct_train=0

            texts_during_train_equally = []
            texts_during_train_all = []

            orig, predicted = torch.max(y_hat_train_t.data,
                                        1)  # max is the identity operation for that one element  - all zero predictions are converted to 1s, all others are rounded to full int because 1 is int
            total_train += batch_targets.size(0)  # batch_size
            correct_train += predicted.eq(batch_targets.data).sum().item() / batch_targets.size(1)  # seq_length
            accuracy_current = (100 * correct_train / total_train)

            loss_current = loss.item()

            accuracy = float(accuracy_current)
            loss = float(loss_current)
            acc_list.append(accuracy_current)
            loss_list.append(loss_current)


            if step % config.print_every == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), int(step),
                        int(config.train_steps), config.batch_size, examples_per_second, accuracy, loss
                ))

            if step % config.sample_every ==0 :
                # Generate some sentences by sampling from the model
                folder = "./generated_texts/"
                sentence="Sleeping Beauty is "
                text = text_generator(model, config.seq_length, dataset, device, config.temperature, finish_sentence=None)
                #text = text_generator(model, config.seq_length, dataset, device, config.temperature)
                texts_during_train_all.append('Step ' + str(step) + ': ' + text)
                print(text)
                filename=folder + 'during_training_all_'+ str(config.temperature)+ "_str_40"
                np.save(filename, texts_during_train_all)
                with open(filename + '.txt', 'a') as f:
                    for item in texts_during_train_all:
                        f.write("%s\n" % item)




            if step_counter == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655

                #total loss over time steps
                total_loss = sum(loss_list[1:(len(loss_list))] )/ (step_counter)
                total_acc= sum(acc_list[1:(len(acc_list))] )/ (step_counter)
                print("The total loss is ", total_loss)
                print("The total Accuracy is ", total_acc)


                break

            step_counter+=1

        print()
        folder = "./results/"
        filename_acc = folder + "LSTM_acc_temp_" + str(config.temperature) + "_seq_" + str(seq_length)
        filename_loss = folder + "LSTM_loss_temp_" + str(config.temperature) + "_seq_" + str(seq_length)
        print(filename_acc)
        np.save(filename_acc, acc_list)
        np.save(filename_loss , loss_list)

        folder = "./models/"
        model_name = folder +  str(epoch) + "_" + str(config.temperature) + "_grim_model.pt"
        print(model_name)
        torch.save(model, model_name)

    print('Done training.')
    total_loss = sum(loss_list[0: (step + 1)]) / (step + 1)
    print("The total loss is ", total_loss)
    folder = "./results/"

    filename_acc = folder + "LSTM_acc_temp_" + str(config.temperature) +"_seq_"+  str(seq_length)
    filename_loss = folder + "LSTM_loss_temp_" +str(config.temperature) +"_seq_"+ str(seq_length)

    np.save(filename_acc , acc_list)
    np.save(filename_loss, loss_list)



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default="grim.txt", required=False, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")


    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e5, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=10, help='How often to sample from the model')
    parser.add_argument('--temperature', type=str, default=0, help='Parameter to steer the amount of randomness for text generation')


    config = parser.parse_args()

    # Train the model
    train(config)
