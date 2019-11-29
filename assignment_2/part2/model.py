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

import torch

import torch.nn as nn
from torch.autograd import Variable


class TextGenerationModel(nn.Module):
    def __init__(self, batch_size, seq_length, vocabulary_size, lstm_num_hidden=256, lstm_num_layers=2,
                 device='cuda:0'):
        super(TextGenerationModel, self).__init__()
        # Initialization here...
        #self.device = device  # 'cuda:0' #necessary?

        # Initialize the device which to run the model on
        # check if cuda device is available

        # Defining some parameters
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size

        # RNN Layer
        # self.rnn = nn.RNN(seq_length, vocabulary_size, lstm_num_hidden, lstm_num_layers, batch_first=True)
        # set up modules for recurrent neural networks
        self.rnn = nn.LSTM(input_size=vocabulary_size,
                           hidden_size=lstm_num_hidden,
                           num_layers=lstm_num_layers,
                           batch_first=True)  # bidirectional = True)  #dropout=dropout

        # self.rnn outputs  out_rnn (hn, cn)

        input = Variable(torch.randn(1, batch_size, seq_length))
        self.lin_layer = nn.Linear(lstm_num_hidden, vocabulary_size, bias=True)


        # self.feature.apply(weights_init)

#indentation mattered here, when it was on the complete left then it was on the wrong indentation leve, needs to be within init
    def forward(self, x, states=None):
        self.x = x

        out_rnn, (h_prev, c_prev) = self.rnn(x, states)  # or states  #(h0, c0)
        #print(out_rnn)
        out = self.lin_layer(out_rnn)

        # v1 = nn.Sequential(seq_length * batch_size, lstm_num_hidden)
        # v2 = nn.Sequential(seq_length, batch_size, vocabulary_size)
        # output = v2(lin(v1(out_rnn)))

        return out, (h_prev, c_prev)


    # def init_hidden(self, batch_size):
    #     # This method generates the first hidden state of zeros which we'll use in the forward pass
    #     hidden = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_num_hidden)  # .to(self.device)
    #     # We'll send the tensor holding the hidden state to the device we specified earlier as well
    #     return hidden
    #
    #

    # batch_size = x.size(0)
    #
    # # Initializing hidden state for first input using method defined below = makes a tensor of dim 2 64 2
    # hidden = self.init_hidden(batch_size)
    #
    # print(x.shape)
    # # Passing in the input and hidden state into the model and obtaining outputs
    # out, hidden = self.rnn(x, hidden)
    # print(out.shape)
    #
    # # Reshaping the outputs such that it can be fit into the fully connected layer
    # out = out.contiguous().view(-1, self.lstm_num_hidden)
    # out = self.fc(out, self.vocabulary_size)

    # h0 = Variable(torch.randn(lstm_num_layers, batch_size, lstm_num_hidden))  # (num_layers, batch, hidden_size)
    # c0 = Variable(torch.randn(lstm_num_layers, batch_size, lstm_num_hidden))
