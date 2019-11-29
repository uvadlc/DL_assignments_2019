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

import torch
import torch.nn as nn
import math
import numpy as np

################################################################################
def xavier_init(m,h):
    return torch.Tensor(m,h).uniform_(-1,1)*math.sqrt(6./(m+h))

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ..

        self.seq_length = seq_length  # number of digits  in palindrome
        self.input_dim=input_dim
        print(self.input_dim)

        # initialize weight matrices as transpose so we don't have to tranpose them later
        self.W_hx = torch.nn.Parameter(xavier_init(seq_length, num_hidden)/num_hidden, requires_grad=True)
        self.W_hh = torch.nn.Parameter(xavier_init(num_hidden, num_hidden)/num_hidden, requires_grad=True)
        self.W_ph= torch.nn.Parameter(xavier_init(num_hidden, num_classes)/num_hidden, requires_grad= True)

        # [1 x H] hidden state, summarising the contents of past
        #self.h = torch.nn.Parameter(torch.zeros(input_dim,num_hidden)/num_hidden)
        self.h = torch.zeros(input_dim,num_hidden)

        # biases
        self.b_h =torch.zeros(input_dim,num_hidden)
        #self.h_new=torch.zeros(num_hidden, num_hidden, requires_grad=True)



    def forward(self, x):

        h_prev = self.h.detach()  #detaches from computation graph

        self.x=x
        # second til second last RNN Cell #the recurrent part
        for t in range(1, self.seq_length):
            # tanh(x*w1 + h_prev*w1 + b) -- hidden state
            self.h= torch.tanh(torch.mm(x, self.W_hx) + torch.mm(h_prev, self.W_hh) + self.b_h)
            # self.h_grad= self.h.clone().detach()
            # self.h_grad.retain_grad()
            p = torch.mm(self.h, self.W_ph)


        return p, self.h  # return hnew


