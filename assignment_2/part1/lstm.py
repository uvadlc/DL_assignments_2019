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

################################################################################

def xavier_init(m,h):
    return torch.Tensor(m,h).uniform_(-1,1)*math.sqrt(6./(m+h))

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

        self.seq_length = seq_length  # number of digits?
        self.input_dim=input_dim

        #input modulation gate g
        self.W_gx = torch.nn.Parameter(xavier_init(seq_length, num_hidden))
        self.W_gh= torch.nn.Parameter(xavier_init(num_hidden, num_hidden))
        self.b_g= torch.nn.Parameter(torch.zeros(input_dim,num_hidden))

        #input gate
        self.W_ix = torch.nn.Parameter(xavier_init(seq_length, num_hidden))
        self.W_ih = torch.nn.Parameter(xavier_init(num_hidden, num_hidden))
        self.b_i = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))

        #forget gate
        self.W_fx = torch.nn.Parameter(xavier_init(seq_length, num_hidden))
        self.W_fh = torch.nn.Parameter(xavier_init(num_hidden, num_hidden))
        self.b_f = torch.nn.Parameter(torch.ones(input_dim, num_hidden))

        #output gate
        self.W_ox = torch.nn.Parameter(xavier_init(seq_length, num_hidden))
        self.W_oh = torch.nn.Parameter(xavier_init(num_hidden, num_hidden))
        self.b_o = torch.nn.Parameter(torch.zeros(input_dim, num_hidden))

        #end
        self.W_ph= torch.nn.Parameter(xavier_init(num_hidden, num_classes))  # already transposed?
        # [1 x H] hidden state, summarising the contents of past, input dim ==1
        self.b_p = torch.nn.Parameter(torch.zeros(input_dim,num_hidden))

        #initialise h t-1)
        self.h=torch.zeros(num_hidden, num_hidden)
        self.c_t=torch.zeros(num_hidden, num_hidden)


    def forward(self, x):

        # if self.time_point !=0:
        #     print(self.h)
        h_prev=self.h.detach()
        c_prev=self.c_t.detach()

        self.x = x

        # second til second last RNN Cell #the recurrent part
        for t in range(1, self.seq_length):
            g_gate = torch.tanh(torch.mm(x, self.W_gx) + torch.mm(h_prev, self.W_gh) + self.b_g)
            i_gate = torch.sigmoid(torch.mm(x, self.W_ix) + torch.mm(h_prev, self.W_ih) + self.b_i)
            f_gate = torch.sigmoid(torch.mm(x, self.W_fx) + torch.mm(h_prev, self.W_fh) + self.b_f)
            o_gate = torch.sigmoid(torch.mm(x, self.W_ox) + torch.mm(h_prev, self.W_oh) + self.b_o)


        #compute cell state
        self.c_t=(torch.mul(g_gate,  i_gate) + torch.mul(c_prev, f_gate))
        #calculate h of current time step
        self.h= torch.mul(torch.tanh(self.c_t), o_gate)

        p = torch.mm(self.h, self.W_ph) + self.b_p.t() # p shape soll 128 10



        return p, self.h


