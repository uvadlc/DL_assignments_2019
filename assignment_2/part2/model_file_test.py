from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import argparse

import numpy as np
#import train
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#from part2.dataset import TextDataset
from model import TextGenerationModel

from dataset import TextDataset

def one_hot_encoding (batch_data, vocab_size):

    # add vocabulary to input dimension size
    one_hot_dim = list(batch_data.shape)  #64, 31
    one_hot_dim.append(vocab_size) # result ist of shape batchsize x seq_length x vocab_Size
    one_hot_encodings = torch.zeros(one_hot_dim, device=batch_data.device)
    u=batch_data.unsqueeze(-1)  # adds the last dimension to the batch x batch_size- vocab_size x 1
    one_hot_encodings.scatter_(2, batch_data.unsqueeze(-1), 1)   #dim- where we want to modify or input the values., then input- thats the 3d tensor we make by unsqueezing i.e. adding a useless dimension, source: the value at that index that will be written to self. basically it means that in the one hot encoding there will be 1s if true
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



def text_generator(model, sentence_length, dataset, device, temperature, random=True, phrase=None):
    with torch.no_grad():
        # generate random character to start with
        if (random == True):
            start_character = torch.randint(dataset.vocab_size, (1, 1), dtype=torch.long, device=device)
        else:
            start_character = phrase

        # this is gonna be our list of characters
        sentence = start_character.view(-1).tolist()


        # get one hot encoding of character and get next_character
        start_character = one_hot_encoding(start_character, dataset.vocab_size)

        out, (h, c) = model(start_character)

        #print(out[:, -1, :])

        next_character = get_next_character(out[:, -1, :], temperature)

        # and append to sentence list
        sentence.append(next_character)

        for t in range(sentence_length - 1):
            # convert current character to one hot and get next character
            input = one_hot_encoding(torch.tensor(next_character, dtype=torch.long, device=device).view(1, -1),
                                     dataset.vocab_size)
            out, (h, c) = model(input, (h, c))
            next_character = get_next_character(out, temperature)
            # and append to sentence list again
            sentence.append(next_character)

        text = dataset.convert_to_string(sentence)

        return text

seq_length=30
seq_length= 30
temperature=0.5
batch_size=64
device='cpu'
filename =  "./grim.txt"
txt_file=filename
dataset = TextDataset(filename, seq_length+1)  # fixme  #+1 weg
data_loader = DataLoader(dataset, batch_size, num_workers=1)
vocabulary_size=TextDataset(filename, seq_length+1).vocab_size
folder = "./generated_texts/"

#save

model=torch.load("./0_0.5_grim_model.pt")
texts_during_test=[]
for step in range(100):
    text = text_generator(model, seq_length, dataset, device, temperature) #config.seq_length  config.temperature
    texts_during_test.append('Step ' + str(step) + ': ' + text)
    print(text)
filename= folder + 'test'
np.save(filename, texts_during_test)

with open(filename+'.txt', 'w') as f:
    for item in texts_during_test:
        f.write("%s\n" % item)
