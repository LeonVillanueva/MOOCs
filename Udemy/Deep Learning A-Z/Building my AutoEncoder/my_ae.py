# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:35:11 2019

@author: FartherSkies
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# data is similar to RBM model

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Convert to arrays
nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    # based on user/movie index from two different RANDOM splits

print (nb_movies, nb_users)

# observations = row, features = columns

def convert (data):
    # list of list for pytorch
    new_data = []
    for id_users in range (1, nb_users+1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        # indices of the RATED movies
        ratings = np.zeros (nb_movies)
        ratings[id_movies-1]=id_ratings
        new_data.append (list(ratings))
    return new_data

torch_train = convert (training_set)
torch_test = convert (test_set)

# to torch tensor
torch_train = torch.FloatTensor (torch_train)
torch_test = torch.FloatTensor (torch_test)

# autoencoder class : {f(x)}->object
# nn(parent) > ae_class(child) >> inheritance

class Stacked_AE (nn.Module):
    def __init__(self,):
        super(Stacked_AE, self).__init__
        # all functions and inherited
        
        # encoding
        self.fc1 = nn.Linear(nb_movies, 30)
        self.fc2 = nn.Linear(30, 10)
        # decoding
        self.fc3 = nn.Linear(10, 30)
        self.fc4 = nn.Linear(30, nb_movies)
        
        self.activation = nn.Sigmoid()
    