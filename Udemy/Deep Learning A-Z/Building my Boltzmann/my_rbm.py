# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:43:58 2019

@author: FartherSkies
"""

# boltzmann machine

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

movies = pd.read_csv ('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv ('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv ('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

training_set = pd.read_csv ('ml-100k/u1.base', delimiter='\t')
training_set = np.array (training_set, dtype='int')
test_set = pd.read_csv ('ml-100k/u1.test', delimiter='\t')
test_set = np.array (test_set, dtype='int')

nb_movies = int (max (max (training_set[:,1]), max(test_set[:,1])))
nb_users = int (max (max (training_set[:,0]), max(test_set[:,0])))

def convert (data):
	# output : list of list, all movies, 0 = not rated
	new_data = []
	# get all the movies and ratings in the training set per user
	for id_users in range (1, nb_users+1):
		id_movies = data[:,1][data[:,0]==id_users]
		id_ratings = data[:,2][data[:,0]==id_users]
		# create a zero list = unwatched
		ratings = np.zeros (nb_movies)
		ratings[id_movies-1] = id_ratings
		new_data.append (list (ratings))
		# pytorch expects a list
	return new_data

training_set = convert (training_set)	
test_set = convert (test_set)

# convert for pytorch

torch_train = torch.FloatTensor (training_set)
torch_test = torch.FloatTensor (test_set)

# convert to binary 0/1 = dislike/like
# with -1 not watched

torch_train[torch_train == 0] = -1
torch_train[torch_train == 1] = 0
torch_train[torch_train == 2] = 0
torch_train[torch_train >= 3] = 1

torch_test[torch_test == 0] = -1
torch_test[torch_test == 1] = 0
torch_test[torch_test == 2] = 0
torch_test[torch_test >= 3] = 1

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)    