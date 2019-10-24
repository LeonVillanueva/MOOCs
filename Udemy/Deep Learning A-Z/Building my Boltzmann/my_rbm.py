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
movies = pd.read_csv ('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
