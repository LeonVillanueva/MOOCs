# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 13:08:50 2019

@author: FartherSkies
"""

# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000 # number of rounds, number of "users"
d = 10 # number of arms : ad choices
ads_selected = []
numbers_of_selections = [0] * d # vector zero : initial selection count
sums_of_rewards = [0] * d # sum of rewards of each version of ad at round N
total_reward = 0

for n in range (0, N): # per round
    ad = 0
    max_upper_bound = 0
    for i in range (0, d): # per AD = d, choices / bandits
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            '''
            https://www.udemy.com/course/machinelearning/learn/lecture/6017492#questions
            '''
            delta_i = math.sqrt (3/2 * math.log(n+1) / numbers_of_selections[i] )
            # number of times selection was selected by round i
            # n starts at zero
            # 3/2 assumes bell curve
            upper_bound = average_reward + delta_i
        else: # this is a trick, to make sure the first 10 selections are the first 10, give us info
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append (ad)
    numbers_of_selections [ad] += 1
    reward = dataset.values [n, ad]
    sums_of_rewards [ad] += reward
    total_reward += reward

print (total_reward) # eventually the algorithm converges into the highest conversion rate AD

from statistics import mode 
print ('AD Number ' + str (mode (ads_selected)+1) + ' has the highest click rate.')


n, bins, patches = plt.hist(ads_selected, bins=d, align='left', color='#95d0fc')
patches[mode (ads_selected)].set_fc('#ff474c')

plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')


plt.show()