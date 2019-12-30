# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 09:44:31 2019

@author: FartherSkies
"""

# Thompson Sampling

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
import random
N = 10000
d = 10
ads_selected = []

'''
    numbers_of_selections = [0] * d # vector zero : initial selection count
    sums_of_rewards = [0] * d # sum of rewards of each version of ad at round N
'''

numbers_of_rewards_1 = [0] * d 
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range (0, N): # per round
    ad = 0
    max_random = 0
    for i in range (0, d): # per AD = d, choices / bandits
        random_beta = random.betavariate(numbers_of_rewards_1[i]+1, numbers_of_rewards_0[i] + 1) # +1, can't be zero
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append (ad)
    reward = dataset.values [n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1 # update the random machines' distribution
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward += reward

print (total_reward) # eventually the algorithm converges into the highest conversion rate AD

from statistics import mode 
print ('AD Number ' + str (mode (ads_selected)+1) + ' has the highest click rate.')


n, bins, patches = plt.hist(ads_selected, bins=d, align='left', color='#95d0fc')
patches[mode (ads_selected)].set_fc('#00035b')

plt.title('Histogram of ads selections: Thompson')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')


plt.show()