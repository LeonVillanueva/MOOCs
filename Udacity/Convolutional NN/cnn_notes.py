# @CREATING FILTERS, EDGE DETECTION
# display

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
import numpy as np

%matplotlib inline

# Read in the image
image = mpimg.imread('images/curved_lane.jpg')

plt.imshow(image)

# grayscale coversion

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

plt.imshow(gray, cmap='gray')

# create a sobel filter
# create a custom [sobel] kernel

# 3x3 array for edge detection

sobel_y = np.array([[ -1, -2, -1],[ 0, 0, 0],[ 1, 2, 1]])
# TODO: Create and apply a Sobel x operator
sobel_x = np.array([[ -1, 0, 1], [ -2, 0, 2],[ -1, 0, 1]])


# filter the image using filter2D,
# which has inputs: (grayscale image, bit-depth, kernel)  

filtered_image = cv2.filter2D(gray, -1, sobel_y)
plt.imshow(filtered_image, cmap='gray')

# @CONVOLUTIONAL LAYER VISUALIZATION

# import the image

import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# TODO: Feel free to try out your own images here by changing img_path
# to a file path to another image on your computer!
img_path = 'data/udacity_sdc.png'

# load color image 
bgr_img = cv2.imread(img_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()

# Define and visualize the filters

import numpy as np

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)

# Filter shape:  (4, 4)

# Defining four different filters, 
# all of which are linear combinations of the `filter_vals` defined above

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 1: \n', filter_1)

# visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')


# Defining a CONVOLUTIONAL layer

import torch
import torch.nn as nn
import torch.nn.functional as F
    
# define a neural network with a single convolutional layer with four filters
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
		# define class NET -> init (in the class) "weight" variable
		
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		# ! important
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
		
			# https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d
			#  torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
			# https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
		
    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
		
        conv_x = self.conv(x) # = self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        activated_x = F.relu(conv_x)
        
        # returns both layers
        return conv_x, activated_x
    
# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
			#torch -> get numpy array -> expand dimensions -> make into torch tensor

model = Net(weight)

# print out the layer in the network
print(model)
print (weight)

# :=
# Net((conv): Conv2d(1, 4, kernel_size=(4, 4), stride=(1, 1), bias=False))
# tensor([[[[-1., -1.,  1.,  1.],
#           [-1., -1.,  1.,  1.],
#           [-1., -1.,  1.,  1.],
#          [-1., -1.,  1.,  1.]]],


#         [[[ 1.,  1., -1., -1.],
#           [ 1.,  1., -1., -1.],
#           [ 1.,  1., -1., -1.],
#           [ 1.,  1., -1., -1.]]],


#        [[[-1., -1., -1., -1.],
#           [-1., -1., -1., -1.],
#           [ 1.,  1.,  1.,  1.],
#           [ 1.,  1.,  1.,  1.]]],


#         [[[ 1.,  1.,  1.,  1.],
#           [ 1.,  1.,  1.,  1.],
#           [-1., -1., -1., -1.],
#           [-1., -1., -1., -1.]]]])

# visualize the output of each filter

# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
			
# plot original image
plt.imshow(gray_img, cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))

    
# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get the convolutional layer (pre and post activation)
conv_layer, activated_layer = model(gray_img_tensor)

# visualize the output of a conv layer
viz_layer(conv_layer)

# @MAXPOOLING LAYER LAYER VISUALIZATION

# image import

import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# TODO: Feel free to try out your own images here by changing img_path
# to a file path to another image on your computer!
img_path = 'data/udacity_sdc.png'

# load color image 
bgr_img = cv2.imread(img_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()

# define and visualize the filters

import numpy as np

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])
# !this can be multiplied by INTERGER FOR MORE OSCILLATION"

print('Filter shape: ', filter_vals.shape)

# Defining four different filters, 
# all of which are linear combinations of the `filter_vals` defined above

# define four filters
# transformations of filter_vals -> four different filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 1: \n', filter_1)

# Defining a Convolutional and Max Pooling Layer

import torch
import torch.nn as nn
import torch.nn.functional as F
    
# define a neural network with a convolutional layer with four filters
# AND a pooling layer of size (2, 2)
class Net(nn.Module):
    
    def __init__(self, weight):
        super(Net, self).__init__()
        # initializes the weights of the convolutional layer to be the weights of the 4 defined filters
        k_height, k_width = weight.shape[2:]
        # defines the convolutional layer, assumes there are 4 grayscale filters
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv = nn.Conv2d(1, 4, kernel_size=(k_height, k_width), bias=False)
        self.conv.weight = torch.nn.Parameter(weight)
        # define a pooling layer
        self.pool = nn.MaxPool2d(2, 2)
		
		# https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html

    def forward(self, x):
        # calculates the output of a convolutional layer
        # pre- and post-activation
        conv_x = self.conv(x)
        activated_x = F.relu(conv_x)
        
        # applies pooling layer
        pooled_x = self.pool(activated_x)
        
        # returns all layers
		# ? why is it returning
		
        return conv_x, activated_x, pooled_x

    
# instantiate the model and set the weights
weight = torch.from_numpy(filters).unsqueeze(1).type(torch.FloatTensor)
model = Net(weight)

# print out the layer in the network
print(model)

# visualize each layer output

# helper function for visualizing the output of a given layer
# default number of filters is 4
def viz_layer(layer, n_filters= 4):
    fig = plt.figure(figsize=(20, 20))
    
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1)
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))
			
# plot original image
plt.imshow(gray_img, cmap='gray')

# visualize all filters
fig = plt.figure(figsize=(12, 6))
fig.subplots_adjust(left=0, right=1.5, bottom=0.8, top=1, hspace=0.05, wspace=0.05)
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))

    
# convert the image into an input Tensor
gray_img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(1)

# get all the layers 
conv_layer, activated_layer, pooled_layer = model(gray_img_tensor)

# visualize the output of the activated conv layer
viz_layer(activated_layer)

# visualize the output of the pooling layer
viz_layer(pooled_layer)

# CAPSULE NETWORKS special
# https://github.com/cezannec/capsule_net_pytorch/blob/master/Capsule_Network.ipynb
# https://github.com/higgsfield/Capsule-Network-Tutorial

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)])
    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        batch_size = x.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = self.squash(u)
        return u_squash
    
    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor
			
# Digit Capsules 

# Convolutional Layers PYTORCH

nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)


    #in_channels refers to the depth of an input. For a grayscale image, this depth = 1
    #out_channels refers to the desired depth of the output, or the number of filtered images you want to get as output
    #kernel_size is the size of your convolutional kernel (most commonly 3 for a 3x3 kernel)
    #stride and padding have default values, but should be set depending on how large you want your output to be in the spatial dimensions x, y

# Max Pooling Layer Pytorch

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

# https://pytorch.org/docs/stable/nn.html#maxpool2d

# kernel_size – the size of the window to take a max over
# stride – the stride of the window. Default value is kernel_size
# padding – implicit zero padding to be added on both sides
# dilation – a parameter that controls the stride of elements in the window
# return_indices – if True, will return the max indices along with the outputs. Useful for torch.nn.MaxUnpool2d later
# ceil_mode – when True, will use ceil instead of floor to compute the output shape

# COnvolutional Layers in PyTorch

import torch.nn as nn

# after __init__

self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

x = F.relu(self.conv1(x))

self.pool = nn.MaxPool2d(2,2)

x = F.relu(self.conv1(x))
x = self.pool(x)

# Example:

self.conv1 = nn.Conv2d(1, 16, 2, stride=2)
self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

# https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/807590ea-abd5-4581-b91d-9eede9a0aad2/concepts/62a4771f-ea82-44a4-afac-dd6bacda27bc

# Defining a Sequential Model

def __init__(self):
        super(ModelName, self).__init__()
        self.features = nn.Sequential(
              nn.Conv2d(1, 16, 2, stride=2),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True),

              nn.Conv2d(16, 32, 3, padding=1),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True) 
         )

# Formula: Number of Parameters in a Convolutional Layer
# number of parameters in the convolutional layer = K*F*F*D_in + K
# K - the number of filters in the convolutional layer
# F - the height and width of the convolutional filters
# D_in - the depth of the previous layer

# Formula: Shape of a Convolutional Layer
# S - the stride of the convolution
# P - the padding
# W_in - the width/height (square) of the previous layer

# depth of the convolutional layer = the number of filters K
# spatial dimensions of a convolutional = (W_in−F+2P)/S+1

# Flattening

# Part of completing a CNN architecture, is to flatten the eventual output of a series of convolutional and pooling layers, so that all parameters can be seen (as a vector) by a linear classification layer. At this step, it is imperative that you know exactly how many parameters are output by a layer.

# INPUT:  130x130 (x, y) and 3 in depth (RGB)

nn.Conv2d(3, 10, 3)
nn.MaxPool2d(4, 4)
nn.Conv2d(10, 20, 5, padding=2)
nn.MaxPool2d(2, 2)

# FINAL DEPTH = 20
# FINAL XY = 130 - [1*2]convoltion = 128 / [4] = 32 + [2*2]padding - [1*2] = 34/[2] = 17 - [1]?
# where does the one come from?
# 130 - 1*2 -> 128 / 4 -> 32 - 1*2 -> 30 + 2 (padding) -> 32 / 2
# FINAL XY = 16

# important := Following the Dimensions

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 32x32x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 16x16x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 8x8x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)
		
# important: SETUP @ init can be not the same order so much better to:
# NEURAL NETWORK SETUP
# setup: CONV, POOL layers -> CoNV, POOL forward -> follow output dimensions -> flatten -> FC layer -> FC forward -> y_hat -> CRITERION -> OPTIMIZER -> training with train/validation loss -> test

# model.train() -> model.eval () -> model.train() CYCLE

train.cuda.is_available()

# train on GPU when available

# @Image Augmentation
# Invariant Representation: Scale, Angle, Rotation, Translation, Position -> model shouldn't change prediction
# Pooling
# How to Make model more statistically invariant -> create random variance for the Network Model to Learn = Data Augmentation -> better generalization

# @Augmentation through Transforamtion

import torchvision.transforms as transforms

transform = transformations.Compose ([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# NOTE TO SELF: AWS for CLOUD COMPUTING:= https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/6a6a46cd-855d-4b48-9bfc-6b72a3d135e7/concepts/83a5ef87-6354-403e-96aa-7d5099ba48c6





