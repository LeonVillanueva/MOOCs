# transfer learning PRE TRAINED model

# VGG Network
# general features -> speific features
# remove the "specific features" FC detectors -> replace with your own
# train only these new layers

# important DATA SET SIZE and DATA SET similarity

# Fine Tune, ReTrain, End of ConvNet, Start of ConvNet
# https://s3.amazonaws.com/video.udacity-data.com/topher/2018/September/5baa60db_screen-shot-2018-09-25-at-9.22.35-am/screen-shot-2018-09-25-at-9.22.35-am.png

# @dataset size and similarity

# SMALL + SIMILAR:
# freeze weights and slice end of pretrained -> initialize and attach new FC layer
# https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/a559990d-e214-4c5d-a424-437f6299383e/concepts/f80c5e76-e812-4bb0-aecd-40921aaf0bf6

# SMALL - DIFFERENT:
# freeze weights and slice high-end features and fully connected layers -> initialize and attach new FC layer
# https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/a559990d-e214-4c5d-a424-437f6299383e/concepts/f80c5e76-e812-4bb0-aecd-40921aaf0bf6

# LARGE + SIMILAR
# remove last FC layer and replace with similar classification number + reinitialize weights -> keep old weights for rest -> re-train
# https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/a559990d-e214-4c5d-a424-437f6299383e/concepts/f80c5e76-e812-4bb0-aecd-40921aaf0bf6

# LARGE - DIFFERENT
# LARGE + SIMILAR (or also re-initilize ALL weights)
# https://classroom.udacity.com/nanodegrees/nd101/parts/2e8d3b5d-aa70-4376-946f-0cdc37127d7d/modules/19a75d10-547d-4497-ae68-609ca1a235fc/lessons/a559990d-e214-4c5d-a424-437f6299383e/concepts/f80c5e76-e812-4bb0-aecd-40921aaf0bf6

# VGG-16 transfer

# @Notebook Transfer Learning

import os
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

%matplotlib inline

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
		
# define training and test data directories
data_dir = 'flower_photos/'
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

# classes are folders in each directory with these names
classes = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# load and transform data using ImageFolder

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([transforms.RandomResizedCrop(224), 
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))
print ()

print (type(train_data)) #= <class 'torchvision.datasets.folder.ImageFolder'>
print (type(train_data[1])) #= <class 'tuple'>
print (type(train_data[1][1])) #= <class 'int'>
print (type(train_data[1][0])) #= <class 'torch.Tensor'>

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                          num_workers=num_workers, shuffle=True)

print (type(train_loader)) #= <class 'torch.utils.data.dataloader.DataLoader'>

# Iterable

# Visualize some sample data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]], loc='left')

# Load the pretrained model from pytorch
vgg16 = models.vgg16(pretrained=True)

# print out the model structure
print(vgg16)

'''
	VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
'''

print(vgg16.classifier[6].in_features) #= 4096
print(vgg16.classifier[6].out_features) #= 1000
print ()
print(vgg16.classifier[6]) #= Linear(in_features=4096, out_features=1000, bias=True)
print (type((vgg16.classifier[6]))) #= <class 'torch.nn.modules.linear.Linear'>

print (vgg16.classifier[6]) #= Linear(in_features=4096, out_features=1000, bias=True)
vgg16.classifier[6] = torch.nn.Linear(4096, 5)
print (vgg16.classifier[6]) #= Linear(in_features=4096, out_features=5, bias=True)

import torch.optim as optim
import torch.nn as nn

# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.001)

#TEST#

# track test loss 
# over 5 flower classes
test_loss = 0.0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))

vgg16.eval() # eval mode

# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg16(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update  test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate avg test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(5):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
	
# @Freezing Weights + Slice / Replace Last Linear FC

 

for param in vgg16.features.parameters():
                    param.require_grad = False # weights frozen

input_n = vgg16.classifier[6].in_features
output_n = len (classes)

vgg16.classifier[6] = nn.Linear (input_n, output_n)

…
for batch_i, (data, target) in enumerate (train_loader)
                    # run the training on a batch by batch 0 to 19 -> (average loss), 20 to 39 -> (average loss)

                    # batch_i is a mini-batch
…
if batch_i % 20 == 19:
                    print (‘epoch: %d, batch: %d, loss: %d’ % (epoch, batch_i + 1, train_loss/20)
                    train_loss = 0.0 #reset train loss

# @Weight Initialization
# how to instantiate the weights at beginning before training
# helpers.py
# with all ZERO initial weights -> weights * 0 -> less learning -> gradients disappear
# with all ZERO initial weights -> weights * 1 0 -> less learning -> gradients move with the same amount, also too high weights

# Weight Initialization -> make better mistakes that allows the model to learn as much as possible
# continuous uniform distribution
…

 

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# number of subprocesses to use for data loading
num_workers = 0

# how many samples per batch to load
batch_size = 100

# percentage of training set to use as validation

valid_size = 0.2

 

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets

train_data = datasets.FashionMNIST(root='data', train=True,
                                   download=True, transform=transform)

test_data = datasets.FashionMNIST(root='data', train=False,
                                  download=True, transform=transform)

 

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

 

# define samplers for obtaining training and validation batches
# ~useful: Subset Random Sampler -> Split randomly train set to train and validation
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

 

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)

valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=valid_sampler, num_workers=num_workers)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
    num_workers=num_workers)

# specify the image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

 

# Visualize the data
import matplotlib.pyplot as plt
%matplotlib inline

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])

import torch.nn as nn
import torch.nn.functional as F

# define the NN architecture
class Net(nn.Module):
    def __init__(self, hidden_1=256, hidden_2=128, constant_weight=None):
        super(Net, self).__init__()
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (hidden_1 -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (hidden_2 -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        self.dropout = nn.Dropout(0.2)

		# initialize the weights to a specified, constant value
        if(constant_weight is not None):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)
   
    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

 

helpers.hist_dist('Random Uniform (low=-3, high=3)', np.random.uniform(-3, 3, [1000]))

# takes in a module and applies the specified weight initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:

        # apply a uniform distribution to the weights and a bias=0

        m.weight.data.uniform_(0.0, 1.0)

        m.bias.data.fill_(0)

 

# create a new model with these weights

model_uniform = Net()

model_uniform.apply(weights_init_uniform)

 

‘’’

Net(

  (fc1): Linear(in_features=784, out_features=256, bias=True)

  (fc2): Linear(in_features=256, out_features=128, bias=True)

  (fc3): Linear(in_features=128, out_features=10, bias=True)

  (dropout): Dropout(p=0.2)

)

‘’’

 

# evaluate behavior

helpers.compare_init_weights([(model_uniform, 'Uniform Weights')],

                             'Uniform Baseline',

                             train_loader,

                             valid_loader)

 

# takes in a module and applies the specified weight initialization

def weights_init_uniform_center(m):

    classname = m.__class__.__name__

    # for every Linear layer in a model..

    if classname.find('Linear') != -1: # using find if the word “Linear” does not exits (?) = -1

        # apply a centered, uniform distribution to the weights

        m.weight.data.uniform_(-0.5, 0.5)

        m.bias.data.fill_(0)

 

# create a new model with these weights

model_centered = Net()

model_centered.apply(weights_init_uniform_center)

 

‘’’

Net(

  (fc1): Linear(in_features=784, out_features=256, bias=True)

  (fc2): Linear(in_features=256, out_features=128, bias=True)

  (fc3): Linear(in_features=128, out_features=10, bias=True)

  (dropout): Dropout(p=0.2)

)

‘’’

 

# takes in a module and applies the specified weight initialization

def weights_init_uniform_rule(m):

    classname = m.__class__.__name__

    # for every Linear layer in a model..

    if classname.find('Linear') != -1:

        # get the number of the inputs

        n = m.in_features

        y = 1.0/np.sqrt(n)

        m.weight.data.uniform_(-y, y)

        m.bias.data.fill_(0)

 

# create a new model with these weights

model_rule = Net()

model_rule.apply(weights_init_uniform_rule)

 

‘’’

Net(

  (fc1): Linear(in_features=784, out_features=256, bias=True)

  (fc2): Linear(in_features=256, out_features=128, bias=True)

  (fc3): Linear(in_features=128, out_features=10, bias=True)

  (dropout): Dropout(p=0.2)

)

‘’’

 

# compare these two models

model_list = [(model_centered, 'Centered Weights [-0.5, 0.5)'),

              (model_rule, 'General Rule [-y, y)')]

 

# evaluate behavior

helpers.compare_init_weights(model_list,

                             '[-0.5, 0.5) vs [-y, y)',

                             train_loader,

                             valid_loader)

 

# helpers.hist_dist('Random Normal (mean=0.0, stddev=1.0)', np.random.normal(size=[1000]))

 

## complete this function

def weights_init_normal(m):

    '''Takes in a module and initializes all linear layers with weight

       values taken from a normal distribution.'''

   

    classname = m.__class__.__name__

    # for every Linear layer in a model

    # m.weight.data shoud be taken from a normal distribution

    # m.bias.data should be 0

    # get the number of the inputs

    if classname.find('Linear') != -1:

        # get the number of the inputs

        y = m.in_features

        # y =

        m.weight.data.normal_(0.0,1/np.sqrt(y))

        m.bias.data.fill_(0)

 

# create a new model with these weights

model_rule = Net()

model_rule.apply(weights_init_normal)

 

‘’’

Net(

  (fc1): Linear(in_features=784, out_features=256, bias=True)

  (fc2): Linear(in_features=256, out_features=128, bias=True)

  (fc3): Linear(in_features=128, out_features=10, bias=True)

  (dropout): Dropout(p=0.2)

)

‘’’

 

## -- no need to change code below this line -- ##

 

# create a new model with the rule-based, uniform weights

model_uniform_rule = Net()

model_uniform_rule.apply(weights_init_uniform_rule)

 

# create a new model with the rule-based, NORMAL weights

model_normal_rule = Net()

model_normal_rule.apply(weights_init_normal)

 

‘’’

Net(

  (fc1): Linear(in_features=784, out_features=256, bias=True)

  (fc2): Linear(in_features=256, out_features=128, bias=True)

  (fc3): Linear(in_features=128, out_features=10, bias=True)

  (dropout): Dropout(p=0.2)

)

‘’’

 

# compare the two models

model_list = [(model_uniform_rule, 'Uniform Rule [-y, y)'),

              (model_normal_rule, 'Normal Distribution')]

 

# evaluate behavior

helpers.compare_init_weights(model_list,

                             'Uniform vs Normal',

                             train_loader,

                             valid_loader)
