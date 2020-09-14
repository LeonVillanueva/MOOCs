'''
    Trax
    class implementation
'''

class MyClass (Object): # subclass of Object class
    def __init__ (self, y):
        self.y = y
    def my_method (self, x):
        return x + self.y
    def __call__ (self, x): # if already initialized
        return self.my_method (x)

class SubClass (MyClass): # MyClass is the parent class
    def my_method (self, x):
        return x + self.y**2

'''
    Trax layers
'''

# https://github.com/google/trax
#!pip install trax==1.3.1 Use this version for this notebook

import numpy as np  # regular ol' numpy

from trax import layers as tl  # core building block
from trax import shapes  # data signatures: dimensionality and type
from trax import fastmath  # uses jax, offers numpy on steroids

# Trax version 1.3.1 or better
!pip list | grep trax

# Layers
# Create a relu trax layer
relu = tl.Relu()

# Inspect properties
print("-- Properties --")
print("name :", relu.name)
print("expected inputs :", relu.n_in)
print("promised outputs :", relu.n_out, "\n")

# Inputs
x = np.array([-2, -1, 0, 1, 2])
print("-- Inputs --")
print("x :", x, "\n")

# Outputs
y = relu(x)
print("-- Outputs --")
print("y :", y)

'''
-- Properties --
name : Relu
expected inputs : 1
promised outputs : 1

-- Inputs --
x : [-2 -1  0  1  2]

-- Outputs --
y : [0 0 0 1 2]
'''

# Create a concatenate trax layer
concat = tl.Concatenate()
print("-- Properties --")
print("name :", concat.name)
print("expected inputs :", concat.n_in)
print("promised outputs :", concat.n_out, "\n")

# Inputs
x1 = np.array([-10, -20, -30])
x2 = x1 / -10
print("-- Inputs --")
print("x1 :", x1)
print("x2 :", x2, "\n")

# Outputs
y = concat([x1, x2])
print("-- Outputs --")
print("y :", y)

# Configure a concatenate layer
concat_3 = tl.Concatenate(n_items=3)  # configure the layer's expected inputs
print("-- Properties --")
print("name :", concat_3.name)
print("expected inputs :", concat_3.n_in)
print("promised outputs :", concat_3.n_out, "\n")

# Inputs
x1 = np.array([-10, -20, -30])
x2 = x1 / -10
x3 = x2 * 0.99
print("-- Inputs --")
print("x1 :", x1)
print("x2 :", x2)
print("x3 :", x3, "\n")

# Outputs
y = concat_3([x1, x2, x3])
print("-- Outputs --")
print("y :", y)

'''
-- Properties --
name : Concatenate
expected inputs : 3
promised outputs : 1

-- Inputs --
x1 : [-10 -20 -30]
x2 : [1. 2. 3.]
x3 : [0.99 1.98 2.97]

-- Outputs --
y : [-10.   -20.   -30.     1.     2.     3.     0.99   1.98   2.97]
'''

# Uncomment any of them to see information regarding the function
help(tl.LayerNorm)
help(shapes.signature)

# Layer initialization
norm = tl.LayerNorm()
# You first must know what the input data will look like
x = np.array([0, 1, 2, 3], dtype="float")

# Use the input data signature to get shape and type for initializing weights and biases
norm.init(shapes.signature(x)) # We need to convert the input datatype from usual tuple to trax ShapeDtype

print("Normal shape:",x.shape, "Data Type:",type(x.shape))
print("Shapes Trax:",shapes.signature(x),"Data Type:",type(shapes.signature(x)))

# Inspect properties
print("-- Properties --")
print("name :", norm.name)
print("expected inputs :", norm.n_in)
print("promised outputs :", norm.n_out)
# Weights and biases
print("weights :", norm.weights[0])
print("biases :", norm.weights[1], "\n")

# Inputs
print("-- Inputs --")
print("x :", x)

# Outputs
y = norm(x)
print("-- Outputs --")
print("y :", y)

'''
Normal shape: (4,) Data Type: <class 'tuple'>
Shapes Trax: ShapeDtype{shape:(4,), dtype:float64} Data Type: <class 'trax.shapes.ShapeDtype'>
-- Properties --
name : LayerNorm
expected inputs : 1
promised outputs : 1
weights : [1. 1. 1. 1.]
biases : [0. 0. 0. 0.]

-- Inputs --
x : [0. 1. 2. 3.]
-- Outputs --
y : [-1.3416404  -0.44721344  0.44721344  1.3416404 ]
'''

# Define a custom layer
# In this example you will create a layer to calculate the input times 2

def TimesTwo():
    layer_name = "TimesTwo" #don't forget to give your custom layer a name to identify

    # Custom function for the custom layer
    def func(x):
        return x * 2

    return tl.Fn(layer_name, func)


# Test it
times_two = TimesTwo()

# Inspect properties
print("-- Properties --")
print("name :", times_two.name)
print("expected inputs :", times_two.n_in)
print("promised outputs :", times_two.n_out, "\n")

# Inputs
x = np.array([1, 2, 3])
print("-- Inputs --")
print("x :", x, "\n")

# Outputs
y = times_two(x)
print("-- Outputs --")
print("y :", y)

'''
-- Properties --
name : TimesTwo
expected inputs : 1
promised outputs : 1

-- Inputs --
x : [1 2 3]

-- Outputs --
y : [2 4 6]
'''

'''
Serial Combinator

This is the most common and easiest to use. For example could build a simple neural network by combining layers into a single layer using the Serial combinator. This new layer then acts just like a single layer, so you can inspect intputs, outputs and weights. Or even combine it into another layer! Combinators can then be used as trainable models. Try adding more layers

Note:As you must have guessed, if there is serial combinator, there must be a parallel combinator as well. Do try to explore about combinators and other layers from the trax documentation and look at the repo to understand how these layers are written.
'''

# Serial combinator
serial = tl.Serial(
    tl.LayerNorm(),         # normalize input
    tl.Relu(),              # convert negative values to zero
    times_two,              # the custom layer you created above, multiplies the input recieved from above by 2

    ### START CODE HERE
      tl.Dense(n_units=2),  # try adding more layers. eg uncomment these lines
#     tl.Dense(n_units=1),  # Binary classification, maybe? uncomment at your own peril
      tl.LogSoftmax()       # Yes, LogSoftmax is also a layer
    ### END CODE HERE
)

# Initialization
x = np.array([-2, -1, 0, 1, 2]) #input
serial.init(shapes.signature(x)) #initialising serial instance

print("-- Serial Model --")
print(serial,"\n")
print("-- Properties --")
print("name :", serial.name)
print("sublayers :", serial.sublayers)
print("expected inputs :", serial.n_in)
print("promised outputs :", serial.n_out)
print("weights & biases:", serial.weights, "\n")

# Inputs
print("-- Inputs --")
print("x :", x, "\n")

# Outputs
y = serial(x)
print("-- Outputs --")
print("y :", y)

'''
-- Serial Model --
Serial[
  LayerNorm
  Relu
  TimesTwo
  Dense_2
  LogSoftmax
]

-- Properties --
name : Serial
sublayers : [LayerNorm, Relu, TimesTwo, Dense_2, LogSoftmax]
expected inputs : 1
promised outputs : 1
weights & biases: [(DeviceArray([1, 1, 1, 1, 1], dtype=int32), DeviceArray([0, 0, 0, 0, 0], dtype=int32)), (), (), (DeviceArray([[ 0.26194617, -0.7237064 ],
             [ 0.7749321 , -0.39890552],
             [ 0.12293135, -0.7340453 ],
             [ 0.479154  ,  0.5708689 ],
             [ 0.4335323 ,  0.8114192 ]], dtype=float32), DeviceArray([-2.0563094e-08,  6.7215638e-07], dtype=float32)), ()]

-- Inputs --
x : [-2 -1  0  1  2]

-- Outputs --
y : [-1.4621532  -0.26362276]
'''

# Numpy vs fastmath.numpy have different data types
# Regular ol' numpy
x_numpy = np.array([1, 2, 3])
print("good old numpy : ", type(x_numpy), "\n")

# Fastmath and jax numpy
x_jax = fastmath.numpy.array([1, 2, 3])
print("jax trax numpy : ", type(x_jax))

'''
good old numpy :  <class 'numpy.ndarray'>

jax trax numpy :  <class 'jax.interpreters.xla.DeviceArray'>
'''

'''
Mean, Embedding Layers
'''

# Training using grad

trax.math.grad (f)

y = model (x)
grads = grad(y.forward)(y.weights, x) #argument for grad, arguments for returned from grad

weights -= alpha * grads  #gradient descent #in a loop


#data generators

import random
import numpy as np

# Example of traversing a list of indexes to create a circular list
a = [1, 2, 3, 4]
b = [0] * 10

a_size = len(a)
b_size = len(b)
lines_index = [*range(a_size)] # is equivalent to [i for i in range(0,a_size)], the difference being the advantage of using * to pass values of range iterator to list directly
index = 0                      # similar to index in data_generator below
for i in range(b_size):        # `b` is longer than `a` forcing a wrap
    # We wrap by resetting index to 0 so the sequences circle back at the end to point to the first index
    if index >= a_size:
        index = 0

    b[i] = a[lines_index[index]]     #  `indexes_list[index]` point to a index of a. Store the result in b
    index += 1

print(b)

# Example of traversing a list of indexes to create a circular list
a = [1, 2, 3, 4]
b = []

a_size = len(a)
b_size = 10
lines_index = [*range(a_size)]
print("Original order of index:",lines_index)

# if we shuffle the index_list we can change the order of our circular list
# without modifying the order or our original data
random.shuffle(lines_index) # Shuffle the order
print("Shuffled order of index:",lines_index)

print("New value order for first batch:",[a[index] for index in lines_index])
batch_counter = 1
index = 0                # similar to index in data_generator below
for i in range(b_size):  # `b` is longer than `a` forcing a wrap
    # We wrap by resetting index to 0
    if index >= a_size:
        index = 0
        batch_counter += 1
        random.shuffle(lines_index) # Re-shuffle the order
        print("\nShuffled Indexes for Batch No.{} :{}".format(batch_counter,lines_index))
        print("Values for Batch No.{} :{}".format(batch_counter,[a[index] for index in lines_index]))

    b.append(a[lines_index[index]])     #  `indexes_list[index]` point to a index of a. Store the result in b
    index += 1
print()
print("Final value of b:",b)

def data_generator(batch_size, data_x, data_y, shuffle=True):
    '''
      Input:
        batch_size - integer describing the batch size
        data_x - list containing samples
        data_y - list containing labels
        shuffle - Shuffle the data order
      Output:
        a tuple containing 2 elements:
        X - list of dim (batch_size) of samples
        Y - list of dim (batch_size) of labels
    '''

    data_lng = len(data_x) # len(data_x) must be equal to len(data_y)
    index_list = [*range(data_lng)] # Create a list with the ordered indexes of sample data


    # If shuffle is set to true, we traverse the list in a random way
    if shuffle:
        random.shuffle(index_list) # Inplace shuffle of the list

    index = 0 # Start with the first element
    # START CODE HERE
    # Fill all the None values with code taking reference of what you learned so far
    while True:
        X = [0] * batch_size  # We can create a list with batch_size elements.
        Y = [0] * batch_size  # We can create a list with batch_size elements.

        for i in range(batch_size):

            # Wrap the index each time that we reach the end of the list
            if index >= data_lng:
                index = 0
                # Shuffle the index_list if shuffle is true
                if shuffle:
                    random.shuffle(index_list) # re-shuffle the order

            X[i] = data_x[index_list[index]]  # We set the corresponding element in x
            Y[i] = data_y[index_list[index]] # We set the corresponding element in x
    # END CODE HERE
            index += 1

        yield((X, Y))
