'''
Hidden State Activation

ℎ<𝑡>=𝑔(𝑊_ℎ[ℎ<𝑡−1>,𝑥<𝑡>]+𝑏ℎ)
ℎ<𝑡>=𝑔(𝑊_ℎℎℎ<𝑡−1>⊕𝑊ℎ_𝑥𝑥<𝑡>+𝑏ℎ)

𝑊_ℎ in the first formula is denotes the horizontal concatenation of 𝑊ℎℎ and 𝑊ℎ𝑥 from the second formula.
𝑊ℎ in the first formula is then multiplied by [ℎ<𝑡−1>,𝑥<𝑡>], another concatenation of parameters from the second formula but this time in a different direction, i.e vertical!
'''

# random Initializations

w_hh = np.random.standard_normal((3,2))
w_hx = np.random.standard_normal((3,3))

w_h2 = np.hstack((w_hh, w_hx)) # or w_h1 = np.concatenate((w_hh, w_hx), axis=1)

'''
[ℎ<𝑡−1>,𝑥<𝑡>]=[ℎ<𝑡−1> / 𝑥<𝑡>]

We will ignore the bias term 𝑏ℎ and the activation function 𝑔( )

because the transformation will be identical for each formula. So what we really want to compare is the result of the following parameters inside each formula:

𝑊ℎ[ℎ<𝑡−1>,𝑥<𝑡>] ⇔ 𝑊ℎℎℎ<𝑡−1>⊕𝑊ℎ𝑥𝑥<𝑡>
'''

w_hh = np.full((3, 2), 1)  # returns an array of size 3x2 filled with all 1s
w_hx = np.full((3, 3), 9)  # returns an array of size 3x3 filled with all 9s
h_t_prev = np.full((2, 1), 1)  # returns an array of size 2x1 filled with all 1s
x_t = np.full((3, 1), 9)       # returns an array of size 3x1 filled with all 9s


# 𝑊ℎ[ℎ<𝑡−1>,𝑥<𝑡>]
stack_1 = np.hstack((w_hh, w_hx))
stack_2 = np.vstack((h_t_prev, x_t))

formula_1 = np.matmul(np.hstack((w_hh, w_hx)), np.vstack((h_t_prev, x_t)))

# 𝑊ℎℎℎ<𝑡−1>⊕𝑊ℎ𝑥𝑥<𝑡>
mul_1 = np.matmul(w_hh, h_t_prev)
mul_2 = np.matmul(w_hx, x_t)

formula_2 = np.matmul(w_hh, h_t_prev) + np.matmul(w_hx, x_t)

'''
Working with JAX numpy and calculating perplexity: Ungraded Lecture Notebook
'''
import numpy
import trax
import trax.fastmath.numpy as np

# Setting random seeds
trax.supervised.trainer_lib.init_random_number_generators(32)
numpy.random.seed(32)

'''
The rationale behind this change is that you will be using Trax's numpy (which is compatible with JAX) far more often. Trax's numpy supports most of the same functions as the regular numpy so the change won't be noticeable in most cases.

One important change to take into consideration is that the types of the resulting objects will be different depending on the version of numpy. With regular numpy you get numpy.ndarray but with Trax's numpy you will get jax.interpreters.xla.DeviceArray.
'''

trax_numpy_array = trax.fastmath.numpy.array(numpy_array)

'''
predictions : batch of tensors corresponding to lines of text predicted by the model.
targets : batch of actual tensors corresponding to lines of text.
'''

# Load from .npy files
predictions = numpy.load('predictions.npy')
targets = numpy.load('targets.npy')

# Cast to jax.interpreters.xla.DeviceArray
predictions = np.array(predictions)
targets = np.array(targets)

reshaped_targets = tl.one_hot(targets, predictions.shape[-1])
    #trax's one_hot function takes the input as one_hot(x, n_categories, dtype=optional)
print(f'reshaped_targets has shape: {reshaped_targets.shape}')

# Total Log Perplexity
total_log_ppx = np.sum(predictions * reshaped_targets, axis= -1)

'''
Now you will need to account for the padding so this metric is not artificially deflated (since a lower perplexity means a better model). For identifying which elements are padding and which are not, you can use np.equal() and get a tensor with 1s in the positions of actual values and 0s where there are paddings.
'''

non_pad = 1.0 - np.equal(targets, 0)
print(f'non_pad has shape: {non_pad.shape}\n')
print(f'non_pad looks like this: \n\n {non_pad}')

real_log_ppx = total_log_ppx * non_pad
print(f'real perplexity still has shape: {real_log_ppx.shape}')

'''
real perplexity still has shape: (32, 64)
'''
print(f'log perplexity tensor before filtering padding: \n\n {total_log_ppx}\n')
print(f'log perplexity tensor after filtering padding: \n\n {real_log_ppx}')

log_ppx = np.sum(real_log_ppx) / np.sum(non_pad)
log_ppx = -log_ppx
print(f'The log perplexity and perplexity of the model are respectively: {log_ppx} and {np.exp(log_ppx)}')
