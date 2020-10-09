'''
transformer (attention) model

problems with rnn
  no parallel computing / fully sequential
  vanishing gradient problem

transformer:
    multi-headed attention
    positional encoding

T5, text-to-text transformation
    text summarization
    q/a
    chatbots

GPT2/3, BERT

T5, classification
    regression
    classification

Dot product attention
    queries by key matrix

Queries, Keys, Values
    I AM HAPPY > "I"
    embedding
    Linear Q, K, V
    softmax (QK.T) * V = Z
    softmax (QK.T) = probability OR attention weights
    K: L_K, D, Q: L_Q, D
    Key: Embedding of the Original
    Query: Embedding of the Translation space

    Z = attention (QK.T) * V = softmax (QK.T) * V
    attention: 2 matrix multiplication

Casual attention (decoder)
    Ways of attention
        1. encoder/decoder
        2. causal (self): text generation
        3. bidirectional
    + constant matrix (M) mass

    W = softmax (QK.T + Mask)*V:[L:L]
        Mask is a triangle matrix of 0,-inf

Multihead attention
    word
    > QKV embedding
    > scaled dot prodcut (dense layer)
    > attention + concatentation
    > linear

    'Embeddings'
    Input (QKV): [batch, length, d_model]
    Linear layer: [batch, length, n_heads*d_head]
    Transpose: [batch, n_heads, length, d_head]
    apply attention treating n_heads like a batch
    Reshape: [batch, n_heads, length, d_head]
    Transpose: [batch, length, n_heads*d_head]
    Linear layer input: [batch, length, d_model]

    Multihead (QKV) = Concat (h1,...,h_h)W0
        where h_i = Attention (QW^Q, W^K, W^V)

    multiple lookups in parallel

transformer decoder
    input
    > shift right <start>
    > input embeddings, with positional encoding [skip]
        > multihead attention
        > add & norm [skip]
        > feed forward (with ReLU), most parameters here
        > add & norm [skip]
        : repeated N times
    > dense layer, linear
    > softmax

    Decoder block
        positional embedding input
        > multihead attention
        > output vector
        > layer normalization + skip (residual connection)
        > each word feed forward layer (dense)
        > dropout (regularization)
        > layer normalization (N times)
        > layer encoder

    Feed forward layer replaces the states of RNN

    # Remember, the encoder-decoder attention layer works like multiple-headed self-attention, except that it creates its query matrix from the previous layer

    weighted cross entropy to ignore / highlight for labeled summary

'''

# Lab

import sys

import numpy as np
import scipy.special

import textwrap
wrapper = textwrap.TextWrapper(width=70)

# to print the entire np array
np.set_printoptions(threshold=sys.maxsize)

# helper functions

def create_tensor(t):
    """Create tensor from list of lists"""
    return np.array(t)


def display_tensor(t, name):
    """Display shape and tensor"""
    print(f'{name} shape: {t.shape}\n')
    print(f'{t}\n')

q = create_tensor([[1, 0, 0], [0, 1, 0]])
display_tensor(q, 'query')
k = create_tensor([[1, 2, 3], [4, 5, 6]])
display_tensor(k, 'key')
v = create_tensor([[0, 1, 0], [1, 0, 1]])
display_tensor(v, 'value')
m = create_tensor([[0, 0], [-1e9, 0]])
display_tensor(m, 'mask')

def DotProductAttention(query, key, value, mask, scale=True):
    """Dot product self-attention.
    Args:
        query (numpy.ndarray): array of query representations with shape (L_q by d)
        key (numpy.ndarray): array of key representations with shape (L_k by d)
        value (numpy.ndarray): array of value representations with shape (L_k by d) where L_v = L_k
        mask (numpy.ndarray): attention-mask, gates attention with shape (L_q by L_k)
        scale (bool): whether to scale the dot product of the query and transposed key

    Returns:
        numpy.ndarray: Self-attention array for q, k, v arrays. (L_q by L_k)
    """

    assert query.shape[-1] == key.shape[-1] == value.shape[-1], "Embedding dimensions of q, k, v aren't all the same"

    # Save depth/dimension of the query embedding for scaling down the dot product
    if scale:
        depth = query.shape[-1]
    else:
        depth = 1

    # Calculate scaled query key dot product according to formula above
    dots = np.matmul(query, np.swapaxes(key, -1, -2)) / np.sqrt(depth)

    # Apply the mask
    if mask is not None:
        dots = np.where(mask, dots, np.full_like(dots, -1e9))

    # Softmax formula implementation
    # Use scipy.special.logsumexp of masked_qkT to avoid underflow by division by large numbers
    # Note: softmax = e^(dots - logaddexp(dots)) = E^dots / sumexp(dots)
    logsumexp = scipy.special.logsumexp(dots, axis=-1, keepdims=True)

    # Take exponential of dots minus logsumexp to get softmax
    # Use np.exp()
    dots = np.exp(dots - logsumexp)

    # Multiply dots by value to get self-attention
    # Use np.matmul()
    attention = np.matmul(dots, value)

    return attention

def dot_product_self_attention(q, k, v, scale=True):
    """ Masked dot product self attention.
    Args:
        q (numpy.ndarray): queries.
        k (numpy.ndarray): keys.
        v (numpy.ndarray): values.
    Returns:
        numpy.ndarray: masked dot product self attention tensor.
    """

    # Size of the penultimate dimension of the query
    mask_size = q.shape[-2]

    # Creates a matrix with ones below the diagonal and 0s above. It should have shape (1, mask_size, mask_size)
    # Use np.tril() - Lower triangle of an array and np.ones()
    mask = np.tril(np.ones((1, mask_size, mask_size), dtype=np.bool_), k=0)

    return DotProductAttention(q, k, v, mask, scale=scale)

dot_product_self_attention(q, k, v)

'''
array([[[0.        , 1.        , 0.        ],
        [0.84967455, 0.15032545, 0.84967455]]])
'''

# Lab2

import sys
import os

import time
import numpy as np
import gin

import textwrap
wrapper = textwrap.TextWrapper(width=70)

import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp

# to print the entire np array
np.set_printoptions(threshold=sys.maxsize)

'''
INFO:tensorflow:tokens_length=568 inputs_length=512 targets_length=114 noise_density=0.15 mean_noise_span_length=3.0
'''

def PositionalEncoder(vocab_size, d_model, dropout, max_len, mode):
    """Returns a list of layers that:
    1. takes a block of text as input,
    2. embeds the words in that text, and
    3. adds positional encoding,
       i.e. associates a number in range(max_len) with
       each word in each sentence of embedded input text

    The input is a list of tokenized blocks of text

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train' or 'eval'.
    """
    # Embedding inputs and positional encoder
    return [
        # Add embedding layer of dimension (vocab_size, d_model)
        tl.Embedding(vocab_size, d_model),
        # Use dropout with rate and mode specified
        tl.Dropout(rate=dropout, mode=mode),
        # Add positional encoding layer with maximum input length and mode specified
        tl.PositionalEncoding(max_len=max_len, mode=mode)]

def FeedForward(d_model, d_ff, dropout, mode, ff_activation):
    """Returns a list of layers that implements a feed-forward block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """

    # Create feed-forward block (list) with two dense layers with dropout and input normalized
    return [
        # Normalize layer inputs
        tl.LayerNorm(),
        # Add first feed forward (dense) layer (don't forget to set the correct value for n_units)
        tl.Dense(d_ff),
        # Add activation function passed in as a parameter (you need to call it!)
        ff_activation(),  # Generally ReLU
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode),
        # Add second feed forward layer (don't forget to set the correct value for n_units)
        tl.Dense(d_model),
        # Add dropout with rate and mode specified (i.e., don't use dropout during evaluation)
        tl.Dropout(rate=dropout, mode=mode)
    ]

def DecoderBlock(d_model, d_ff, n_heads,
                 dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """

    # Add list of two Residual blocks: the attention with normalization and dropout and feed-forward blocks
    return [
      tl.Residual(
          # Normalize layer input
          tl.LayerNorm(),
          # Add causal attention
          tl.CausalAttention(d_feature, n_heads=n_heads, dropout=dropout, mode=mode)
        ),
      tl.Residual(
          # Add feed-forward block
          # We don't need to normalize the layer inputs here. The feed-forward block takes care of that for us.
          FeedForward(d_model, d_ff, dropout, mode, ff_activation)
        ),
      ]

def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
        to activations over a vocab set.
    """

    # Create stack (list) of decoder blocks with n_layers with necessary parameters
    decoder_blocks = [
        DecoderBlock(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers)]

    # Create the complete model as written in the figure
    return tl.Serial(
        # Use teacher forcing (feed output of previous step to current step)
        tl.ShiftRight(mode=mode),
        # Add embedding inputs and positional encoder
        PositionalEncoder(vocab_size, d_model, dropout, max_len, mode),
        # Add decoder blocks
        decoder_blocks,
        # Normalize layer
        tl.LayerNorm(),

        # Add dense layer of vocab_size (since need to select a word to translate to)
        # (a.k.a., logits layer. Note: activation already set by ff_activation)
        tl.Dense(vocab_size),
        # Get probabilities with Logsoftmax
        tl.LogSoftmax()
    )
