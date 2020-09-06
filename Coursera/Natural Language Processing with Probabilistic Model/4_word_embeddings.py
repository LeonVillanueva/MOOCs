# word embeddings
'''
word vector trainings
    1. word representations
    2. generate word embeddings
    3. prepare text for machine learning
    4. implement Bag of Words

basic word representations
    1. integers
    2. one-hot vectors
    3. word embeddings

word embeddings
    meaning as vectors
        relatively low dimension
        embedding meanings
            e.g. semantic distance, analogies

create word embeddings
    1. corpus
        words in context, can't just be a vocabulary
        special or general purpose set of articles : must have nuances
    2. embedding method
        machine learning model
        self supervised = data contains the "label" but without label
    3. (1) transformation (2) > word embeddings

word embedding methods
    word2vec
        shallow neural network
            continuous bag of words
            continuous n-skip-grams
    GLoVE
    fastText

    Vector representations
    Deep Learning embeddings
            polysimi
        BERT
        ELMo
        GPT-2
            high quality, domain specific

continuous bag-of-words model
    corpus > (transformation) > machine learning model = word embeddings
    semantically related words

cleaning and tokenization
    cleaning and tokenization matters
    punctuations
    numbers

    same case (all lower)
    same punctuation
    how to deal with numerical values
    special characters
    special words : hashtags, emojis

    punkt, special tokenization nltk
        nltk.download ('punkt')
        import emoji

sliding windows
    sliding windows for extracting center words in continuous bag-of-words
    generator, yield function

architecture of CBOW
    shallow of neural network
    dense fully connected

neural network structure
    NxV,Nx1 (ReLU)
    VxN,Vx1 (softmax)

'''

# Import Python libraries and helper functions (in utils2)
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils2 import sigmoid, get_batches, compute_pca, get_dict

# Download sentence tokenizer
nltk.data.path.append('.')

# Load, tokenize and process the data
import re                                                           #  Load the Regex-modul
with open('shakespeare.txt') as f:
    data = f.read()                                                 #  Read in the data
data = re.sub(r'[,!?;-]', '.',data)                                 #  Punktuations are replaced by .
data = nltk.word_tokenize(data)                                     #  Tokenize string to words
data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']    #  Lower case and drop non-alphabetical tokens
print("Number of tokens:", len(data),'\n', data[:15])               #  print data sample

# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist = nltk.FreqDist(word for word in data)
print("Size of vocabulary: ",len(fdist) )
print("Most frequent tokens: ",fdist.most_common(20) ) # print the 20 most frequent words and their freq.

# get_dict creates two dictionaries, converting words to indices and viceversa.
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
print("Size of vocabulary: ", V)

# example of word to index mapping
print("Index of the word 'king' :  ",word2Ind['king'] )
print("Word which has index 2743:  ",Ind2word[2743] )

# Print the tokenized version of the corpus
print(f'Initial list of tokens:  {data}')

# Filter tokenized corpus using list comprehension
data = [ ch.lower() for ch in data
         if ch.isalpha()
         or ch == '.'
         or emoji.get_emoji_regexp().search(ch)
       ]

# Print the tokenized and filtered version of the corpus
print(f'After cleaning:  {data}')

# Define the 'tokenize' function that will include the steps previously seen
def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)  # tokenize string to words
    data = [ ch.lower() for ch in data
             if ch.isalpha()
             or ch == '.'
             or emoji.get_emoji_regexp().search(ch)
           ]
    return data

# Define new corpus
corpus = 'I am happy because I am learning'

# Print new corpus
print(f'Corpus:  {corpus}')

# Save tokenized version of corpus into 'words' variable
words = tokenize(corpus)

# Print the tokenized version of the corpus
print(f'Words (tokens):  {words}')
