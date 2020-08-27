# NGRAM

'''
text corpus = language model
    estiamte probabilities from a sequence of words

    text corpus + language + ... autocomplete

    speech recognition
    spelling correction
    augmentative communication

1. process text corpus
2. out-of-vocabulary words
3. smoothing previously unseen words
4. propensity model

N-Gram
    sequence of words
        corpus
            unigram set []
            bigram set [], must appear next to each other
            trigram set []

            probability of bigram

            P(y|x) = C(x y) / C(x)
            n-grams
            P(W_n|W^n-1) = C(W^n-1 W_n) / C(W^n-1)

N-gram probabilities
    guess :
    given n-gram:
        = log P(W_n|W^n-1) + P(W_n-1|W^n-2) ... n-k = 1

    P(A,B,C,D) = P(A)P(B|A)P(C|A,B)P(D|A,B,C) ideal scenario
    simplified, because it's likely the exact n-gram prior is not in the samples

    P(A)P(B|A)P(C|B)P(D|C)
    markov assumption

Starting and Ending sentences
    <s>*n word_start; where n = n-gram amount
    word_end </s>*n; where n = n-gram amount

    makes it easier to count n-gram probabilities

N-gram Language model
    count matrix
    probability matrix
    language models
    log to avoid probability underflow
    generative laguage models

    count matrix
        P (Wn | W^n-1_n-N+1) = C (W^n-1_n-N+1, Wn) / C (W^n-1_n-N+1)
        use a sliding window
            ROW-concept (Wn-1)
            COLUMN-concept (Wn)
     probability matrix
        update count matrix
        divide by the row sum

    definition of a language models
        splitting sentence given n-grams
        extract n-1 gram of a language

language model evaluation (perplexity)

    PP(W) = P (s1,s2,...,s_m)^(-1/m)
        s = ith sentence
        m = number of words, + </s>
        lower perplexity more likely human

    perplexity for bigram model
    sometimes use log perplexity (base 2)

out-of-vocabulary (oov) words
    <unk> unknown
    closed or open corpus
    min frequency = threshold of <unk>

    vocabulary criteria
        minimum word frequency f
        max vocabulary size
        use <unk> sparingly, biases to lower perplexity

    perplexity = compare only those with same vocabulary

smoothing
    train n-grams from a specific corpus
    n-gram made of known words that might be missing in the training corpus

    laplacian smoothing
        n-gram probability (+1/+V)
    add k-smoothing
        n-gram probability (+k/+k*V)

        kneser, good-turing

    back-off, lower n-gram probability
        3-gram > 2-gram > 1-gram
        katz backoff discounted
        'stupid' backoff

    linear interpolation
        LAMBDA p * n-gram + LAMBDA p * n-1-gram + ... + LAMBDA p * 1-gram
        sum (LAMDA) == 1
'''


import nltk
import re
nltk.download('punkt')
corpus = "Learning% makes 'me' happy. I am happy be-cause I am learning! :)"
corpus = corpus.lower()
corpus = re.sub(r"[^a-zA-Z0-9.?! ]+", "", corpus)

input_date="Sat May  9 07:33:35 CEST 2020"
date_parts = input_date.split(" ")
time_parts = date_parts[4].split(":")

sentence = 'i am happy because i am learning.'
tokenized_sentence = nltk.word_tokenize(sentence)
sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
word_lengths = [(word, len(word)) for word in sentence] # list of word, word_lengths

def sentence_to_trigram(tokenized_sentence):
    # note that the last position of i is 3rd to the end
    for i in range(len(tokenized_sentence) - 3 + 1):
        # the sliding window starts at position i and contains 3 words
        trigram = tokenized_sentence[i : i + 3]
        print(trigram)
tokenized_sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']

# tokenize then n-gram

fourgram = ['i', 'am', 'happy','because']
trigram = fourgram[0:-1] # Get the elements from 0, included, up to
print(trigram)
'''
    ['i', 'am', 'happy']

'''

n = 3
tokenized_sentence = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
tokenized_sentence = ["<s>"] * (n - 1) + tokenized_sentence + ["</s>"]
print(tokenized_sentence)
    # have to prepend n-1 characters so <s> <s> word, if n=3
    # the last word is always the first word of the corpus
    # have to append </s> in a similar manner

prefix = ('i', 'am', 'happy')
word = 'because'# note here the syntax for creating a tuple for a single word
n_gram = prefix + (word,)


import numpy as np
import pandas as pd
from collections import defaultdict
def single_pass_trigram_count_matrix(corpus):
    """
    Creates the trigram count matrix from the input corpus in a single pass through the corpus.
    Args:
        corpus: Pre-processed and tokenized corpus.
    Returns:
        bigrams: list of all bigram prefixes, row index
        vocabulary: list of all found words, the column index
        count_matrix: pandas dataframe with bigram prefixes as rows,
                      vocabulary words as columns
                      and the counts of the bigram/word combinations (i.e. trigrams) as values
    """
    bigrams = []
    vocabulary = []
    count_matrix_dict = defaultdict(dict)

    # go through the corpus once with a sliding window
    for i in range(len(corpus) - 3 + 1):
        # the sliding window starts at position i and contains 3 words
        trigram = tuple(corpus[i : i + 3])
        bigram = trigram[0 : -1]
        if not bigram in bigrams:
            bigrams.append(bigram)
        last_word = trigram[-1]
        if not last_word in vocabulary:
            vocabulary.append(last_word)
        if (bigram,last_word) not in count_matrix_dict:
            count_matrix_dict[bigram,last_word] = 0
        count_matrix_dict[bigram,last_word] += 1

    # convert the count_matrix to np.array to fill in the blanks
    count_matrix = np.zeros((len(bigrams), len(vocabulary)))
    for trigram_key, trigam_count in count_matrix_dict.items():
        count_matrix[bigrams.index(trigram_key[0]), \
                     vocabulary.index(trigram_key[1])]\
        = trigam_count

    # np.array to pandas dataframe conversion
    count_matrix = pd.DataFrame(count_matrix, index=bigrams, columns=vocabulary)
    return bigrams, vocabulary, count_matrix

corpus = ['i', 'am', 'happy', 'because', 'i', 'am', 'learning', '.']
bigrams, vocabulary, count_matrix = single_pass_trigram_count_matrix(corpus)
print(count_matrix)

'''
                  happy  because    i   am  learning    .
(i, am)             1.0      0.0  0.0  0.0       1.0  0.0
(am, happy)         0.0      1.0  0.0  0.0       0.0  0.0
(happy, because)    0.0      0.0  1.0  0.0       0.0  0.0
(because, i)        0.0      0.0  0.0  1.0       0.0  0.0
(am, learning)      0.0      0.0  0.0  0.0       0.0  1.0
'''

# probabilities matrix

# create the probability matrix from the count matrix
# delete each row by its sum
row_sums = count_matrix.sum(axis=1)
prob_matrix = count_matrix.div(row_sums, axis=0)
print(prob_matrix)

# find the probability of a trigram in the probability matrix
trigram = ('i', 'am', 'happy')

# find the prefix bigram
bigram = trigram[:-1]
print(f'bigram: {bigram}')
# find the last word of the trigram
word = trigram[-1]
print(f'word: {word}')
# we are using the pandas dataframes here, column with vocabulary word comes first, row with the prefix bigram second
trigram_probability = prob_matrix[word][bigram]
print(f'trigram_probability: {trigram_probability}')

# lists all words in vocabulary starting with a given prefix
vocabulary = ['i', 'am', 'happy', 'because', 'learning', '.', 'have', 'you', 'seen','it', '?']
starts_with = 'ha'
print(f'words in vocabulary starting with prefix: {starts_with}\n')
for word in vocabulary:
    if word.startswith(starts_with):
        print(word)

# language model evaluation
# we only need train and validation %, test is the remainder
import random
def train_validation_test_split(data, train_percent, validation_percent):
    """
    Splits the input data to  train/validation/test according to the percentage provided

    Args:
        data: Pre-processed and tokenized corpus, i.e. list of sentences.
        train_percent: integer 0-100, defines the portion of input corpus allocated for training
        validation_percent: integer 0-100, defines the portion of input corpus allocated for validation

        Note: train_percent + validation_percent need to be <=100
              the reminder to 100 is allocated for the test set

    Returns:
        train_data: list of sentences, the training part of the corpus
        validation_data: list of sentences, the validation part of the corpus
        test_data: list of sentences, the test part of the corpus
    """
    # fixed seed here for reproducibility
    random.seed(87)

    # reshuffle all input sentences
    random.shuffle(data)

    train_size = int(len(data) * train_percent / 100)
    train_data = data[0:train_size]

    validation_size = int(len(data) * validation_percent / 100)
    validation_data = data[train_size:train_size + validation_size]

    test_data = data[train_size + validation_size:]

    return train_data, validation_data, test_data

data = [x for x in range (0, 100)]

train_data, validation_data, test_data = train_validation_test_split(data, 80, 10)
print("split 80/10/10:\n",f"train data:{train_data}\n", f"validation data:{validation_data}\n",
      f"test data:{test_data}\n")

train_data, validation_data, test_data = train_validation_test_split(data, 98, 1)
print("split 98/1/1:\n",f"train data:{train_data}\n", f"validation data:{validation_data}\n",
      f"test data:{test_data}\n")

# perplexity
# to calculate the exponent, use the following syntax
p = 10 ** (-250)
M = 100
perplexity = p ** (-1 / M)
print(perplexity)
