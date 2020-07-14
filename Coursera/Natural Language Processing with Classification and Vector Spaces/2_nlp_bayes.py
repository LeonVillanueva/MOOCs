# P (A|B) = P(B|A) * P (A) / P (B)

# laplacian Smoothing - naive bayes inference condition rule
# P (W | class) = (freq (W, class) + 1) / (N_classes + V)
# where V is the number of unique words in the vocabulary, ensures a non-zero
# where N is the frequency of all words in the class

# log likelihood = SUM ( P(W|Positive) / P(W|Negative))
# a priori = prior ratio

# sentiment probabilities = multiplication -> numerical underflow
# log (a) + log (b) = log (a * b) = log (a priori) + log (likelihood)
# where log (likelihood) = SUM ( log (( P(W|Positive) / P(W|Negative))))

# https://www.coursera.org/learn/classification-vector-spaces-in-nlp/ungradedLab/NuNV6/visualizing-likelihoods-and-confidence-ellipses

'''
 0. dataset cleanup
 1. preprocess data
 2. freq (w|class), laplacian (freq_w+1) / (class+v)
 3. P (w|positive) and P (w|negative)
 4. lambda w = log (P(w|pos) / P(w|pos))
 5. compute log
'''

'''
P(A|X) / P(B|Z) ~= P(A) / P(B) * SUM ( P(x|A) / P(x|B))
: x is component of X
'''

# Assumptions : word independence

for y, tweet in zip(ys, tweets):
    for word in process_tweet(tweet):
        pair = (word, y)
        if pair in result:
            result[pair] += 1
        else:
            result[pair] = 1


def train_naive_bayes(freqs, train_x, train_y):
    '''
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # calculate V, the number of unique words in the vocabulary
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # calculate N_pos and N_neg
    N_pos = N_neg = 0
    for pair in freqs.keys():
        # if the label is positive (greater than zero)
        if pair[1] > 0:

            # Increment the number of positive words by the count for this (word, label) pair
            N_pos += 1

        # else, the label is negative
        else:

            # increment the number of negative words by the count for this (word,label) pair
            N_neg += 1

    # Calculate D, the number of documents
    D = len (train_y)

    # Calculate D_pos, the number of positive documents (*hint: use sum(<np_array>))
    D_pos = sum (train_y)

    # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
    D_neg = D - D_pos

    # Calculate logprior
    logprior = np.log (D_pos) - np.log (D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word
        if (word, 1.0) in freqs:
            freq_pos = freqs[(word, 1.0)]
        else:
            freq_pos = 0
        if (word, 0.0) in freqs:
            freq_neg = freqs[(word, 0.0)]
        else:
            freq_neg = 0

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos+1) / (D_pos+V)
        p_w_neg = (freq_neg+1) / (D_neg+V)

        # calculate the log likelihood of the word
        loglikelihood[word] = np.log (p_w_pos / p_w_neg)

    ### END CODE HERE ###

    return logprior, loglikelihood