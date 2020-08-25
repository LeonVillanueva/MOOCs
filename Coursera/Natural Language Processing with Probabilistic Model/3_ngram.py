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
