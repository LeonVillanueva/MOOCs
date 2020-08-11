# autocorrect

'''
1. identify
2. n-distance away (insert, delete, switch, replace)
3. filter candidates
4. calculate word prob

text_lowercase = text.lower()
words = re.findall(r'\w+', text_lowercase)

p(w) = c(w)/v
    v : corpus size

minimum edit distance, can have different cost / weight per edit
np.sum ([edit]*[cost])

edit cost table : minimum edit distance : levenshtein distance
'''
import re

def process_data(file_name):
    words = []
    with open(file_name,'r') as file:
        doc = file.read ()
        words = re.findall ('\w+', doc)
    words = list (map (str.lower,words))
    return words

def get_count(word_l):
    word_count_dict = {}
    for word in word_l:
        if word in word_count_dict:
            word_count_dict[word] += 1
        else:
            word_count_dict[word] = 1
    return word_count_dict

def get_probs(word_count_dict):
    probs = {}
    for word in word_count_dict.keys():
        probs[word] = word_count_dict[word] / sum (word_count_dict.values())
    return probs

def delete_letter(word):
    delete_l = []
    split_l = []

    for char in range(len(word)):
        L = ''.join(word[0:char])
        R = ''.join(word[char:])
        split_l.append ((L, R))
    for char in range(len(word)):
        chars = [c for c in word]
        chars.pop (char)
        delete_l.append (''.join(chars))

    return delete_l

def switch_letter(word, verbose=False):
    switch_l = []
    split_l = []

    for char in range(len(word)):
        L = ''.join(word[0:char])
        R = ''.join(word[char:])
        split_l.append ((L, R))

    chars = [char for char in word]
    length = len(chars)
    if length >= 2:
        for i in range(length-1):
            first = i
            second = i+1
            switch_l.append(''.join(chars[:first])+chars[second]+chars[first]+''.join(chars[second+1:]))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nswitch_l = {switch_l}")
    return switch_l

def replace_letter(word, verbose=False):

    letters = 'abcdefghijklmnopqrstuvwxyz'
    replace_l = []
    split_l = []

    for char in range(len(word)):
        L = ''.join(word[0:char])
        R = ''.join(word[char:])
        split_l.append ((L, R))

    for i in range (len(word)):
        chars = [char for char in word]
        for j in range (len(letters)):
            chars[i] = [z for z in letters][j]
            replace_l.append (''.join(chars))

    replace_l = filter(lambda i: i not in word, replace_l)
    replace_set = set (replace_l)

    replace_l = sorted(list(replace_set))

    if verbose: print(f"Input word = {word} \nsplit_l = {split_l} \nreplace_l {replace_l}")

    return replace_l
