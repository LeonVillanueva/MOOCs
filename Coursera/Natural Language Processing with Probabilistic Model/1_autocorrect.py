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

def insert_letter(word, verbose=False):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    insert_l = []
    split_l = []

    for char in range(len(word)):
        L = ''.join(word[0:char])
        R = ''.join(word[char:])
        split_l.append ((L, R))

    chars = [char for char in word]
    alphabet = [letter for letter in letters]

    for i in range(len(chars)+1):
        for j in range(len(alphabet)):
            insert_l.append( ''.join(chars[:i]) + alphabet[j] + ''.join(chars[i:]))

    if verbose: print(f"Input word {word} \nsplit_l = {split_l} \ninsert_l = {insert_l}")

    return insert_l

def edit_one_letter(word, allow_switches = True):
    edit_one_set = set()

    list_a = insert_letter (word)
    list_b = delete_letter (word)
    list_c = switch_letter (word)
    list_d = replace_letter (word)

    edit_one_set = set (list_a+list_b+list_c+list_d)

    return edit_one_set

def edit_two_letters(word, allow_switches = True):
    edit_two_set = set()

    list_of_sets = []
    first = edit_one_letter (word)
    list_of_sets.append (first)
    for i in list(first):
        list_of_sets.append(edit_one_letter(i))

    edit_two_set = set.union(*list_of_sets)

    return edit_two_set

def get_corrections(word, probs, vocab, n=2, verbose = False):
    suggestions = []
    n_best = []

    suggestions = list((word in vocab and word)
                       or edit_one_letter(word).intersection(vocab)
                       or edit_two_letters(word).intersection(vocab))

    n_list = [(s,probs[s]) for s in list((suggestions))]
    n_best = sorted(n_list, key=lambda x: x[1], reverse=True)[:n]
    suggestions = {x[0] for x in n_best}

    if verbose: print("entered word = ", word, "\nsuggestions = ", suggestions)

    return n_best
