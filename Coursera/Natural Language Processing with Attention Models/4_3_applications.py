'''
question and answers
transfer learning
BERT
T5 model (GPT2/3)
single vs multi task

transfer learning
    1.feature based & fine tuning
    1a.pre-trained data
    1b.pre-training task

    CBOW - word bidirectional estimate from context

    feature based
        pre-trained
            weights as word vectors
    fine tuning
        downstream task
            initialized weights on a second training
            "adding a second layer"

    data and performance (larger > more complex model)

    label vs unlabeled data
        unlabeled
            self-supervised data
            word prediction (masking)
        secondary tasks

ELMo, GPT, BERT, T5
    history: CBOW, ELMo, GPT, BERT, T5
        1. fixed window CBOW
        2. bidirectional RNN context
        3. open AI GPT, TRANSFORMER encodedr-decoder
        4. GPT decoder only, still undirectional
        5. BERT biderictional, encoder only
           BERT, pretraining task, multi mask
    T5
        encoder-decoder
        text-to-text: classify, summarize, ask

BERT architecture
    pre-training vs fine-tuning
    a multilayer bidirectional transformer
    positional embeddings
    BERT_base

    BERT pretraining
        masked language model
        BERT is designed to pretrain deep bi-directional representations from unlabeled text.

        BERT objective
            word, sentence, positional embeddings
                positional embeddings
                segment embeddings
                token embeddings including: <SEP>, <EOS>
                [CLS] classification symbol
                    sum
                        embeddings
                        transformer blocks
                            predict masked word, cross entropy loss
                            + binary loss, next sentence prediction

Q/A architecture
    feed forward
        Serial [LayerNorm,
        dense,
        activation,
        dropout_mid,
        dense,
        dropout_end]

    encoder[
        Residual (LayerNorm,
                  attention,
                  dropout_,),
        Residual (feed_forward,)]

    Q:question C:context A:answer
'''

# SentencePiece and Byte Pair Encoding

# encode text as numbers with ids (such as the embedding vectors we've been using in the previous assignments), since the tensor operations act on numbers (text, words, morphemes, phonemes, characters)

# In SentencePiece unicode characters are grouped together using either a unigram language model (used in this week's assignment) or BPE, byte-pair encoding

eaccent = '\u00E9'
e_accent = '\u0065\u0301'
print(f'{eaccent} = {e_accent} : {eaccent == e_accent}')

from unicodedata import normalize

norm_eaccent = normalize('NFKC', '\u00E9')
norm_e_accent = normalize('NFKC', '\u0065\u0301')
print(f'{norm_eaccent} = {norm_e_accent} : {norm_eaccent == norm_e_accent}')

def get_hex_encoding(s):
    return ' '.join(hex(ord(c)) for c in s)

def print_string_and_encoding(s):
    print(f'{s} : {get_hex_encoding(s)}')

for s in [eaccent, e_accent, norm_eaccent, norm_e_accent]:
    print_string_and_encoding(s)

s = 'Tokenization is hard.'
s_ = s.replace(' ', '\u2581')
s_n = normalize('NFKC', 'Tokenization is hard.')

print(get_hex_encoding(s))
print(get_hex_encoding(s_))
print(get_hex_encoding(s_n))

s = 'Tokenization is hard.'
sn = normalize('NFKC', 'Tokenization is hard.')
sn_ = s.replace(' ', '\u2581')

print(get_hex_encoding(s))
print(get_hex_encoding(sn))
print(get_hex_encoding(sn_))

# BPE Algorithm

import ast

def convert_json_examples_to_text(filepath):
    example_jsons = list(map(ast.literal_eval, open(filepath))) # Read in the json from the example file
    texts = [example_json['text'].decode('utf-8') for example_json in example_jsons] # Decode the byte sequences
    text = '\n\n'.join(texts)       # Separate different articles by two newlines
    text = normalize('NFKC', text)  # Normalize the text

    with open('example.txt', 'w') as fw:
        fw.write(text)

    return text

text = convert_json_examples_to_text('data.txt')
print(text[:900])

from collections import Counter

vocab = Counter(['\u2581' + word for word in text.split()])
vocab = {' '.join([l for l in word]): freq for word, freq in vocab.items()}

def show_vocab(vocab, end='\n', limit=20):
    shown = 0
    for word, freq in vocab.items():
        print(f'{word}: {freq}', end=end)
        shown +=1
        if shown > limit:
            break

show_vocab(vocab)

'''
▁ B e g i n n e r s: 1
▁ B B Q: 3
▁ C l a s s: 2
▁ T a k i n g: 1
▁ P l a c e: 1
▁ i n: 15
▁ M i s s o u l a !: 1
▁ D o: 1
▁ y o u: 13
▁ w a n t: 1
▁ t o: 33
▁ g e t: 2
▁ b e t t e r: 2
▁ a t: 1
▁ m a k i n g: 2
▁ d e l i c i o u s: 1
▁ B B Q ?: 1
▁ Y o u: 1
▁ w i l l: 6
▁ h a v e: 4
▁ t h e: 31
'''

print(f'Total number of unique words: {len(vocab)}')
print(f'Number of merges required to reproduce SentencePiece training on the whole corpus: {int(0.60*len(vocab))}')

import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_sentence_piece_vocab(vocab, frac_merges=0.60):
    sp_vocab = vocab.copy()
    num_merges = int(len(sp_vocab)*frac_merges)

    for i in range(num_merges):
        pairs = get_stats(sp_vocab)
        best = max(pairs, key=pairs.get)
        sp_vocab = merge_vocab(best, sp_vocab)

    return sp_vocab

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file='sentencepiece.model')

s0 = 'Beginners BBQ Class Taking Place in Missoula!'

# encode: text => id
print(sp.encode_as_pieces(s0))
print(sp.encode_as_ids(s0))

# decode: id => text
print(sp.decode_pieces(sp.encode_as_pieces(s0)))
print(sp.decode_ids([12847, 277]))

['▁Beginn', 'ers', '▁BBQ', '▁Class', '▁', 'Taking', '▁Place', '▁in', '▁Miss', 'oul', 'a', '!']
[12847, 277, 15068, 4501, 3, 12297, 3399, 16, 5964, 7115, 9, 55]
Beginners BBQ Class Taking Place in Missoula!
Beginners

uid = 15068
spiece = "\u2581BBQ"
unknown = "__MUST_BE_UNKNOWN__"

# id <=> piece conversion
print(f'SentencePiece for ID {uid}: {sp.id_to_piece(uid)}')
print(f'ID for Sentence Piece {spiece}: {sp.piece_to_id(spiece)}')

# returns 0 for unknown tokens (we can change the id for UNK)
print(f'ID for unknown text {unknown}: {sp.piece_to_id(unknown)}')

'''
SentencePiece for ID 15068: ▁BBQ
ID for Sentence Piece ▁BBQ: 15068
ID for unknown text __MUST_BE_UNKNOWN__: 2
'''

print(f'Beginning of sentence id: {sp.bos_id()}')
print(f'Pad id: {sp.pad_id()}')
print(f'End of sentence id: {sp.eos_id()}')
print(f'Unknown id: {sp.unk_id()}')
print(f'Vocab size: {sp.vocab_size()}')

print('\nId\tSentP\tControl?')
print('------------------------')
# <unk>, <s>, </s> are defined by default. Their ids are (0, 1, 2)
# <s> and </s> are defined as 'control' symbol.
for uid in range(10):
    print(uid, sp.id_to_piece(uid), sp.is_control(uid), sep='\t')

# for uid in range(sp.vocab_size()-10,sp.vocab_size()):
#     print(uid, sp.id_to_piece(uid), sp.is_control(uid), sep='\t')

'''
Id	SentP	Control?
------------------------
0	<pad>	True
1	</s>	True
2	<unk>	False
3	▁	False
4	X	False
5	.	False
6	,	False
7	s	False
8	▁the	False
9	a	False
'''

spm.SentencePieceTrainer.train('--input=example.txt --model_prefix=example_bpe --vocab_size=450 --model_type=bpe')
sp_bpe = spm.SentencePieceProcessor()
sp_bpe.load('example_bpe.model')

print('*** BPE ***')
print(sp_bpe.encode_as_pieces(s0))

show_vocab(sp_vocab, end = ', ')

from heapq import heappush, heappop

def heapsort(iterable):
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

a = [1,4,3,1,3,2,1,4,2]

'''
[1, 1, 1, 2, 2, 3, 3, 4, 4]
'''

# sentence piece: https://github.com/google/sentencepiece/blob/master/python/sentencepiece_python_module_example.ipynb
