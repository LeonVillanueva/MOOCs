# knn for machine translation, and document search
# locality sensitive hashing

'''
Machine Translation
	two languages vector spaces X, Y
	Loss = || XR - Y ||_f (calculated norm, frobenius norm = ||A||_f = np.sqrt ( SUM Aij **2 )
		f(A) = np.sqrt (np.sum (np.square (A)))
	
	g = d Loss / d R
		= 2/m * X.T (XR - Y)
	
	R = R - alpha * g
	

angle = 100 * (np.pi / 180)

Ro = np.array([[np.cos(angle), -np.sin(angle)],
              [np.sin(angle), np.cos(angle)]])

np.sqrt(np.sum(Ro * Ro)) == np.linalg.norm(Ro)
	
hash_function (vector) = hash_value
	locality sensitive hashing
	
	hash = 2^0 * h1 + 2^1 * h2 + 2^2 * h3 + ... 2^n * hn+1
	h (0, 1) : (False, True) : (Negative, Postive)

Random planes
	

'''

def side_of_plane_matrix(P, v):
    dotproduct = np.dot(P, v.T)
    sign_of_dot_product = np.sign(dotproduct) # Get a boolean value telling if the value in the cell is positive or negative
    return sign_of_dot_product

def hash_multi_plane(P_l, v):
    hash_value = 0
    for i, P in enumerate(P_l):
        sign = side_of_plane(P,v)
        hash_i = 1 if sign >=0 else 0
        hash_value += 2**i * hash_i
    return hash_value


'''

    English embeddings from Google code archive word2vec look for GoogleNews-vectors-negative300.bin.gz
        You'll need to unzip the file first.
    and the French embeddings from cross_lingual_text_classification.
        in the terminal, type (in one line) curl -o ./wiki.multi.fr.vec https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.fr.vec

'''

# Use this code to download and process the full dataset on your local computer

from gensim.models import KeyedVectors

en_embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
fr_embeddings = KeyedVectors.load_word2vec_format('./wiki.multi.fr.vec')


# loading the english to french dictionaries
en_fr_train = get_dict('en-fr.train.txt')
print('The length of the english to french training dictionary is', len(en_fr_train))
en_fr_test = get_dict('en-fr.test.txt')
print('The length of the english to french test dictionary is', len(en_fr_train))

english_set = set(en_embeddings.vocab)
french_set = set(fr_embeddings.vocab)
en_embeddings_subset = {}
fr_embeddings_subset = {}
french_words = set(en_fr_train.values())

for en_word in en_fr_train.keys():
    fr_word = en_fr_train[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


for en_word in en_fr_test.keys():
    fr_word = en_fr_test[en_word]
    if fr_word in french_set and en_word in english_set:
        en_embeddings_subset[en_word] = en_embeddings[en_word]
        fr_embeddings_subset[fr_word] = fr_embeddings[fr_word]


pickle.dump( en_embeddings_subset, open( "en_embeddings.p", "wb" ) )
pickle.dump( fr_embeddings_subset, open( "fr_embeddings.p", "wb" ) )