# co-occurence > vector space models
# co-occurence, number of times two items (words) occur within a certain distance (k)

'''
Euclidean Distance = a = sqrt (b^2+c^2)
	expansion add new lengths : norm of the difference

Cosine Similarity
	cosine of the (inner) angle of two points, closer to 90 degrees (orthogonal), closer to zero
	useful for non equal corpora (population samples)
	acute angle (closer to 1), obtuse angle (closer to -1) 
		: similar meaning vs opposite meaning

	# v_vector . w_vector = ||v_vector||*||w_vector|| cos(B)
		cos(B) = v_vector . w_vector / (||v_vector|| * ||w_vector||)

		where || || the norms

Principal Component Analysis (PCA)
	dimensionality reduction using uncorrelated features
	
	Eigenvalues / Eigenvectors
	Retained Information / Uncorrelated features
		eigenvalues are the variances of the new features

		mean normalize > get covariance matrix > singular value decomposition
		
		eigenvalues . eigenvectors = new principal components

'''

np.dot(a, b)  / (np.linalg.norm(a) * np.linalg.norm(b))

#

import nltk
from gensim.models import KeyedVectors


embeddings = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary = True)
f = open('capitals.txt', 'r').read()
set_words = set(nltk.word_tokenize(f))
select_words = words = ['king', 'queen', 'oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']
for w in select_words:
    set_words.add(w)

def get_word_embeddings(embeddings):

    word_embeddings = {}
    for word in embeddings.vocab:
        if word in set_words:
            word_embeddings[word] = embeddings[word]
    return word_embeddings


# Testing your function
word_embeddings = get_word_embeddings(embeddings)
print(len(word_embeddings))
pickle.dump( word_embeddings, open( "word_embeddings_subset.p", "wb" ) )

# PCA

def compute_pca(X, n_components=2):
    X_demeaned = X - np.mean(X, axis=0)
    covariance_matrix = np.cov (X_demeaned, rowvar=False)
    eigen_vals, eigen_vecs = np.linalg.eigh (covariance_matrix) 
    idx_sorted = np.argsort (eigen_vals)
    idx_sorted_decreasing = idx_sorted [::-1]
    eigen_vals_sorted = eigen_vals [idx_sorted_decreasing]
    eigen_vecs_sorted = eigen_vecs [:,idx_sorted_decreasing]
    eigen_vecs_subset = eigen_vecs_sorted [:,:n_components]
    X_reduced = np.dot(eigen_vecs_subset.T, X_demeaned.T).T
    return X_reduced


'''
rowvarbool, optional

    If rowvar is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.


rowvar = True
[[ 0.03232399 -0.00964737 -0.02267662]
 [-0.00964737  0.04757433 -0.03792696]
 [-0.02267662 -0.03792696  0.06060358]]

rowvar = False
[[ 4.88046950e-02  3.38429177e-02  2.70410354e-02  1.33348089e-02
   1.00609001e-01  6.57707762e-02 -2.75184938e-02 -5.25695002e-02
  -1.27339493e-02  6.48227388e-02]
 [ 3.38429177e-02  2.38029956e-02  1.68919133e-02  3.98205302e-03
   7.08994547e-02  4.03429340e-02 -2.12082910e-02 -3.84257778e-02
  -6.48868839e-03  4.80954113e-02]
 [ 2.70410354e-02  1.68919133e-02  2.52986469e-02  3.65993232e-02
   4.94545211e-02  6.56528268e-02 -3.45131361e-03 -1.81844337e-02
  -2.00468908e-02  1.84664070e-02]
 [ 1.33348089e-02  3.98205302e-03  3.65993232e-02  8.63566931e-02
   9.67985927e-03  1.00685091e-01  2.58818405e-02  1.66212907e-02
  -4.02656116e-02 -3.16988513e-02]
 [ 1.00609001e-01  7.08994547e-02  4.94545211e-02  9.67985927e-03
   2.11236189e-01  1.17774282e-01 -6.39199532e-02 -1.15041459e-01
  -1.83299257e-02  1.44268308e-01]
 [ 6.57707762e-02  4.03429340e-02  6.56528268e-02  1.00685091e-01
   1.17774282e-01  1.71350909e-01 -3.68356893e-03 -3.98590657e-02
  -5.39476528e-02  3.79461186e-02]
 [-2.75184938e-02 -2.12082910e-02 -3.45131361e-03  2.58818405e-02
  -6.39199532e-02 -3.68356893e-03  2.90038970e-02  4.21533131e-02
  -7.67476391e-03 -5.65027401e-02]
 [-5.25695002e-02 -3.84257778e-02 -1.81844337e-02  1.66212907e-02
  -1.15041459e-01 -3.98590657e-02  4.21533131e-02  6.82317479e-02
  -6.40769360e-05 -8.83324737e-02]
 [-1.27339493e-02 -6.48868839e-03 -2.00468908e-02 -4.02656116e-02
  -1.83299257e-02 -5.39476528e-02 -7.67476391e-03 -6.40769360e-05
   1.96830541e-02  5.06165683e-03]
 [ 6.48227388e-02  4.80954113e-02  1.84664070e-02 -3.16988513e-02
   1.44268308e-01  3.79461186e-02 -5.65027401e-02 -8.83324737e-02
   5.06165683e-03  1.15614106e-01]]
'''