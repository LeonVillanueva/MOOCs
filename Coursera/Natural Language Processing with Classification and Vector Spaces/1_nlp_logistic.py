#supervised ML training
#vector
#sentiment analysis

#freqs
# Xm = [1, SUM_positive(w,1), SUM_negative(w,1)]

def sigmoid(z): 
    h = 1 / (1 + np.exp (-z))
    return h
	
def gradientDescent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for i in range(0, num_iters):
        z = np.dot (x, theta)
        h = sigmoid (z)
        J = (-1/m) * ( np.dot (np.transpose(y), np.log(h)) + np.dot (np.transpose((1-y)), np.log((1-h))) )
        theta = theta - (alpha/m) * (np.dot (np.transpose(x), (h-y)))
    J = float(J)
    return J, theta

for word in word_l:
        if (word, 1.0) in freqs:
            x[0,1] += freqs[(word, 1.0)]
        if (word, 0.0) in freqs:
            x[0,2] += freqs[(word, 0.0)]
			
def predict_tweet(tweet, freqs, theta):
    x = extract_features (tweet, freqs)
    y_pred = sigmoid (np.dot (x, theta))
    return y_pred