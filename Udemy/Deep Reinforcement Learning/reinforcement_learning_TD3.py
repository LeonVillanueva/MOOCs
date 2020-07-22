'''
https://arxiv.org/pdf/1802.09477.pdf
agent > action > environment > state, reward > agent

accumulated reward = expected return
policy gradient
agent == policy

q-learning
q-value : state / action

	1. initialize, q-values = 0
	at t >= 1
	2. random state (s)
	3. (s)t+1, reward R(s)t(a)t : we get the reward at from prior state and action
	4. temporal difference
		TD(s,a) = R(s,a) + LAMBDA * MAX (Q(s)t+1,a) - (Q(s),a)
	5. update Q value
		(Q(s),a) += ALPHA * TD(s,a)

deep q-learning

	only on discrete cases
	
	M = memory, initialize empty list
	
	1. initialize
	2. current state (s)
	3. a = argmax{Q(s,a}, play the action with the highest Q
	4. R{s,a}
	5. (s)t+1
	6. (s,a,rt,st+1) append the transition to memory (M)
			Q(s,a), predictions
			R(s,a) + LAMBDA * MAX (Q(s)t+1,a) - (Q(s),a)
			LOSS : 1/2 SUM [R(s,a) + LAMBDA * MAX (Q(s)t+1,a) - (Q(s),a)]^2 = 1/2 SUM TD (s,a)^2
			
			Back propogate
			
	
'''