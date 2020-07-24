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

deep q-learning (input > q-target, and backprop)

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

policy gradient (neural network states > action)

	PIo = policy, o = phi
	R = SUM LAMBDA^(i-t) r[s,a]
		^(i-t) : discount factor
	
	
	GOAL : Maxmize expedted return
	
	Compute the gradient of the policy return
		DELTA J(o)
		ot+1 += ot ALPHA * DELTA J(o)

actor-critic (states > actions, actions + state > q-values)

	2 models working at the same time
	
taxonomy

	model based, model free : simulated (predicted) environment vs data from environment itself
	value based, policy based : update using q-value (intermidiary), directly update weights of policy (actions taken)
	off-policy, on-policy : learn from past data (memory), learn only on new data 
	
	

	
	
'''