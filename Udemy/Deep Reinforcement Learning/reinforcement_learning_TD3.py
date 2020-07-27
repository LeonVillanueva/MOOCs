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

'''
TD3 : Twin Delayed Deep Dertiministic Polocy Gradient

	solution to continuous space of actions
	
		q-values alone can't understand infinte (continuous) actions
		include actor critic
		
		learn both policy and q-values that are optimal
		
		twin : two critics
		
			actor (states >> policy)
			critic1 (states + action >> q-values)
			critic2 (states + action >> q-values)

	explore-exploit 
	
	transition (LAST STATE, NEXT STATE, ACTION, REWARD) > get the next action from next state
		(s, s', a, r)
	
	Q learning
	
	1. initialize M (amount of transitions stored)
	2. actor model neural network
	   actor target neural network
	3a. corresponding two critic models for (2)
	3b. random explore for making transitions
	4. sample transitions
	5. from transition get s'
	6. add gausian noise for next a, and clamp within a range
	7. two critic models (4 neural network)
		Qt1(s',a')
		Qt2(s',a')
	8. keep the minimum of (7), approximation of the next state s'
		prevent too optimistic estimate = stability
		min(Qt1, Qt2)
	9. Qt = r + GAMMA * min(Qt1, Qt2)
		GAMMA = discount factor
	10. Q1(s,a)
	    Q2(s,a)
		
		compare (9)
	11. critic loss = MSE_loss(Q1-Qt) + MSE_loss(Q2-Qt)
	12. back propogate critic loss (11) and update parameter weights

'''