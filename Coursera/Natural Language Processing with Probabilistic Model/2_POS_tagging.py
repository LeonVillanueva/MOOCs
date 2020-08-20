# part of speech

'''
lexical term in the language
    names and named entities
    co-reference
    defintion (speech recognition)

# markov chains

part of speech dependencies
stochastic model for decribing previous to post events
directed graph Q = {q1, q2, q3, ...qN}
    states of the model (q)
    * initial state

graphs of states and transitions
markov probability : each state is dependent only the prior event

part of speech = state
probability of staying in the same state or moving to a different state
    * markov table (known)

initial states > transition matrix (N+1, N)

hidden markov models
    not directly observable probability for transitions
    emission probabilities hidden to observable transition state probability
    * emission matrix

transition probabilities
    1. C(ti-1, ti)
    2. P(ti, ti-1) = C(ti-1, ti)+1-e10 / SUM C(ti-1, tj)

    epsilon = 1-e10 (smoothing, prevents zero
    populating the transition matrix
    [#,nn,vb,o]
    [<s>,1,0,2]
    [nn,0,0,6]
    [vb,0,0,0]
    [o,0,0,8]

    calculate using (2)

    first row
    [<s>,0.33,0.00,0.66]

populating the emission matrix
    number of p(occurence) = n/total
    = (C(t,w) + e) / C(t) + N + e

    part of speech tag to estimate the most likely next part of speech

VITERBI algorithm
    <st> to <o> to "I"
    transmition to emission
    0.3 * 0.5 = 0.15 total
    "I" to <v> to "love"
    0.5 * 0.5 = 0.25 total
    probability of ssequence "I love"
    0.15 * 0.25 = 0.0375

    1.initialization
        D matrix
        c = <st> * b_c_index(w_1)
          = a_1,i * b_c_index(w_1)
    2.forward pass
        c_i,j = max (c_k,j-1) * a_k,i * b_i_index(w_i)
        d_i,j = argmax (c_i,j) ... returns the k that maximizes the function argument
    3.backward pass
        retrieve the most likely path
        s = argmax c_i,k
        from word 1 to word k
        calculate the index of c_i,k

        get the highest probability of the last column, get the (t) state
        get the the highest argmax (D) from position (t) state, word (k)
        recurse until completed to word (0) == <st>, get the part of speech tag <t_i>

        implementation notes
            be careful with indices (0 v 1)
            use log probabilities instead to overcome very small (p)
'''
