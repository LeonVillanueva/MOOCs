'''
longer task NPLs
larger transformer models
    reversible transformer
        writing books
        chatbots

chatbot models
    all prior contexts
    longer context windows

transformer complexity
    L -> L**2 time
    N layers, times N memory
    (QK**t)*V
    more layers = more memory = forward pass activations
    (re)compute vs memory tradeoff

LSH Attention
    locality sensitive hashing
        hashing (q) == hashing (k)
        hash (x) = sign (xR), R: [d, n_hash_bins]

        A (QKV) = softmax (QK**t)*V

        1. hash Q and K
        2. standard attention with hash bins
        3. repeat to increase probability (p) in the same bin

        QK attention = finding Q and K in the same bin

        output a single vector in each position which is both Query and Key

        a. Sequence of Query = Keys
        b. LSH bucketing
        c. sort by LSH bucketing
        d. chunk sorted sequnece to parallelize
        e. attend within the same bucket, of self chunk and prior chunk

    The LSH buckets are sorted and chunked. Then you let each chunk attend within itself and adjacent chunks.

    Probabilistic model

Reversible Layer
    memory efficiency
    Transformers = Attention layers and feed forward layers.

    each layer (typically) needs to remember each activation/residual
    OR reverse the residuals
    no caching for the backward pass

    You can run the network backwards by computing the activations at each step.

    STANDARD
        y_a = x + attention (x)
        y_b = y_a + FeedForward (y_a)
    Reversible
        y_1 = x1 + attention (x2)
        y_2 = x2 + FeedForward (y1)

        recompute:

        x1 = attention (x2) - y_1
        x2 = FeedForward (y1) - y_2

        Rotate (transpose?)

Reformer model
    reversible transformer
        Reformer is a transformer model designed to be memory efficient so it can handle very large context windows of upto 1 million words.
'''

# UNQ_C1
DIALOGUE_DB['SNG0073.json']['log'][0]
DIALOGUE_DB['SNG0073.json']['log'][i]

result += cur_log
