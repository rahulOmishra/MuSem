

def strip_eos(sents):
    return [sent[:sent.index('<eos>')] if '<eos>' in sent else sent
        for sent in sents]

def feed_dictionary(model, batch, rho, gamma, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.rho: rho,
                 model.gamma: gamma,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights'],
                 model.labels: batch['labels']}
    return feed_dict

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def reorder(order, _x):
    x = [0 for i in range(len(_x))]
    for i, a in zip(order, _x):
        x[i] = a
    return x


def get_batch(d, x, y, word2id, min_len=5):
    # d is the paragraph and x is the headline
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']


    rev_x, go_x, x_eos, weights = [], [], [], []
    max_len = max([len(sent) for sent in x])
    max_len = max(max_len, min_len)
    for sent in x:
        sent_id = [word2id[w] if w in word2id else unk for w in sent]
        l = len(sent)
        padding = [pad] * (max_len - l)
        rev_x.append(padding + sent_id[::-1])
        go_x.append([go] + sent_id + padding)
        x_eos.append(sent_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len-l))

    rev_d, go_d, d_eos, weights_rec = [], [], [], []
    max_len_d = max([len(sent_d) for sent_d in d])
    max_len_d = max(max_len_d, min_len)
    for send_d in d:
        send_d_id = [word2id[w] if w in word2id else unk for w in send_d]
        l = len(send_d)
        padding = [pad] * (max_len_d - l)
        rev_d.append(padding + send_d_id[::-1])
        go_d.append([go] + send_d_id + padding)
        d_eos.append(send_d_id + [eos] + padding)
        weights_rec.append([1.0] * (l+1) + [0.0] * (max_len_d-l))


    return {'enc_inputs': rev_d, # paragraph as encoder input
            'dec_inputs': go_x, # headline as the decoder input
            'targets':    x_eos, # headline with filling elements as the targets
            'weights':    weights, # weight for each words,the "filling elements" should be weighted as 0
            'labels':     y,# labels of headlines
            'size':       len(x), #length of headline
            'len':        max_len+1} # max length of all headlines

def get_batches(d0,d1,x0, x1, word2id, batch_size):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
        d0 = makeup(d0, len(d1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
        d1 = makeup(d1, len(d0))
    n = len(x0)

    order0 = range(n)
    z = sorted(zip(order0, x0,d0), key=lambda i: len(i[1]))
    order0, x0,d0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1,d1), key=lambda i: len(i[1]))
    order1, x1,d1= zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(d0[s:t]+d1[s:t], x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id))
        s = t

    return batches, order0, order1


