import tensorflow as tf
import numpy as np

def leaky_relu(x, alpha=0.01):
    return tf.maximum(alpha * x, x)

def create_cell(dim, n_layers, dropout):
    cell = tf.nn.rnn_cell.GRUCell(dim)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell,
        input_keep_prob=dropout)
    if n_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * n_layers)
    return cell

def retrive_var(scopes):
    var = []
    for scope in scopes:
        var += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)
    return var

def linear(inp, dim_out, scope, reuse=False):
    dim_in = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_in, dim_out])
        b = tf.get_variable('b', [dim_out])
    return tf.matmul(inp, W) + b


def combine(x, y, scope, reuse=False):
    dim_x = x.get_shape().as_list()[-1]
    dim_y = y.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W = tf.get_variable('W', [dim_x+dim_y, dim_x])
        b = tf.get_variable('b', [dim_x])

    h = tf.matmul(tf.concat([x, y], 1), W) + b
    return leaky_relu(h)

def feed_forward(inp, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        W1 = tf.get_variable('W1', [dim, dim])
        b1 = tf.get_variable('b1', [dim])
        W2 = tf.get_variable('W2', [dim, 1])
        b2 = tf.get_variable('b2', [1])
    h1 = leaky_relu(tf.matmul(inp, W1) + b1)
    logits = tf.matmul(h1, W2) + b2

    return tf.reshape(logits, [-1])

def gumbel_softmax(logits, gamma, eps=1e-20):
    U = tf.random_uniform(tf.shape(logits))
    G = -tf.log(-tf.log(U + eps) + eps)
    return tf.nn.softmax((logits + G) / gamma)

def softsample_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = gumbel_softmax(logits, gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def softmax_word(dropout, proj_W, proj_b, embedding, gamma):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        prob = tf.nn.softmax(logits / gamma)
        inp = tf.matmul(prob, embedding)
        return inp, logits

    return loop_func

def argmax_word(dropout, proj_W, proj_b, embedding):

    def loop_func(output):
        output = tf.nn.dropout(output, dropout)
        logits = tf.matmul(output, proj_W) + proj_b
        word = tf.argmax(logits, axis=1)
        inp = tf.nn.embedding_lookup(embedding, word)
        return inp, logits

    return loop_func

def rnn_decode(h, inp, length, cell, loop_func, scope):
    h_seq, logits_seq = [], []

    with tf.variable_scope(scope):
        tf.get_variable_scope().reuse_variables()
        for t in range(length):
            # h_seq.append(tf.expand_dims(h, 1))
            output, h = cell(inp, h)
            inp, logits = loop_func(output)
            logits_seq.append(tf.expand_dims(logits, 1))
            h_seq.append(tf.expand_dims(h, 1))

    return tf.concat(h_seq, 1), tf.concat(logits_seq, 1)

def cnn(inp, filter_sizes, n_filters, dropout, scope, reuse=False):
    dim = inp.get_shape().as_list()[-1]
    inp = tf.expand_dims(inp, -1)

    with tf.variable_scope(scope) as vs:
        if reuse:
            vs.reuse_variables()

        outputs = []
        for size in filter_sizes:
            with tf.variable_scope('conv-maxpool-%s' % size,reuse=tf.AUTO_REUSE):
                W = tf.get_variable('W', [size, dim, 1, n_filters])
                b = tf.get_variable('b', [n_filters])
                conv = tf.nn.conv2d(inp, W,
                    strides=[1, 1, 1, 1], padding='VALID')
                h = leaky_relu(conv + b)
                # max pooling over time
                pooled = tf.reduce_max(h, reduction_indices=1)
                pooled = tf.reshape(pooled, [-1, n_filters])
                outputs.append(pooled)
        outputs = tf.concat(outputs, 1)
        outputs = tf.nn.dropout(outputs, dropout)

        with tf.variable_scope('output',reuse=tf.AUTO_REUSE):
            W = tf.get_variable('W', [n_filters*len(filter_sizes), 1])
            b = tf.get_variable('b', [1])
            logits = tf.reshape(tf.matmul(outputs, W) + b, [-1])

    return logits

def discriminator_fake_real(x_real, x_fake,labels_onehot, scope):
    features = tf.concat([x_real,x_fake], 0)
    pred = softmax_classify(features,scope)
    loss = tf.reduce_mean(-tf.reduce_sum(labels_onehot * tf.log(pred), reduction_indices=1))
    return loss,pred


def discriminator_linear(feature_h,label_h,scope):

    pred = softmax_classify(feature_h,scope)
    loss_d = tf.reduce_mean(-tf.reduce_sum(label_h*tf.log(pred), reduction_indices=1))
    return loss_d,pred

def softmax_classify(inp, scope):
    dim = inp.get_shape().as_list()[-1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as vs:
        W = tf.get_variable('W', [dim, 2])
        b = tf.get_variable('b',[2])
        logits = tf.nn.softmax(tf.matmul(inp, W) + b)
    return logits

def discriminator_hz(h,z,flag,scope):
    dim_h = h.get_shape().as_list()[-1]
    dim_z = z.get_shape().as_list()[-1]
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as vs:
        W = tf.get_variable('W',[dim_h,dim_z])
        ttmp = tf.reduce_sum(tf.multiply(tf.matmul(h,W),z),reduction_indices=1)
        loss_hz = -tf.reduce_mean(tf.log_sigmoid(flag*ttmp))

    return loss_hz