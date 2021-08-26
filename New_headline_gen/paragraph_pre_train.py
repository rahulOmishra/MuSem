import os
import sys
import time
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf
from losses_pre import Losses
import argparse
import math
import pprint

from vocab import Vocabulary, build_vocab
from file_io import load_sent,load_desc
from nn import *

class Model(object):

    def __init__(self, args, vocab):
        dim_z = args.dim_z
        n_layers = args.n_layers
        beta1, beta2 = 0.5, 0.999

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.gamma = tf.placeholder(tf.float32,
            name='gamma')

        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.enc_inputs = tf.placeholder(tf.int32, [None, None],    #size * len
            name='enc_inputs')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')

        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_z, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        enc_inputs = tf.nn.embedding_lookup(embedding, self.enc_inputs)
        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####
        init_state = tf.zeros([self.batch_size, dim_z])

        cell_e = create_cell(dim_z, n_layers, self.dropout)
        _, z = tf.nn.dynamic_rnn(cell_e, enc_inputs,
            initial_state=init_state, scope='encoder')
        self.z = z

        cell_g = create_cell(dim_z, n_layers, self.dropout)
        g_outputs, _ = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=z, scope='generator')

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, dim_z])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b

        loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        loss_g *= tf.reshape(self.weights, [-1])
        self.loss_g = tf.reduce_sum(loss_g) / tf.to_float(self.batch_size)

        #####   optimizer   #####
        theta_eg = retrive_var(['encoder', 'generator',
            'embedding', 'projection'])

        self.optimizer_ae = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_g, var_list=theta_eg)

        self.saver = tf.train.Saver()

def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model


def get_batch(d,word2id, min_len=5):
    # d is the paragraph
    pad = word2id['<pad>']
    go = word2id['<go>']
    eos = word2id['<eos>']
    unk = word2id['<unk>']

    ori_d,rev_d, go_d, d_eos, weights = [],[], [], [], []
    max_len_d = max([len(sent_d) for sent_d in d])
    max_len_d = max(max_len_d, min_len)
    for send_d in d:
        send_d_id = [word2id[w] if w in word2id else unk for w in send_d]
        l = len(send_d)
        padding = [pad] * (max_len_d - l)
        rev_d.append(padding + send_d_id[::-1])
        ori_d.append(padding + send_d_id)
        go_d.append([go] + send_d_id + padding)
        d_eos.append(send_d_id + [eos] + padding)
        weights.append([1.0] * (l+1) + [0.0] * (max_len_d-l))

    ## 'enc_inputs': rev_d,  # reversed paragraph as encoder input
    return {
            'enc_inputs': ori_d, # original paragraph as encoder input
            'dec_inputs': go_d, # paragraph as encoder input
            'targets':    d_eos, # headline with filling elements as the targets
            'weights':    weights, # weight for each words,the "filling elements" should be weighted as 0
            'size':       len(d), #length of paragraph
            'len':        max_len_d+1} # max length of all paragraphs

def get_batches(d, word2id, batch_size):

    order = range(len(d))
    z = sorted(zip(order,d), key=lambda i: len(i[1]))
    order, d= zip(*z)

    batches = []
    n = len(d)
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(d[s:t],word2id))
        s = t

    return batches, order

def get_batches_no_shuffle(d, word2id, batch_size):
    batches = []
    n = len(d)
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(d[s:t],word2id))
        s = t

    return batches

def feed_dictionary(model, batch, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.enc_inputs: batch['enc_inputs'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.weights: batch['weights']}
    return feed_dict

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument('--train_desc',
            type = str,
            default='../data2/all_descs')
    argparser.add_argument('--all_desc',
            type=str,
            default='../data2/all_descs')
    argparser.add_argument('--test_desc',
            type=str,
            default='../data2/test_descs')
    argparser.add_argument('--output',
            type=str,
            default='')
    argparser.add_argument('--vocab',
            type=str,
            default='') #clickbait_challenge.vocab
    argparser.add_argument('--model',
            type=str,
            default='../pre_tmp2/model')
    argparser.add_argument('--embedding',
            type=str,
            default='')#../data/glove.6B.100d.txt
    argparser.add_argument('--z_path',
            type=str,
            default='')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--pretrain_model',
            type=str,
            default='')
    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--max_epochs',
            type=int,
            default=300)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=20)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)
    argparser.add_argument('--max_desc_len',
            type=int,
            default=100)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=0.5)
    argparser.add_argument('--dim_z',
            type=int,
            default=64) #64
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.005)
    argparser.add_argument('--gamma_init',              # softmax(logit / gamma)
            type=float,
            default=1)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=0.5)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.001)
    argparser.add_argument('--mini_occur',
            type=int,
            default=3)

    args = argparser.parse_args()

    print ('------------------------------------------------')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print ('------------------------------------------------')

    return args

def reorder(order, _x):
    x = [0 for i in range(len(_x))]
    for i, a in zip(order, _x):
        x[i] = a
    return x

def getAllZ():

    args = load_arguments()
    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ('vocabulary size:', vocab.size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model = Model(args, vocab)
        model.saver.restore(sess, args.model)
        if args.all_desc:
            all_desc0 = load_desc(args.all_desc+'.0',args.max_desc_len)
            all_desc1 = load_desc(args.all_desc+'.1',args.max_desc_len)

        learning_rate = args.learning_rate
        gamma = args.gamma_init
        dropout = args.dropout_keep_prob

        f0 = open(args.z_path + 'all_head_z.0', 'w+')
        batches0 = get_batches_no_shuffle(all_desc0, vocab.word2id, args.batch_size)
        for batch in batches0:
            feed_dict = feed_dictionary(model, batch,
                                        dropout, learning_rate)
            loss_g, z, _ = sess.run([model.loss_g, model.z, model.optimizer_ae], feed_dict=feed_dict)

            for zz in z:
                f0.write('\t'.join([str(x) for x in zz]))
                f0.write('\n')
        f0.close()

        # 1
        f1 = open(args.z_path + 'all_head_z.1', 'w+')
        batches1 = get_batches_no_shuffle(all_desc1, vocab.word2id, args.batch_size)
        for batch in batches1:
            feed_dict = feed_dictionary(model, batch,
                                        dropout, learning_rate)
            loss_g, z, _ = sess.run([model.loss_g, model.z, model.optimizer_ae], feed_dict=feed_dict)
            for zz in z:
                f1.write('\t'.join([str(x) for x in zz]))
                f1.write('\n')
        f1.close()

def train():
    args = load_arguments()


    #####   data preparation   #####
    if args.train_desc:

        train_desc0 = load_desc(args.train_desc+'.0',args.max_desc_len)
        train_desc1 = load_desc(args.train_desc+'.1',args.max_desc_len)
        print ( '#descs of training file 0:', len(train_desc0))
        print ( '#descs of training file 1:', len(train_desc1))

        if not os.path.isfile(args.vocab):
            build_vocab(train_desc0 + train_desc1, args.vocab,min_occur=args.mini_occur)
        build_vocab(train_desc0 + train_desc1, args.vocab, min_occur=args.mini_occur)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ( 'vocabulary size:', vocab.size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)

        variables_names = [v.name for v in tf.trainable_variables()]
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print "Variable: ", k
            print "Shape: ", v.shape
            print v

        train_desc = []
        train_desc.extend(train_desc0)
        train_desc.extend(train_desc1)
        batches, order= get_batches(train_desc, vocab.word2id, args.batch_size)
        random.shuffle(batches)

        start_time = time.time()
        best_training = float('inf')
        losses = Losses(len(batches))
        learning_rate = args.learning_rate
        dropout = args.dropout_keep_prob

        for epoch in range(1, 1+args.max_epochs):
            all_z = []
            print ( '--------------------epoch %d--------------------' % epoch)
            print ( 'learning_rate:', learning_rate)

            for batch in batches:
                feed_dict = feed_dictionary(model, batch,
                    dropout, learning_rate)
                loss_g, z,_ = sess.run([model.loss_g, model.z,model.optimizer_ae],feed_dict=feed_dict)
                all_z.extend(z)

                losses.add(loss_g)

            losses.output('time %.0fs,'
                % (time.time() - start_time))

            if losses.g < best_training:
                best_training = losses.g

                print ('saving model...')
                model.saver.save(sess, args.model)

            losses.clear()


if __name__ == '__main__':
    train()
    # getAllZ()