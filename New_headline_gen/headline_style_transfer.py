import os
import sys
import time
import random
import cPickle as pickle
import numpy as np
import tensorflow as tf
import sys
import argparse
import pprint
from vocab import Vocabulary, build_vocab
from file_io import load_sent, write_sent,load_z
from nn import *
import beam_search, greedy_decoding
import random
from sklearn import preprocessing

class Model(object):

    def __init__(self, args, vocab):
        dim_y = args.dim_y
        dim_z = args.dim_z
        dim_h = dim_y + dim_z
        dim_emb = args.dim_emb
        n_layers = args.n_layers
        max_len = args.max_seq_length
        filter_sizes = [int(x) for x in args.filter_sizes.split(',')]
        n_filters = args.n_filters
        beta1, beta2 = 0.5, 0.999

        self.dropout = tf.placeholder(tf.float32,
            name='dropout')
        self.learning_rate = tf.placeholder(tf.float32,
            name='learning_rate')
        self.alpha = tf.placeholder(tf.float32,
            name='alpha')
        self.beta = tf.placeholder(tf.float32,
            name='beta')
        self.eta = tf.placeholder(tf.float32,
            name='eta')
        self.gamma = tf.placeholder(tf.float32,
            name='gamma')
        self.batch_len = tf.placeholder(tf.int32,
            name='batch_len')
        self.batch_size = tf.placeholder(tf.int32,
            name='batch_size')
        self.dec_inputs = tf.placeholder(tf.int32, [None, None],
            name='dec_inputs')
        self.targets = tf.placeholder(tf.int32, [None, None],
            name='targets')
        self.weights = tf.placeholder(tf.float32, [None, None],
            name='weights')
        self.labels = tf.placeholder(tf.float32, [None],
            name='labels')
        self.z = tf.placeholder(tf.float32,[None,dim_z],
            name='z')
        self.z_shuffle = tf.placeholder(tf.float32,[None,dim_z],
            name='z_shuffle')
        self.z_shuffle1 = tf.placeholder(tf.float32,[None,dim_z],
            name='z_shuffle1')
        self.z_shuffle2 = tf.placeholder(tf.float32,[None,dim_z],
            name='z_shuffle2')

        labels = tf.reshape(self.labels, [-1, 1])

        embedding = tf.get_variable('embedding',
            initializer=vocab.embedding.astype(np.float32))
        with tf.variable_scope('projection'):
            proj_W = tf.get_variable('W', [dim_h, vocab.size])
            proj_b = tf.get_variable('b', [vocab.size])

        dec_inputs = tf.nn.embedding_lookup(embedding, self.dec_inputs)

        #####   auto-encoder   #####

        self.h_ori = tf.concat([linear(labels, dim_y,
            scope='generator'), self.z], 1)

        self.h_tsf = tf.concat([linear(1-labels, dim_y,
            scope='generator', reuse=True), self.z], 1)
        cell_g = create_cell(dim_h, n_layers, self.dropout)
        g_outputs, self.states = tf.nn.dynamic_rnn(cell_g, dec_inputs,
            initial_state=self.h_ori, scope='generator')

        # Teacher forced reconstruction
        teach_h = g_outputs
        self.teach_h = teach_h

        g_outputs = tf.nn.dropout(g_outputs, self.dropout)
        g_outputs = tf.reshape(g_outputs, [-1, dim_h])
        g_logits = tf.matmul(g_outputs, proj_W) + proj_b

        loss_g = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=g_logits)
        loss_g *= tf.reshape(self.weights, [-1])
        self.loss_g = tf.reduce_sum(loss_g) / tf.to_float(self.batch_size)

        #####   feed-previous decoding   #####
        go = dec_inputs[:,0,:]
        soft_func = softsample_word(self.dropout, proj_W, proj_b, embedding,
            self.gamma)
        hard_func = argmax_word(self.dropout, proj_W, proj_b, embedding)


        soft_h_tsf, soft_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, soft_func, scope='generator')

        hard_h_ori, self.hard_logits_ori = rnn_decode(self.h_ori, go, max_len,
            cell_g, hard_func, scope='generator')

        hard_h_tsf, self.hard_logits_tsf = rnn_decode(self.h_tsf, go, max_len,
            cell_g, hard_func, scope='generator')

        #####   discriminator for predicting clickbait/non-clickbait #####

        half = self.batch_size / 2
        soft_h_tsf = soft_h_tsf[:, :self.batch_len, :]

        weight_rev = tf.reverse(self.weights,[1])
        self.last_ind_rev = tf.reshape(tf.argmax(weight_rev,1),[-1,1])
        self.ind_size = tf.reshape(tf.tile([self.batch_len-1],[self.batch_size]),[-1,1])
        self.last_ind = tf.subtract(tf.cast(self.ind_size,tf.int64),self.last_ind_rev)
        self.last_ind_onehot = tf.reshape(tf.one_hot(self.last_ind,depth=self.batch_len),[self.batch_size,self.batch_len])
        self.last_ind_onehot_ext = tf.stack([self.last_ind_onehot]*dim_h,axis=2)
        self.last_ind_bool = tf.cast(self.last_ind_onehot_ext,tf.bool)
        last_teach_h = tf.reshape(tf.boolean_mask(teach_h,self.last_ind_bool),[self.batch_size,dim_h])
        self.last_teach_h = last_teach_h

        last_soft_h_tsf = soft_h_tsf[:,-1,:]
        self.last_soft_h_tsf = last_soft_h_tsf
        self.features_h = tf.concat([last_teach_h,last_soft_h_tsf],0)

        # Labels blocks are [0][1][1][0]
        self.labels_h = tf.concat([tf.zeros([half, 1], tf.int32),tf.ones([half, 1], tf.int32),
                                   tf.ones([half, 1], tf.int32),tf.zeros([half, 1], tf.int32)],0)

        self.labels_h_onehot = tf.reshape(tf.one_hot(self.labels_h,depth=2),[-1,2])

        self.loss_d, self.pred= discriminator_linear(self.features_h, self.labels_h_onehot,scope='discriminator')

        ##### discriminator for predicting (headline,paragraph) pair, negative sampling loss#####

        self.loss_d_hz_pos = discriminator_hz(last_teach_h,self.z,1.0,scope='discriminator_hz')
        self.loss_d_hz_pos_gen = discriminator_hz(last_soft_h_tsf, self.z, 1.0,scope='discriminator_hz')

        self.loss_d_hz_neg = discriminator_hz(last_teach_h,self.z_shuffle,-1.0,scope='discriminator_hz')+\
                             discriminator_hz(last_teach_h,self.z_shuffle1,-1.0,scope='discriminator_hz')+\
                             discriminator_hz(last_teach_h,self.z_shuffle2,-1.0,scope='discriminator_hz')
        self.loss_d_hz_neg_gen = discriminator_hz(last_soft_h_tsf,self.z_shuffle,-1.0,scope='discriminator_hz')+\
                             discriminator_hz(last_soft_h_tsf,self.z_shuffle1,-1.0,scope='discriminator_hz')+\
                             discriminator_hz(last_soft_h_tsf,self.z_shuffle2,-1.0,scope='discriminator_hz')

        self.loss_d_hz = self.loss_d_hz_pos+self.loss_d_hz_pos_gen+self.loss_d_hz_neg+self.loss_d_hz_neg_gen

        ##### discriminator for adversarial training#####
        self.labels_fr = tf.concat([tf.ones([half, 1], tf.int32),tf.zeros([half, 1], tf.int32)],0)
        self.labels_fr_onehot = tf.reshape(tf.one_hot(self.labels_fr,depth=2),[-1,2])
        # real non-clickbait and generated non-clickbait loss
        self.loss_d0_t,self.d_fr = discriminator_fake_real(last_teach_h[:half], last_soft_h_tsf[half:],self.labels_fr_onehot,scope='discriminator_t0')

        # real clickbait and generated clickbait loss
        self.loss_d1_t,_ = discriminator_fake_real(last_teach_h[half:], last_soft_h_tsf[:half],self.labels_fr_onehot,scope='discriminator_t1')

        self.loss_d_t = self.loss_d0_t+self.loss_d1_t

        #####   optimizer   #####
        self.loss = self.loss_g + self.alpha * self.loss_d + self.beta * self.loss_d_hz - self.eta* self.loss_d_t

        theta_eg = retrive_var(['generator',
            'embedding', 'projection'])
        theta_d = retrive_var(['discriminator'])
        theta_d_hz = retrive_var(['discriminator_hz'])
        theta_d0_t = retrive_var(['discriminator_t0'])
        theta_d1_t = retrive_var(['discriminator_t1'])
        theta_all = retrive_var(['generator',
            'embedding', 'projection'])

        self.optimizer_all = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss, var_list=theta_all)
        self.optimizer_ae = tf.train.AdamOptimizer(self.learning_rate,
            beta1, beta2).minimize(self.loss_g, var_list=theta_eg)

        self.optimizer_d = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1, beta2).minimize(self.loss_d, var_list=theta_d)
        self.optimizer_d_hz = tf.train.AdamOptimizer(self.learning_rate,
                                                     beta1, beta2).minimize(self.loss_d_hz, var_list=theta_d_hz)
        self.optimizer_d0_t = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1, beta2).minimize(self.loss_d0_t, var_list=theta_d0_t)
        self.optimizer_d1_t = tf.train.AdamOptimizer(self.learning_rate,
                                                  beta1, beta2).minimize(self.loss_d1_t, var_list=theta_d1_t)
        self.saver = tf.train.Saver()

def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])
    dir = 'tmp2_adv_1_1_1_0.005_16_64'

    argparser.add_argument('--train',
            type=str,
            default='')

    argparser.add_argument('--train_z',
            type = str,
            default='')
    argparser.add_argument('--dev',
            type=str,
            default='')
    argparser.add_argument('--dev_z',
            type=str,
            default='')
    argparser.add_argument('--test',
            type=str,
            default='')
    argparser.add_argument('--test_z',
            type=str,
            default='')
    argparser.add_argument('--output',
            type=str,
            default='../'+dir+'/clickbait_challenge.dev')
    argparser.add_argument('--vocab',
            type=str,
            default='../'+dir+'/clickbait_challenge.vocab')
    argparser.add_argument('--embedding',
            type=str,
            default='')
    argparser.add_argument('--info_reg_coeff',
            type=float,
            default=1)
    argparser.add_argument('--model',
            type=str,
            default='../'+dir+'/model')
    argparser.add_argument('--load_model',
            type=bool,
            default=False)
    argparser.add_argument('--pretrain_model',
            type=str,
            default='../'+dir+'/pre_model')
    argparser.add_argument('--batch_size',
            type=int,
            default=64)
    argparser.add_argument('--max_epochs',
            type=int,
            default=300)
    argparser.add_argument('--steps_per_checkpoint',
            type=int,
            default=20)
    argparser.add_argument('--max_seq_length',
            type=int,
            default=25)
    argparser.add_argument('--max_train_size',
            type=int,
            default=-1)
    argparser.add_argument('--beam',
            type=int,
            default=1)
    argparser.add_argument('--dropout_keep_prob',
            type=float,
            default=1) #0.5
    argparser.add_argument('--n_layers',
            type=int,
            default=1)
    argparser.add_argument('--dim_y',
            type=int,
            default=16) #16
    argparser.add_argument('--dim_z',
            type=int,
            default=64) #64
    argparser.add_argument('--dim_emb',
            type=int,
            default=100)
    argparser.add_argument('--learning_rate',
            type=float,
            default=0.005)
    argparser.add_argument('--alpha',
            type=float,
            default=1)
    argparser.add_argument('--beta',
            type=float,
            default=1)
    argparser.add_argument('--eta',
            type=float,
            default=1)
    argparser.add_argument('--gamma_init',              # softmax(logit / gamma)
            type=float,
            default=3)
    argparser.add_argument('--gamma_decay',
            type=float,
            default=0.5)
    argparser.add_argument('--gamma_min',
            type=float,
            default=0.005)
    argparser.add_argument('--filter_sizes',
            type=str,
            default='3,4,5')
    argparser.add_argument('--n_filters',
            type=int,
            default=128)
    argparser.add_argument('--mini_occur',
            type=int,
            default=1)

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

def transfer(model, decoder, sess, args, vocab, data_z0, data_z1, data0, data1, out_path):
    batches, order0, order1 = get_batches(data_z0, data_z1, data0, data1,
                                          vocab.word2id, args.batch_size)

    data0_tsf, data1_tsf = [], []
    losses = Losses(len(batches))
    for batch in batches:
        ori, tsf = decoder.rewrite(batch)
        half = int(batch['size'] / 2)
        data0_tsf += tsf[:half]
        data1_tsf += tsf[half:]

        loss, loss_g, loss_d, loss_d_hz = sess.run([model.loss,model.loss_g, model.loss_d, model.loss_d_hz],
                                                              feed_dict=feed_dictionary(model, batch, args.alpha,args.beta,
                                                                                        args.gamma_min,args.eta))
        losses.add(loss, loss_g,loss_d,loss_d_hz,loss_d_t)

    n0, n1 = len(data0), len(data1)
    reorder_tmp0 = reorder(order0, data0_tsf)
    data0_tsf = reorder_tmp0[:n0]
    reorder_tmp1 = reorder(order1, data1_tsf)
    data1_tsf = reorder_tmp1[:n1]

    if out_path:
        write_sent(data0_tsf, out_path + '.0' + '.tsf')
        write_sent(data1_tsf, out_path + '.1' + '.tsf')

    return losses

def feed_dictionary(model, batch, alpha, beta, gamma,eta, dropout=1, learning_rate=None):
    feed_dict = {model.dropout: dropout,
                 model.learning_rate: learning_rate,
                 model.gamma: gamma,
                 model.alpha:alpha,
                 model.beta: beta,
                 model.eta:eta,
                 model.z:batch['z'],
                 model.z_shuffle:batch['z_shuffle'],
                 model.z_shuffle1: batch['z_shuffle1'],
                 model.z_shuffle2: batch['z_shuffle2'],
                 model.batch_len: batch['len'],
                 model.batch_size: batch['size'],
                 model.dec_inputs: batch['dec_inputs'],
                 model.targets: batch['targets'],
                 model.labels: batch['labels'],
                 model.weights: batch['weights']}
    return feed_dict


def create_model(sess, args, vocab):
    model = Model(args, vocab)
    if args.load_model:
        print 'Loading model from', args.model
        model.saver.restore(sess, args.model)
    else:
        print 'Creating model with fresh parameters.'
        sess.run(tf.global_variables_initializer())
    return model

def makeup(_x, n):
    x = []
    for i in range(n):
        x.append(_x[i % len(_x)])
    return x

def z_shuffling(tmp_z):
    z_shuffle = []
    for i in range(len(tmp_z)):
        idx = np.random.randint(len(tmp_z))
        while idx==i:
            idx = np.random.randint(len(tmp_z))
        z_shuffle.append(tmp_z[idx])
    return z_shuffle

def get_batch(z, x, y, word2id, min_len=5):
    tmp_z = list(z)
    z_shuffle = z_shuffling(tmp_z)
    z_shuffle1 = z_shuffling(tmp_z)
    z_shuffle2 = z_shuffling(tmp_z)
    # z is the paragraph representation and x is the headline
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


    return {'z':z,
            'z_shuffle':z_shuffle,
            'z_shuffle1':z_shuffle1,
            'z_shuffle2': z_shuffle2,
            'dec_inputs': go_x, # headline as the decoder input
            'targets':    x_eos, # headline with filling elements as the targets
            'weights':    weights, # weight for each words,the "filling elements" should be weighted as 0
            'labels':     y,# labels of headlines
            'size':       len(x), #length of headline
            'len':        max_len+1} # max length of all headlines

def get_batches(z0,z1,x0, x1, word2id, batch_size):
    if len(x0) < len(x1):
        x0 = makeup(x0, len(x1))
        z0 = makeup(z0, len(z1))
    if len(x1) < len(x0):
        x1 = makeup(x1, len(x0))
        z1 = makeup(z1, len(z0))
    n = len(x0)

    order0 = range(n)
    z = sorted(zip(order0, x0,z0), key=lambda i: len(i[1]))
    order0, x0,z0 = zip(*z)

    order1 = range(n)
    z = sorted(zip(order1, x1,z1), key=lambda i: len(i[1]))
    order1, x1,z1= zip(*z)

    batches = []
    s = 0
    while s < n:
        t = min(s + batch_size, n)
        batches.append(get_batch(z0[s:t]+z1[s:t], x0[s:t] + x1[s:t],
            [0]*(t-s) + [1]*(t-s), word2id))
        s = t

    return batches, order0, order1

class Losses(object):
    def __init__(self, div):
        self.div = div
        self.clear()

    def clear(self):
        self.loss, self.g, self.d, self.d_hz , self.d_t= 0.0, 0.0, 0.0, 0.0, 0.0

    def add(self, loss, g, d,d_hz,d_t):
        self.loss += loss / self.div
        self.g += g / self.div
        self.d += d / self.div
        self.d_hz+=d_hz / self.div
        self.d_t+=d_t / self.div

    def output(self, s):
        print ('%s loss %.2f, g %.2f, d %.2f, d_hz %.2f, d_t %.2f' \
            % (s, self.loss, self.g, self.d,self.d_hz,self.d_t))

if __name__ == '__main__':

    args = load_arguments()

    #####   data preparation   #####
    if args.train:
        train0 = load_sent(args.train + '.0')
        train1 = load_sent(args.train + '.1')
        print ( '#heads of training file 0:', len(train0))
        print ( '#heads of training file 1:', len(train1))

        train_z0 = load_z(args.train_z+'.0')
        train_z1 = load_z(args.train_z + '.1')

        ## Normalize
        # train_z0 = preprocessing.normalize(train_z0)
        # train_z1 = preprocessing.normalize(train_z1)

        print ( '#descs of training z 0:', len(train_z0))
        print ( '#descs of training z 1:', len(train_z1))

        if not os.path.isfile(args.vocab):
            build_vocab(train0 + train1, args.vocab,min_occur=args.mini_occur)
        build_vocab(train0 + train1, args.vocab, min_occur=args.mini_occur)

    vocab = Vocabulary(args.vocab, args.embedding, args.dim_emb)
    print ( 'vocabulary size:', vocab.size)

    if args.dev:
        dev0 = load_sent(args.dev + '.0')
        dev1 = load_sent(args.dev + '.1')
        dev_z0 = load_z(args.dev_z+'.0')
        dev_z1 = load_z(args.dev_z+'.1')

    if args.test:
        test0 = load_sent(args.test + '.0')
        test1 = load_sent(args.test + '.1')
        test_z0 = load_z(args.test_z+'.0')
        test_z1 = load_z(args.test_z+'.1')



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        model = create_model(sess, args, vocab)


        if args.beam > 1:
            decoder = beam_search.Decoder(sess, args, vocab, model)
        else:
            decoder = greedy_decoding.Decoder(sess, args, vocab, model)

        if args.train:
            batches, _, _ = get_batches(train_z0, train_z1,train0, train1, vocab.word2id,
                args.batch_size)
            random.shuffle(batches)

            start_time = time.time()
            step = 0
            losses = Losses(len(batches))
            best_dev = float('inf')
            best_training = float('inf')
            learning_rate = args.learning_rate
            alpha = args.alpha
            gamma = args.gamma_init
            beta = args.beta
            eta = args.eta
            dropout = args.dropout_keep_prob

            for epoch in range(1, 1+args.max_epochs):
                print ( '--------------------epoch %d--------------------' % epoch)
                print ( 'learning_rate:', learning_rate, '  gamma:', gamma)

                for batch in batches:
                    feed_dict = feed_dictionary(model, batch, alpha, beta,gamma,eta,
                        dropout, learning_rate)

                    # train discriminator first

                    loss_d0_t, _ = sess.run([model.loss_d0_t, model.optimizer_d0_t],
                                          feed_dict=feed_dict)
                    loss_d1_t, _ = sess.run([model.loss_d1_t, model.optimizer_d1_t],
                                          feed_dict=feed_dict)
                    loss_d, _ = sess.run([model.loss_d, model.optimizer_d],
                                          feed_dict=feed_dict)
                    loss_d_hz, _ = sess.run([model.loss_d_hz, model.optimizer_d_hz],
                                          feed_dict=feed_dict)

                    # do not back-propagate from the discriminator
                    # when it is too poor
                    if loss_d0_t<1.5 and loss_d0_t<1.5 and loss_d<0.5 and loss_d_hz<10:
                        optimizer = model.optimizer_all
                    else:
                        optimizer = model.optimizer_ae

                    loss, loss_g, loss_d, loss_d_hz,loss_d_t,_ = sess.run(
                        [model.loss, model.loss_g, model.loss_d,model.loss_d_hz,model.loss_d_t,optimizer],
                        feed_dict=feed_dict)

                    losses.add(loss, loss_g, loss_d, loss_d_hz,loss_d_t)

                if losses.g < 30:
                    _ = transfer(model, decoder, sess, args, vocab, train_z0, train_z1,
                                      train0, train1, args.output + '.epoch%d' % epoch)

                losses.output('time %.0fs,'
                             % (time.time() - start_time))
                if losses.loss < best_training:

                    best_training = losses.loss
                    print ('saving model...')
                    model.saver.save(sess, args.model)

                losses.clear()

                gamma = max(args.gamma_min, gamma * args.gamma_decay)



