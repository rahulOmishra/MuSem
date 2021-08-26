#!/usr/bin/env python
# coding: utf-8

import pickle
import os
import datetime
import time

import tensorflow as tf
import numpy as np

from model_dpc import Model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "../Dataset_50_300"
backup_directory = "../Models/"

n_class = 2
embedding_dim = 300
max_sen_len = 50

hidden_size = 300

learning_rate = 0.001
batch_size = 100
test_batch_size = 200
num_epochs = 20
evaluate_every = 500

nb_batch_per_epoch = 2885

allow_soft_placement = True
log_device_placement = False

with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement
    )
    session_config.gpu_options.allow_growth = False
    session_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=session_config)

    with sess.as_default():

        print("Model creation")

        model = Model(
        max_sen_len = max_sen_len,
        embedding_dim = embedding_dim,
        class_num = n_class,
        hidden_size = hidden_size
        )

        print("Model construction")

        model.build_model()

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath(backup_directory+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_dpc_50_300")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        best_accuracy = 0.
        predict_round = 0

        for epoch in range(num_epochs):
            
            for batch in range(nb_batch_per_epoch):

                print("Restore batch :", batch, " epoch :", epoch)

                dataset_file_path = data_directory+"/train_"+str(epoch)+"_"+str(batch)

                with open(dataset_file_path, 'rb') as f:
                    dataset = pickle.load(f)

                x1 = dataset[0]
                x2 = dataset[1]
                l1 = dataset[2]
                l2 = dataset[3]
                y = dataset[4]

 

                feed_dict = {
                    model.x1: x1,
                    model.x2: x2,
                    model.y: y,
                    #model.class_weights: class_weights,
                    #model.class_weights_accuracy: class_weights_accuracy,
                }

                _, step, loss, accuracy, c_matrix = sess.run(
                    [train_op, global_step, model.loss, model.accuracy, model.c_matrix], 
                    feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}/{}, loss {:g}, acc {:g}".format(time_str, step, num_epochs*nb_batch_per_epoch, loss, accuracy))
                print(c_matrix)

                current_step = tf.train.global_step(sess, global_step)

                '''if current_step % evaluate_every == 0:
                    predict_round += 1
                    print("\nEvaluation round %d:" % (predict_round))
                    
                    indices = np.arange(len(test_X))
                    np.random.shuffle(indices)
                    test_X = test_X[indices]
                    test_y = test_y[indices]

                    accuracy = 0
                    c_matrix = np.zeros((n_class, n_class))

                    for test_batch in range(nb_batch_per_epoch_test):
                        idx_min = test_batch * test_batch_size
                        idx_max = min((test_batch+1) * test_batch_size, len(test_X)-1)
                        
                        x1 = test_X[idx_min:idx_max, 0]
                        x2 = test_X[idx_min:idx_max, 1]

                        y = test_y[idx_min:idx_max]

                        
                        
                        feed_dict = {
                            model.x1: x1,
                            model.x2: x2,
                            model.y: y,
                            #model.class_weights: class_weights,
                            #model.class_weights_accuracy: class_weights_accuracy,
                        }

                        batch_accuracy, batch_c_matrix = sess.run([model.accuracy, model.c_matrix], feed_dict=feed_dict)
                        accuracy = accuracy + batch_accuracy
                        c_matrix = np.add(c_matrix, batch_c_matrix)

                    accuracy = accuracy/nb_batch_per_epoch_test
                    print("Test acc {:g}".format(accuracy))
                    print("C_matrix ", c_matrix)

                    if accuracy >= best_accuracy: 
                        best_accuracy = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))'''