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

data_directory = "../Data"
backup_directory = "../Models/"

dataset_file_path = data_directory+"/train_dataset"

print("Restore Data")

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)

print("DATASET :", np.shape(dataset))
print("train_X :", np.shape(dataset[0]))
print("train_y :", np.shape(dataset[1]))
print("test_X :", np.shape(dataset[2]))
print("test_y :", np.shape(dataset[3]))

train_X = np.array(dataset[0])
#train_X_lenght = np.array(dataset[1])
train_y = np.array(dataset[1])
test_X = np.array(dataset[2])
#test_X_lenght = np.array(dataset[4])
test_y = np.array(dataset[3])

n_class = 3
embedding_dim = 100
max_sen_len = 30

hidden_size = 100

learning_rate = 0.001
batch_size = 100
test_batch_size = 200
num_epochs = 10
evaluate_every = 500

nb_batch_per_epoch = int(len(train_X)/batch_size+1)
nb_batch_per_epoch_test = int(len(test_X)/test_batch_size+1)

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
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_dpc")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        best_accuracy = 0.
        predict_round = 0

        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.shuffle(indices)
            train_X = train_X[indices]
            train_y = train_y[indices]
            

            for batch in range(nb_batch_per_epoch):

                # normalized batch :
                idx = batch * batch_size
                x1 = np.array([train_X[idx][0]])
                x2 = np.array([train_X[idx][1]])
                y = np.array([train_y[idx]])

                class_sum = train_y[idx]
                while np.sum(class_sum) < batch_size:
                    idx = (idx+1)%len(train_X)
                    if class_sum[np.argmax(train_y[idx])] <= batch_size/n_class+1:
                        x1 = np.append(x1, np.array([train_X[idx][0]]), axis=0)
                        x2 = np.append(x2, np.array([train_X[idx][1]]), axis=0)
                        y = np.append(y, np.array([train_y[idx]]), axis=0)
                        class_sum = np.add(class_sum, train_y[idx])
                
                
                

               
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

                if current_step % evaluate_every == 0:
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
                        print("Saved model checkpoint to {}\n".format(path))