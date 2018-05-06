#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from batch.generate_batch_image_character_stroke import generate_batch_image_character_level
from numpy import *
import math
import os
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import decomposition
import numpy as np
import tensorflow as tf
from pylab import mpl
from matplotlib.font_manager import FontProperties

font_set = FontProperties(fname=r"/Library/Fonts/Arial Unicode.ttf", size=15)
current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))


class Cw2VecSumStrokeTrain(object):
    def __init__(self, batch_size=128, embedding_size=128, skip_window=2, num_skips=4, valid_size=50,
                 valid_window=100, num_sampled=64, vocabulary_size=200000, stroke_size=3876, stroke_seq_length=363):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips

        self.valid_size = valid_size
        self.valid_window = valid_window
        self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
        self.num_sampled = num_sampled
        self.vocabulary_size = vocabulary_size
        self.stroke_size = stroke_size
        self.stroke_seq_length = stroke_seq_length
        self.image_shape = (36, 144)

    def train(self, file_name, words_stroke_filename, reverse_dictionary, save_model=True):

        gragh = tf.Graph()
        with gragh.as_default():

            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size, self.image_shape[0], self.image_shape[1], 1])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            with tf.device('/cpu:0'):
                # 卷积层1的输出是36*144*32
                with tf.variable_scope('layar1_conv1'):
                    conv1_weight = tf.get_variable("weight", [3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
                    conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
                    conv1 = tf.nn.conv2d(train_inputs, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
                    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
                # 池化层1的输出是18*72*32
                with tf.variable_scope('layar2_pool1'):
                    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                # 18*72*32
                with tf.variable_scope('layar3_conv2'):
                    conv2_weight = tf.get_variable("weight", [3, 3, 32, 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
                    conv2_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
                    conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
                    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
                # 池化层2的输出是 9*36*32
                with tf.variable_scope('layar4_pool2'):
                    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                # 9*36*32
                with tf.variable_scope('layar5_conv3'):
                    conv3_weight = tf.get_variable("weight", [3, 3, 32, 32],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
                    conv3_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
                    conv3 = tf.nn.conv2d(pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
                    relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

                relu3_shape = relu3.get_shape().as_list()
                nodes = relu3_shape[1] * relu3_shape[2] * relu3_shape[3]
                reshaped = tf.reshape(relu3, [relu3_shape[0], nodes])
                # 全连接层1：10368 -> 512
                with tf.variable_scope('layar6_fc1'):
                    fc1_weights = tf.get_variable("weight", [nodes, 512],initializer=tf.truncated_normal_initializer(stddev=0.1))
                    fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.0))
                    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
                # 全连接层1：512 -> 128
                with tf.variable_scope('layar7_fc2'):
                    fc2_weights = tf.get_variable("weight", [512, 128],
                                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
                    fc2_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
                    embed = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=embeddings, biases=nce_biases, labels=train_labels, inputs=embed,
                               num_sampled=self.num_sampled, num_classes=self.vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            init = tf.initialize_all_variables()

        num_steps = 2000001

        with tf.Session(graph=gragh) as session:
            init.run()
            print("Initialized")

            generate = generate_batch_image_character_level(file_name, self.batch_size, self.num_skips, self.skip_window,
                                         reverse_dictionary, words_stroke_filename, self.stroke_seq_length)
            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate.next()
                reshape_inputs = np.reshape(batch_inputs, (self.batch_size, self.image_shape[0], self.image_shape[1], 1))
                feed_dict = {train_inputs: reshape_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step", step, ":", average_loss)
                    average_loss = 0
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in range(self.valid_size):
                        valid_word = reverse_dictionary[self.valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)
                        if step % 500000 == 0 and save_model:
                            self.model_save(session, current_relative_path("../../../model/cw2vec/"), "model.ckpt",
                                            step)
            final_embeddings = normalized_embeddings.eval()
            self.plot_labels(final_embeddings, reverse_dictionary)
            if save_model:
                self.model_save(session, current_relative_path("../../../model/cw2vec/"), "model.ckpt", num_steps)

    def model_save(self, sess, path, model_name, global_step):
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(path, model_name), global_step=global_step)

    def plot_with_labels(self, low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom',
                         fontproperties=font_set)
        plt.savefig(filename)

    def plot_labels(self, final_embeddings, reverse_dictionary):
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 100
        X = final_embeddings[:plot_only, :]
        pca = decomposition.PCA(n_components=2, copy=True, whiten=False)
        low_dim_embs = pca.fit_transform(X)
        labels = [reverse_dictionary[i].decode("utf-8") for i in range(plot_only)]
        self.plot_with_labels(low_dim_embs, labels)


def get_length(seq):
    used = tf.sign(tf.reduce_max(tf.abs(seq), 2))
    seq_len = tf.reduce_sum(used, 1)
    seq_len = tf.cast(seq_len, tf.int32)
    return seq_len

if __name__ == '__main__':
    a = tf.constant([[0, 0, 0], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    b = tf.constant([[1, 2, 0, 0], [2, 3, 1, 0]])
    input_leng = [2, 3]
    embed = tf.nn.embedding_lookup(a, b)
    seq_len = get_length(embed)
    # c = tf.reduce_sum(embed, 1)
    output = []
    for i in range(2):
        output.append(tf.reduce_sum(embed[i][:seq_len[i]], 0))

    session = tf.Session()
    print session.run(output)