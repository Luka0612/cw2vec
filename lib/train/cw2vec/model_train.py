#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from batch.generate_batch_cw import generate_batch_cw
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


class Cw2VecTrain(object):
    def __init__(self, batch_size=1280, embedding_size=128, skip_window=2, num_skips=4, valid_size=50,
                 valid_window=100, num_sampled=64, vocabulary_size=200000, stroke_size=3875):
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

    def train(self, file_name, words_stroke_filename, reverse_dictionary, save_model=True):

        gragh = tf.Graph()
        with gragh.as_default():

            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            with tf.device('/cpu:0'):
                embeddings_stroke = tf.Variable(tf.random_uniform([self.stroke_size, self.embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings_stroke, train_inputs)

                embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=embeddings, biases=nce_biases, labels=train_labels, inputs=embed,
                               num_sampled=self.num_sampled, num_classes=self.vocabulary_size))

            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

            init = tf.initialize_all_variables()

        num_steps = 20000001

        with tf.Session(graph=gragh) as session:
            init.run()
            print("Initialized")

            generate = generate_batch_cw(file_name, self.batch_size, self.num_skips, self.skip_window, reverse_dictionary, words_stroke_filename)
            average_loss = 0
            for step in range(num_steps):
                batch_inputs, batch_labels = generate.next()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += loss_val

                if step % 20000 == 0:
                    if step > 0:
                        average_loss /= 20000
                    print("Average loss at step", step, ":", average_loss)
                    average_loss = 0
                if step % 100000 == 0:
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
                        if step % 5000000 == 0 and save_model:
                            self.model_save(session, current_relative_path("../../../model/cw2vec/"), "model.ckpt", step)
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
