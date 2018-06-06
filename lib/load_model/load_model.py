# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import collections


def word_index():
    f = open("../../data/word_index.txt")
    dict_reverse_word_index_return = collections.defaultdict(str)
    dict_word_index_return = collections.defaultdict(int)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index_return[int(i[1])] = i[0]
        dict_word_index_return[i[0]] = int(i[1])
    f.close()
    return dict_reverse_word_index_return, dict_word_index_return

dict_reverse_word_index, dict_word_index = word_index()

vocabulary_size = 200000
embedding_size = 128


def get_normalized_embeddings():
    gragh = tf.Graph()
    with gragh.as_default():
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 模型所在位置
            saver.restore(sess, "../../data/word2vec.ckpt")

            embeddings = sess.run(normalized_embeddings)
            return embeddings

word_normalized_embeddings = get_normalized_embeddings()


def get_word_embeddings(word):
    word_index = dict_word_index[word]
    # gragh = tf.Graph()
    # with gragh.as_default():
    #     valid_dataset = tf.constant(np.array([word_index]), dtype=tf.int32)
    #     valid_embeddings = tf.nn.embedding_lookup(word_normalized_embeddings, valid_dataset)
    #     with tf.Session() as sess:
    #         return sess.run(valid_embeddings)[0]
    return word_normalized_embeddings[word_index]


def word_sim(word1, word2):
    word1_embedding = get_word_embeddings(word1)
    word2_embedding = get_word_embeddings(word2)
    num = np.sum(np.multiply(word1_embedding, word2_embedding))

    denom = np.linalg.norm(word1_embedding) * np.linalg.norm(word2_embedding)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return sim


if __name__ == "__main__":
    # print type(word_normalized_embeddings)
    # word_embedding = get_word_embeddings("快乐")
    # word_embedding = np.array(list(word_embedding))
    # print word_embedding.shape
    print word_sim("快乐", "幸福")
    print word_sim("快乐", "绝望")
    print word_sim("痛苦", "绝望")
