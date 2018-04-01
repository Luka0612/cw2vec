# -*- coding:utf-8 -*-
import sys
import os
import collections
sys.path.append('../lib/train')

from word2vec.model_train import Word2VecTrain

from cw2vec.model_train import Cw2VecTrain


def word2vec_train():
    f = open("../data/word_index.txt")
    dict_reverse_word_index= collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index[int(i[1])] = i[0]

    f.close()

    W2V = Word2VecTrain()
    W2V.train(os.path.abspath("../data/train_data.txt"), dict_reverse_word_index)


def cw2vec_train():
    f = open("../data/word_index.txt")
    dict_reverse_word_index= collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index[int(i[1])] = i[0]

    f.close()

    W2V = Cw2VecTrain()
    W2V.train(os.path.abspath("../data/train_data.txt"), os.path.abspath("../data/words_stroke.txt"), dict_reverse_word_index)


if __name__ == "__main__":
    cw2vec_train()