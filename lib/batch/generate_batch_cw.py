# -*- coding:utf-8 -*-
from generate_batch import generate_batch, generate_one
import numpy as np
import collections


def get_dict_word_stroke_index():
    f = open("../../data/words_stroke.txt")
    dict_word_stroke_index = {}
    stroke_index = 0
    for i in f:
        i = i.strip().split("\t")
        strokes = eval(i[2])
        strokes_transform = [stroke_index+index for index in range(len(strokes))]
        dict_word_stroke_index[i[0]] = strokes_transform
        stroke_index += len(strokes)
    return dict_word_stroke_index


def generate_batch_cw(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index):
    # stroke_index
    dict_word_stroke_index = get_dict_word_stroke_index()
    # data
    generate_word = generate_one(file_name, num_skips, skip_windows)
    tuple_words = generate_word.next()
    # return_data
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_index = 0

    while True:
        for i in range(num_skips):
            tuple_word_batch = tuple_words[i][0]
            tuple_word_label = tuple_words[i][1]

            if dict_reverse_word_index[tuple_word_batch] not in dict_word_stroke_index:continue
            tuple_word_batch_strokes = dict_word_stroke_index[dict_reverse_word_index[tuple_word_batch]]
            for tuple_word_batch_stroke in tuple_word_batch_strokes:
                batch[batch_index] = tuple_word_batch_stroke
                labels[batch_index] = tuple_word_label
                batch_index += 1
                if batch_index == batch_size:
                    yield batch, labels
                    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
                    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
                    batch_index = 0

        tuple_words = generate_word.next()
        while not tuple_words:
            generate_word = generate_one(file_name, num_skips, skip_windows)
            tuple_words = generate_word.next()

if __name__ == "__main__":
    f = open("../../data/word_index.txt")
    dict_reverse_word_index = collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index[int(i[1])] = i[0]

    f.close()
    generate = generate_batch_cw("../../data/train_data.txt", 8, 2, 1, dict_reverse_word_index)

    print generate.next()