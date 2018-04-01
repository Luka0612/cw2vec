# -*- coding:utf-8 -*-
from generate_batch import generate_one
import numpy as np
import collections

stroke_index_dict = {}
for i in range(1, 6):
    for j in range(1, 6):
        for z in range(1, 6):
            stroke_index_dict[str(i) + str(j) + str(z)] = len(stroke_index_dict)
            for m in range(1, 6):
                stroke_index_dict[str(i) + str(j) + str(z) + str(m)] = len(stroke_index_dict)
                for n in range(1, 6):
                    stroke_index_dict[str(i) + str(j) + str(z) + str(m) + str(n)] = len(stroke_index_dict)


def get_dict_word_stroke_index(words_stroke_filename, dict_reverse_word_index):
    f = open(words_stroke_filename)
    other_dict_reverse_word_index = {}
    for key, value in dict_reverse_word_index.iteritems():
        other_dict_reverse_word_index[value] = key
    dict_word_stroke_index = {}
    max_stroke = 363
    for i in f:
        i = i.strip().split("\t")
        strokes = eval(i[2])
        strokes_transform = [stroke_index_dict[index] for index in strokes]
        for _ in range(max_stroke - len(strokes_transform)):
            # len(stroke_index_dict) = 3875
            strokes_transform.append(3875)
        dict_word_stroke_index[other_dict_reverse_word_index[i[0]]] = strokes_transform

    return dict_word_stroke_index


def generate_batch_cw(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index, words_stroke_filename):

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_windows
    generate_word = generate_one(file_name, num_skips, skip_windows)
    dict_word_stroke_index = get_dict_word_stroke_index(words_stroke_filename, dict_reverse_word_index)

    while True:

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):

            tuple_word = generate_word.next()
            while not tuple_word:
                generate_word = generate_one(file_name,num_skips, skip_windows)
                tuple_word = generate_word.next()

            for j in range(num_skips):

                batch[i * num_skips + j] = dict_word_stroke_index[tuple_word[j][0]]
                labels[i * num_skips + j, 0] = tuple_word[j][1]
        yield batch, labels


if __name__ == "__main__":
    print len(stroke_index_dict)
    f = open("../../data/word_index.txt")
    dict_reverse_word_index2 = collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index2[int(i[1])] = i[0]

    f.close()

    get_dict_word_stroke_index("../../data/words_stroke.txt", dict_reverse_word_index2)