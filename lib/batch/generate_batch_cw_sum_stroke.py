# -*- coding:utf-8 -*-
from generate_batch import generate_one
import numpy as np
import collections

stroke_index_dict = {}
for i in range(1, 6):
    for j in range(1, 6):
        for z in range(1, 6):
            stroke_index_dict[str(i) + str(j) + str(z)] = len(stroke_index_dict) + 1
            for m in range(1, 6):
                stroke_index_dict[str(i) + str(j) + str(z) + str(m)] = len(stroke_index_dict) + 1
                for n in range(1, 6):
                    stroke_index_dict[str(i) + str(j) + str(z) + str(m) + str(n)] = len(stroke_index_dict) + 1


def get_dict_word_stroke_index(words_stroke_filename, dict_reverse_word_index, max_stroke):
    f = open(words_stroke_filename)
    other_dict_reverse_word_index = {}
    for key, value in dict_reverse_word_index.iteritems():
        other_dict_reverse_word_index[value] = key
    dict_word_stroke_index = {}
    for i in f:
        i = i.strip().split("\t")
        strokes = eval(i[2])
        strokes_transform = [stroke_index_dict[index] for index in strokes]
        if max_stroke > len(strokes_transform):
            for _ in range(max_stroke - len(strokes_transform)):
                # len(stroke_index_dict) = 3875
                strokes_transform.append(0)
        else:
            strokes_transform = strokes_transform[:max_stroke]
        dict_word_stroke_index[other_dict_reverse_word_index[i[0]]] = strokes_transform

    return dict_word_stroke_index


def generate_batch_cw(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index, words_stroke_filename,
                      stroke_seq_length):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_windows
    generate_word = generate_one(file_name, num_skips, skip_windows)
    dict_word_stroke_index = get_dict_word_stroke_index(words_stroke_filename, dict_reverse_word_index,
                                                        stroke_seq_length)

    while True:

        batch = np.ndarray(shape=(batch_size, stroke_seq_length), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):

            tuple_word = generate_word.next()
            while not tuple_word:
                generate_word = generate_one(file_name, num_skips, skip_windows)
                tuple_word = generate_word.next()
            while True:
                ner_in_dict_word_stroke_index = True
                for j in range(num_skips):
                    if tuple_word[j][0] not in dict_word_stroke_index:
                        ner_in_dict_word_stroke_index = False
                if ner_in_dict_word_stroke_index:
                    break
                else:
                    tuple_word = generate_word.next()
                    while not tuple_word:
                        generate_word = generate_one(file_name, num_skips, skip_windows)
                        tuple_word = generate_word.next()

            for j in range(num_skips):
                batch[i * num_skips + j] = dict_word_stroke_index[tuple_word[j][0]]
                labels[i * num_skips + j, 0] = tuple_word[j][1]
        yield batch, labels


def generate_batch_cw_sum_stroke(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index,
                                 words_stroke_filename, stroke_seq_length):
    dict_word_stroke_index = get_dict_word_stroke_index(words_stroke_filename, dict_reverse_word_index,
                                                        stroke_seq_length)
    # data
    generate_word = generate_one(file_name, num_skips, skip_windows)
    tuple_words = generate_word.next()
    # return_data
    batch = np.ndarray(shape=(batch_size, stroke_seq_length), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_index = 0
    while True:
        for i in range(num_skips):
            tuple_word_batch = tuple_words[i][0]
            tuple_word_label = tuple_words[i][1]
            if tuple_word_batch not in dict_word_stroke_index: continue
            tuple_word_batch_strokes = dict_word_stroke_index[tuple_word_batch]
            batch[batch_index] = tuple_word_batch_strokes
            labels[batch_index] = tuple_word_label
            batch_index += 1
            if batch_index == batch_size:
                yield batch, labels
                batch = np.ndarray(shape=(batch_size, stroke_seq_length), dtype=np.int32)
                labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
                batch_index = 0

        tuple_words = generate_word.next()
        while not tuple_words:
            generate_word = generate_one(file_name, num_skips, skip_windows)
            tuple_words = generate_word.next()


if __name__ == "__main__":
    print len(stroke_index_dict)
    f = open("../../data/word_index.txt")
    dict_reverse_word_index2 = collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index2[int(i[1])] = i[0]

    f.close()

    generate = generate_batch_cw("../../data/train_data.txt", 8, 2, 1, dict_reverse_word_index2,
                                 "../../data/words_stroke.txt", 363)

    print generate.next()
