# -*- coding:utf-8 -*-
import numpy as np
import random


def generate_one(file_name, num_skips, skip_windows):

    f = open(file_name, "r")
    for row in f:
        row = eval(row.strip().split("\t")[1])
        row_length = len(row)
        for word_num in range(row_length):
            list_word_num = list()
            for num in range(1, skip_windows+1):
                if word_num - num >= 0:
                    list_word_num.append(word_num - num)
                if word_num + num < row_length:
                    list_word_num.append(word_num + num)
            if len(list_word_num) < num_skips:continue
            elif len(list_word_num) > num_skips:list_word_num = random.sample(list_word_num, num_skips)
            list_word = list()
            for num in list_word_num:
                list_word.append((row[word_num], row[num]))
            yield list_word

    yield None


def generate_batch(file_name, batch_size, num_skips, skip_windows):

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_windows
    generate_word = generate_one(file_name, num_skips, skip_windows)

    while True:

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):

            tuple_word = generate_word.next()
            while not tuple_word:
                generate_word = generate_one(file_name,num_skips, skip_windows)
                tuple_word = generate_word.next()

            for j in range(num_skips):

                batch[i * num_skips + j] = tuple_word[j][0]
                labels[i * num_skips + j, 0] = tuple_word[j][1]
        yield batch, labels
