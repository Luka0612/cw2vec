# -*- coding:utf-8 -*-
import glob

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

from generate_batch import generate_one


def image2np(filename):
    # 将图像转为0-1
    return (255 - np.array(Image.open(filename).convert('L')).astype('float')) / 255


def image2np_mpimg(filename):
    return 1 - np.array(mpimg.imread(filename))


def get_all_image_data(path, dict_reverse_word_index):

    dict_word_index = {}
    for key, value in dict_reverse_word_index.iteritems():
        dict_word_index[value] = key

    all_image_data = {}
    all_images_filename = glob.glob(path + "*.png")
    for image_filename in all_images_filename:
        text = image_filename.strip().split("/")[-1].split(".png")[0]
        all_image_data[dict_word_index[text]] = image2np(image_filename)
    return all_image_data


def generate_batch_character_level(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index,
                                   words_image_filename):
    all_image_data = get_all_image_data(words_image_filename, dict_reverse_word_index)
    image_shape = all_image_data.values()[0].shape
    # data
    generate_word = generate_one(file_name, num_skips, skip_windows)
    tuple_words = generate_word.next()
    # return_data
    batch = np.ndarray(shape=(batch_size, image_shape[0], image_shape[1]), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    batch_index = 0
    while True:
        for i in range(num_skips):
            tuple_word_batch = tuple_words[i][0]
            tuple_word_label = tuple_words[i][1]
            if tuple_word_batch not in all_image_data: continue
            tuple_word_batch_image = all_image_data[tuple_word_batch]
            batch[batch_index] = tuple_word_batch_image
            labels[batch_index] = tuple_word_label
            batch_index += 1
            if batch_index == batch_size:
                yield batch, labels
                batch = np.ndarray(shape=(batch_size, image_shape[0], image_shape[1]), dtype=np.int32)
                labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
                batch_index = 0

        tuple_words = generate_word.next()
        while not tuple_words:
            generate_word = generate_one(file_name, num_skips, skip_windows)
            tuple_words = generate_word.next()


def generate_batch_image_character_level(file_name, batch_size, num_skips, skip_windows, dict_reverse_word_index, words_image_filename):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_windows
    generate_word = generate_one(file_name, num_skips, skip_windows)
    all_image_data = get_all_image_data(words_image_filename, dict_reverse_word_index)
    image_shape = all_image_data.values()[0].shape

    while True:

        batch = np.ndarray(shape=(batch_size, image_shape[0], image_shape[1]), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        for i in range(batch_size // num_skips):

            tuple_word = generate_word.next()
            while not tuple_word:
                generate_word = generate_one(file_name, num_skips, skip_windows)
                tuple_word = generate_word.next()
            while True:
                ner_in_dict_word_stroke_index = True
                for j in range(num_skips):
                    if tuple_word[j][0] not in all_image_data:
                        ner_in_dict_word_stroke_index = False
                if ner_in_dict_word_stroke_index:
                    break
                else:
                    tuple_word = generate_word.next()
                    while not tuple_word:
                        generate_word = generate_one(file_name, num_skips, skip_windows)
                        tuple_word = generate_word.next()

            for j in range(num_skips):
                batch[i * num_skips + j] = all_image_data[tuple_word[j][0]]
                labels[i * num_skips + j, 0] = tuple_word[j][1]
        yield batch, labels

if __name__ == '__main__':
    import collections

    f = open("../../data/word_index.txt")
    dict_reverse_word_index2 = collections.defaultdict(str)
    for i in f:
        i = i.strip().split("\t")
        dict_reverse_word_index2[int(i[1])] = i[0]

    f.close()

    generate = generate_batch_character_level("../../data/train_data.txt", 8, 2, 1, dict_reverse_word_index2,
                                 "../../data/text_image_pil_length_4/")

    print generate.next()


