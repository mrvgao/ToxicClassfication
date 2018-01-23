import pandas as pd
import random
import config
from utilities import format_string
import numpy as np
from utilities import remove_folder
import os


labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


def remove_corpus_files():
    remove_folder('{}/text'.format(config.root))
    assert len(os.listdir('{}/text'.format(config.root))) == 0


def get_train_dev_corpus_file_name(label):
    return '{}/text/train-corpus-{}.txt'.format(config.root, label), \
           '{}/text/dev-corpus-{}.txt'.format(config.root, label)


def write_one_train_corpus(label, first_write=False):
    if first_write: remove_corpus_files()

    original_content = pd.read_csv('data/train.csv')

    train_ratio = 0.75

    labeled_data = original_content[label].tolist()

    black_indices = np.nonzero(np.array(labeled_data) == 1)[0].tolist()
    white_indices = np.nonzero(np.array(labeled_data) == 0)[0].tolist()

    [np.random.shuffle(black_indices) for _ in range(10)]
    [np.random.shuffle(white_indices) for _ in range(10)]

    train_black_indices_num = int(len(black_indices) * train_ratio)
    train_white_indices_num = int(len(white_indices) * train_ratio)

    expanding_ratio = len(white_indices) // len(black_indices) + 1

    train_black_indices = black_indices[: train_black_indices_num]
    dev_black_indices = black_indices[train_black_indices_num: ]
    train_black_indices = train_black_indices * expanding_ratio
    dev_black_indices = dev_black_indices * expanding_ratio

    train_white_indices = white_indices[: train_white_indices_num]
    dev_white_indices = white_indices[train_white_indices_num:]

    train_length = min(len(train_black_indices), len(train_white_indices))
    train_black_indices = train_black_indices[: train_length]

    # assert train_length == len(train_white_indices) == len(train_black_indices)
    print('ratio of training corpus is {}'.format(len(train_white_indices)/len(train_black_indices)))

    dev_length = min(len(dev_black_indices), len(dev_white_indices))
    dev_black_indices = dev_black_indices[: dev_length]

    print('ratio of dev corpus is {}'.format(len(dev_white_indices)/len(dev_black_indices)))

    assert dev_length == len(dev_white_indices) == len(dev_black_indices)

    train_file, dev_file = get_train_dev_corpus_file_name(label)
    sentences = original_content['comment_text'].tolist()
    Y = original_content[label].tolist()

    train_indices = train_white_indices + train_black_indices
    dev_indices = dev_white_indices + dev_black_indices

    [random.shuffle(train_indices) for _ in range(10)]
    [random.shuffle(dev_indices) for _ in range(10)]

    train_indices_set = set(train_indices)
    dev_indices_set = set(dev_indices)

    assert train_indices_set.isdisjoint(dev_indices_set)

    assert len(sentences) == len(Y)

    def write_to_file(file, indices):
        with open(file, 'w', encoding='utf-8') as f:
            for ii in indices:
                sentence = format_string(sentences[ii])
                f.write('__label__{} {}\n'.format(Y[ii], sentence))

    write_to_file(train_file, train_indices)
    write_to_file(dev_file, dev_indices)


if __name__ == '__main__':
    for l in labels:
        print('label {}'.format(l))
        write_one_train_corpus(l)

