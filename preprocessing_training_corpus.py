import pandas as pd
import random
import config
from utilities import format_string
import numpy as np


labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate',
]


def get_train_dev_corpus_file_name(label):
    return '{}/text/train-corpus-{}.txt'.format(config.root, label), \
           '{}/text/dev-corpus-{}.txt'.format(config.root, label)


def write_one_train_corpus(label):
    original_content = pd.read_csv('data/train.csv')

    labeled_data = original_content[label].tolist()
    black_indices = np.nonzero(np.array(labeled_data) == 1)[0].tolist()
    white_indices = np.nonzero(np.array(labeled_data) == 0)[0].tolist()

    expanding_ratio = (len(white_indices) - len(black_indices)) // len(black_indices)
    black_indices = black_indices * expanding_ratio
    print('expanding ratio is {}'.format(expanding_ratio))

    print('white indices is {}'.format(len(white_indices)))
    print('black indices is {}'.format(len(black_indices)))

    indices = black_indices + expanding_ratio

    [random.shuffle(indices) for _ in range(10)]

    train_ratio = 0.8
    train_length = int(len(indices) * train_ratio)
    train_indices = indices[: train_length]
    dev_indices = indices[train_length:]

    train_file, dev_file = get_train_dev_corpus_file_name(label)
    sentences = original_content['comment_text'].tolist()
    labels = original_content[label].tolist()

    def write_to_file(file, indices):
        with open(file, 'w') as f:
            for ii in indices:
                sentence = format_string(sentences[ii])
                f.write('__label__{} {}\n'.format(labels[ii], sentence))

    write_to_file(train_file, train_indices)
    write_to_file(dev_file, dev_indices)


if __name__ == '__main__':
    for l in labels:
        print('label {}'.format(l))
        write_one_train_corpus(l)

