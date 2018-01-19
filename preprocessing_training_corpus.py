import pandas as pd
import random


labels = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hat',
]


def get_train_dev_corpus_file_name(label):
    return './cust_data/train-corpus-{}.txt'.format(label), './cust_data/dev-corpus-{}.txt'.format(label)


def write_one_train_corpus(label):
    original_content = pd.read_csv('data/train.csv')

    indices = list(range(len(original_content)))

    [random.shuffle(indices) for _ in range(10)]

    train_ratio = 0.8
    train_length = int(len(indices) * train_ratio)
    train_indices = indices[: train_length]
    dev_indices = indices[train_length:]

    train_file, dev_file = get_train_dev_corpus_file_name(label)
    sentences = original_content['comment_text'].tolist()
    labels = original_content[label].tolist()
    with open(train_file) as f:
        for index in train_indices:
            f.write('__label__{} {}\n'.format(labels[index], sentences[index]))

    with open(dev_file) as f:
        for index in dev_indices:
            f.write('__label__{} {}\n'.format(labels[index], sentences[index]))


if __name__ == '__main__':
    for l in labels:
        write_one_train_corpus(l)




