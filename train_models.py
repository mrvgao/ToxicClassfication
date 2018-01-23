from itertools import product
import config
from preprocessing_training_corpus import get_train_dev_corpus_file_name
from preprocessing_training_corpus import labels
from paramters import Parameters
from get_w2v_embedding import get_embedding_name
import fasttext
from functools import reduce
from tqdm import tqdm
from utilities import remove_folder
import os


def get_classifier_path(label, method, dim, lr, windows, epoch):
    clf_path = '{}{}-{}-{}-{}-{}-{}_model'.format(config.clf_root, label, method, dim, lr, windows, epoch)
    return clf_path


def train(label, model, dim, lr, windows, epoch, first_train=False):
    if first_train:
        remove_folder(config.clf_root)
        assert len(os.listdir(config.clf_root)) == 0

    w2v_model_name = get_embedding_name(model, dim, lr, windows, epoch)
    w2v_model_path = w2v_model_name + '.vec'
    clf_path = get_classifier_path(label, model, dim, lr, windows, epoch)
    train_file, dev_file = get_train_dev_corpus_file_name(label=label)
    _ = fasttext.supervised(
            train_file,
            clf_path,
            dim=dim,
            pretrained_vectors=w2v_model_path,
            thread=80,
            word_ngram=2,
    )


def get_length(*iterators):
    return reduce(lambda x, y: x * y, map(len, iterators), 1)


if __name__ == '__main__':
    P = Parameters
    length = get_length(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)
    for ii, args in tqdm(enumerate(product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)), total=length):
        # print('batch {}/{}'.format(ii, length))
        train(*args)
