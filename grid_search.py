from pyfasttext import FastText
from itertools import product
import config
import os
from preprocessing_training_corpus import get_train_dev_corpus_file_name
from preprocessing_training_corpus import labels
from paramters import Parameters
from get_w2v_embedding import get_embedding_name
from multiprocessing import Pool
from functools import reduce
# import fasttext


def train_and_predicate(label, model, dim, lr, windows, epoch):
    w2v_model_name = get_embedding_name(model, dim, lr, windows, epoch)
    w2v_model_path = w2v_model_name + '.vec'

    clf_path = '{}/clf/{}-{}-{}-{}-{}_model'.format(config.root, model, dim, lr, windows, epoch)

    train_file, dev_file = get_train_dev_corpus_file_name(label=label)

    fasttext = FastText()

    if not os.path.exists(clf_path + '.bin'):
        fasttext.supervised(
        # classifier = fasttext.supervised(
            train_file,
            clf_path,
            dim=dim,
            pretrained_vectors=w2v_model_path,
        )
    else:
        # classifier = fasttext.load_model(clf_path + '.bin')
        fasttext.load_model(clf_path + '.bin')

    print('WHEN LABEL = {} DIM = {}, LR = {}, windows = {}, epoch = {}, model = {}'.format(label, dim, lr, windows, epoch, model))
    # result = classifier.test(dev_file)
    fasttext.test(dev_file)
    # print(" PRECISION: {}, RECALL: {}".format(result.precision, result.recall))

    del fasttext

    # with open(record, 'a') as f:
    # return '{}-{}-{}-{}-{}-{}-precision-{}-recall-{}\n'.format(label, model, dim, lr, windows, epoch, result.precision, result.recall)


def merge_result(already_notes, file):
    return already_notes + [file]


if __name__ == '__main__':
    P = Parameters
    cpu_num = 50

    pool = Pool(processes=cpu_num)

    # for label in labels:
    #     record = '{}_train_recoding.txt'.format(label)
    file = 'train_recoding.txt'
    results = []
    # results = pool.starmap(train_and_predicate, product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs))
    for args in product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs):
        results.append(train_and_predicate(*args))

    # result = reduce(merge_result, results, [])
    #
    # with open(file, 'a') as f:
    #     f.writelines(result)
