from itertools import product
import config
import os
from preprocessing_training_corpus import get_train_dev_corpus_file_name
from preprocessing_training_corpus import labels
from paramters import Parameters
from get_w2v_embedding import get_embedding_name
from multiprocessing import Pool
from functools import reduce
import fasttext
from evaluation import evaluation


def predicate(label, model_path):
    train_file, dev_file = get_train_dev_corpus_file_name(label=label)
    classifier = fasttext.load_model(model_path)
    p, r, f1 = evaluation(classifier, dev_file)
    print('p, r, f1 by self is : P: {} R: {} f1: {}'.format(p, r, f1))
    print(" {}_PRECISION: {}, RECALL: {}".format(label, result.precision, result.recall))
    return '{}-{}-precision-{}-recall-{}-f1-{}\n'.format(label, model_path, p, r, f1)


def train_and_predicate(label, model, dim, lr, windows, epoch):
    w2v_model_name = get_embedding_name(model, dim, lr, windows, epoch)
    w2v_model_path = w2v_model_name + '.vec'

    clf_path = '{}/clf/{}-{}-{}-{}-{}_model'.format(config.root, model, dim, lr, windows, epoch)

    train_file, dev_file = get_train_dev_corpus_file_name(label=label)

    if not os.path.exists(clf_path + '.bin'):
        classifier = fasttext.supervised(
            train_file,
            clf_path,
            dim=dim,
            pretrained_vectors=w2v_model_path,
        )
    else:
        classifier = fasttext.load_model(clf_path + '.bin')

    print('WHEN LABEL = {} DIM = {}, LR = {}, windows = {}, epoch = {}, model = {}'.format(label, dim, lr, windows, epoch, model))
    result = classifier.test(dev_file)
    p, r, f1 = evaluation(classifier, dev_file)
    print('p, r, f1 by self is : P: {} R: {} f1: {}'.format(p, r, f1))
    print(" PRECISION: {}, RECALL: {}".format(result.precision, result.recall))

    del classifier


def merge_result(already_notes, file):
    return already_notes + [file]


if __name__ == '__main__':
    P = Parameters
    cpu_num = 60

    pool = Pool(processes=cpu_num)

    file = 'path_train_recoding.txt'
    results = []
    # results = pool.starmap(train_and_predicate, product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs))
    # index = 0
    # start = 30 * index
    # step = 30 * (index + 1)
    # for ii, args in enumerate(product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)):
    #     if ii < start: continue
    #     if ii >= step: break
    #     try:
    #         results.append(train_and_predicate(*args))
    #     except MemoryError as e:
    #         print(e)
    #         continue

    pathes = [p.strip() for p in open('right_pathes.txt')]

    results = pool.starmap(predicate, product(labels, pathes))
    # for l, p in product(labels, pathes):
    #     results.append(predicate(l, p))

    result = reduce(merge_result, results, [])

    with open(file, 'a') as f:
        for r in result:
            f.write(r + '\n')
        # f.writelines(result)
