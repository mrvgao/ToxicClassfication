from itertools import product
import os
from preprocessing_training_corpus import get_train_dev_corpus_file_name
from preprocessing_training_corpus import labels
from paramters import Parameters
from get_w2v_embedding import get_embedding_name
from multiprocessing import Pool
from functools import reduce
import fasttext
from evaluation import evaluation
from train_models import get_classifier_path
import config


def predicate(model_path):
    label = model_path.split('-')[0].split('/')[-1]
    print('label is {}'.format(label))
    train_file, dev_file = get_train_dev_corpus_file_name(label=label)
    classifier = fasttext.load_model(model_path)

    t_p, t_r, t_f1 = evaluation(classifier, train_file)
    d_p, d_r, d_f1 = evaluation(classifier, dev_file)
    print(" {}_@T-P:{}, @T-R:{} @T-f1: {};  @D-P:{}, @D-R:{} @D-f1: {} ".format(
        label, t_p, t_r, t_f1, d_p, d_r, d_f1))
    return '{}-{}-t_precision-{}-t_recall-{}-t_f1-{}-d_precision-{}-d_recall-{}-d_f1-{}\n'.format(
        label, model_path, t_p, t_r, t_f1, d_p, d_r, d_f1)


def train_and_predicate(label, model, dim, lr, windows, epoch):
    w2v_model_name = get_embedding_name(model, dim, lr, windows, epoch)
    w2v_model_path = w2v_model_name + '.vec'

    # clf_path = '{}/clf/{}-{}-{}-{}-{}_model'.format(config.root, model, dim, lr, windows, epoch)
    clf_path = get_classifier_path(model, dim, lr, windows, epoch)

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


if __name__ == '__main__':
    pass

