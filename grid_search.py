import fasttext
from itertools import product


def train(model, dim, lr, windows, epoch, f, thread):
    w2v_model_name = './cust_data/w2v-{}-{}-{}-{}-{}'.format(model, dim, lr, windows, epoch)
    if model == 'skip':
        func = fasttext.skipgram
    elif model == 'cbow':
        func = fasttext.cbow

    print('training embedding')
    func('./cust_data/train_corpus.txt',
         w2v_model_name,
         dim=dim, lr=lr, ws=windows, epoch=epoch, thread=thread)

    w2v_model_path = w2v_model_name + '.vec'
    clf_path = './cust_data/toxic_clasifier-{}-{}-{}-{}-{}_model'.format(model, dim, lr, windows, epoch),

    classifier = fasttext.supervised(
        './cust_data/train_corpus.txt',
        clf_path,
        dim=100, pretrained_vectors=w2v_model_path
    )

    print('WHEN DIM = {}, LR = {}, windows = {}, epoch = {}, model = {}'.format(dim, lr, windows, epoch, model))
    result = classifier.test('./cust_data/dev_corpus.txt')
    print(" PRECISION: {}, RECALL: {}".format(result.precision, result.recall))

    f.write('{}-{}-{}-{}-{}-prcision-{}-recall-{}'.format(model, dim, lr, windows, epoch, result.precision, result.recall))


if __name__ == '__main__':
    models = ['cbow', 'skip']
    dimensons = [30, 50, 80, 100, 150, 200, 300]
    learning_rates = [1e-3, 1e-2, 0.05, 1e-1]
    ws = [3, 5, 7]
    epoch = [3, 5, 10]
    threads = 50

    with open('train_recoding.txt', 'w') as f:
        for m, d, l, w, e in product(models, dimensons, learning_rates, ws, epoch):
            train(model=m, dim=d, lr=l, windows=w, epoch=e, f=f, thread=threads)
