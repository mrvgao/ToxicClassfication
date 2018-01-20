import fasttext
import config
from multiprocessing import Pool
from paramters import Parameters
from itertools import product


def get_embedding_name(model, dim, lr, windows, epoch):
    w2v_model_name = '{}/w2v/w2v-{}-{}-{}-{}-{}'.format(config.root, model, dim, lr, windows, epoch)

    return w2v_model_name


def train_embedding(model, dim, lr, windows, epoch):
    if model == 'skip':
        func = fasttext.skipgram
    elif model == 'cbow':
        func = fasttext.cbow

    w2v_model_name = get_embedding_name(model, dim, lr, windows, epoch)
    print('training embedding {}'.format(w2v_model_name))
    func(config.line_corpus, w2v_model_name, dim=dim, lr=lr, ws=windows, epoch=epoch, thread=60)
    print('embedding end: {}'.format(w2v_model_name))


if __name__ == '__main__':
    cpu_number = 60
    pool = Pool(processes=cpu_number)

    P = Parameters

    for arg in product(P.models, P.dimensons, P.learning_rates, P.ws, P.epochs):
        train_embedding(*arg)

    # pool.starmap(train_embedding, product(P.models, P.dimensons, P.learning_rates, P.ws, P.epochs))



