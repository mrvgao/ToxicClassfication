from preprocessing_training_corpus import write_one_train_corpus
from preprocessing_training_corpus import labels
from grid_search import predicate
from paramters import Parameters
from functools import reduce
from tqdm import tqdm
from itertools import product
from train_models import train
from multiprocessing import Pool
from get_w2v_embedding import train_embedding
import config
import os
from parse_recodings import parse_file


def get_length(*iterators):
    return reduce(lambda x, y: x * y, map(len, iterators), 1)


def merge_result(already_notes, file):
    return already_notes + [file]


train_new_embedding = True

P = Parameters

if train_new_embedding:
    for ii, l in enumerate(labels):
        if ii == 0: first = True
        else: first = False
        print('label {}'.format(l))
        write_one_train_corpus(l, first_write=first)

    for ii, arg in enumerate(product(P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)):
        first = ii == 0
        train_embedding(*arg, first=first)


length = get_length(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)
for ii, args in tqdm(enumerate(product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)), total=length):
    print('batch {}/{}'.format(ii, length))
    first = ii == 0
    train(*args, first_train=first)

cpu_num = 60

pool = Pool(processes=cpu_num)

model_perfermance_recoding_file_name = 'cust_data/path_train_recoding.txt'
pathes = [config.clf_root + m for m in os.listdir(config.clf_root)]

results = pool.map(predicate, pathes)

result = reduce(merge_result, results, [])

with open(model_perfermance_recoding_file_name, 'w') as f:
    for r in result:
        f.write(r)


parse_file(model_perfermance_recoding_file_name, 'cust_data/different_model_performance.csv')
