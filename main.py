from preprocessing_training_corpus import write_one_train_corpus
from preprocessing_training_corpus import labels
from grid_search import predicate
from paramters import Parameters
from functools import reduce
from tqdm import tqdm
from itertools import product
from train_models import train
from multiprocessing import Pool
import config
import os


def get_length(*iterators):
    return reduce(lambda x, y: x * y, map(len, iterators), 1)


def merge_result(already_notes, file):
    return already_notes + [file]


for l in labels:
    print('label {}'.format(l))
    write_one_train_corpus(l)


P = Parameters
length = get_length(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)
for ii, args in tqdm(enumerate(product(labels, P.models, P.dimensons, P.learning_rates, P.ws, P.epochs)), total=length):
    print('batch {}/{}'.format(ii, length))
    train(*args)

cpu_num = 60

pool = Pool(processes=cpu_num)

file = 'path_train_recoding.txt'
pathes = [config.clf_root + m for m in os.listdir(config.clf_root)]

results = pool.map(predicate, pathes)

result = reduce(merge_result, results, [])

with open(file, 'w') as f:
    for r in result:
        f.write(r)
