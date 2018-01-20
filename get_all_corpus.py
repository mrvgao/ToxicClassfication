from itertools import chain
import pandas as pd
from utilities import format_string
import config

files = ['data/test.csv', 'data/train.csv', 'data/train_old.csv', 'data/test_old.csv']

contents = [pd.read_csv(f) for f in files]

with open(config.line_corpus, 'w') as f:
    for row in chain(*[df.iterrows() for df in contents]):
        try:
            sentence = format_string(str(row[1].comment_text))
            f.write(sentence + '\n')
        except TypeError:
            continue
