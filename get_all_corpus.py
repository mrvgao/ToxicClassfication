from itertools import chain
import pandas as pd
from utilities import format_string
import config

test_content = pd.read_csv('data/test.csv')
original_content = pd.read_csv('data/train.csv')

with open(config.line_corpus, 'w') as f:
    for row in chain(*[df.iterrows() for df in [test_content, original_content]]):
        try:
            sentence = format_string(str(row[1].comment_text))
            f.write(sentence + '\n')
        except TypeError:
            continue
