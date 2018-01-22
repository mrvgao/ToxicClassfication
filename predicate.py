import fasttext
from grid_search import get_classifier_path
import pandas as pd
import csv


parameters = """identity_hate	cbow	30	0.001	9	15
insult	skip	100	0.001	3	5
obscene	cbow	100	0.001	3	10
severe_toxic	cbow	60	0.001	9	15
threat	cbow	30	0.001	9	10
toxic	skip	100	0.01	9	5"""

best_model_parameters = {}

lines = parameters.split('\n')


for line in lines:
    words = line.split()
    label, method, dim, lr, ws, epoch = words[0], words[1], int(words[2]), float(words[3]), int(words[4]), int(words[5])
    clf_path = get_classifier_path(label=label, method=method, dim=dim, lr=lr, windows=ws, epoch=epoch)
    best_model_parameters[words[0]] = fasttext.load_model(clf_path + '.bin')


def get_label_prob(result):
    labels = '__label__0', '__label__1'
    for r in result:
        if r[0] == labels[1]: return r[1]


with open('test_result.csv', 'w') as f:
    writer = csv.writer(f)
    columns = "id, toxic, severe_toxic, obscene, threat, insult, identity_hate"
    columns = columns.split(', ')
    writer.writerow(columns)
    index = 0
    test_data = pd.read_csv('data/test.csv')
    for row in test_data.iterrows():
        print('{}/{}'.format(index, len(test_data))); index += 1
        _id, text = row[1]['id'], row[1]['comment_text']

        new_row = [_id]
        for label in columns[1:]:
            result = best_model_parameters[label].predict_proba([str(text)], k=2)
            prob = result[0]
            p = get_label_prob(prob)
            new_row.append(p)

        print(new_row)
        writer.writerow(new_row)
