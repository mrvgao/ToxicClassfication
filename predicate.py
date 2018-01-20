import fasttext
from grid_search import get_classifier_path
import pandas as pd
import csv


parameters = """threat	cbow	30	0.001	9	15
severe_toxic	cbow	60	0.001	9	15
identity_hate	cbow	100	0.01	6	10
obscene	cbow	60	0.001	9	5
insult	cbow	100	0.01	6	5
toxic	skip	100	0.01	9	15"""

best_model_parameters = {}

lines = parameters.split('\n')


for line in lines:
    words = line.split()
    method, dim, lr, ws, epoch = words[1], int(words[2]), float(words[3]), int(words[4]), int(words[5])
    clf_path = get_classifier_path(model=method, dim=dim, lr=lr, windows=ws, epoch=epoch)
    best_model_parameters[words[0]] = fasttext.load_model(clf_path + '.bin')


test_data = pd.read_csv('data/test.csv')

with open('test_result.csv', 'w') as f:
    writer = csv.writer(f)
    columns = "id, toxic, severe_toxic, obscene, threat, insult, identity_hate"
    columns = columns.split(', ')
    writer.writerow(columns)
    index = 0
    for row in test_data.iterrows():
        print('{}/{}'.format(index, len(test_data))); index += 1
        _id, text = row[1]['id'], row[1]['comment_text']

        new_row = [_id]
        for label in columns[1:]:
            result = best_model_parameters[label].predict_proba([text], k=1)
            prob = result[0]
            new_row.append(prob)

        print(new_row)
        writer.writerow(new_row)
