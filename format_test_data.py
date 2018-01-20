from utilities import format_string
import csv
import pandas as pd

with open('data/new_test.csv', 'w', encoding='utf-8') as f:
    writer = csv.writer()
    writer.writerow(['id', 'comment_text'])
    for row in pd.read_csv('data/test.csv').iterrows():
        _id, content = row[1]['id'], row[1]['comment_text']
        writer.writerow([_id, format_string(str(content))])
