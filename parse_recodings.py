import csv


with open('results_recoding.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'model', 'method', 'dimension', 'lr', 'ws', 'epoch', 'p', 'r', 'f1'])

    for line in open('cust_data/path_train_recoding.txt'):
        dash_split = line.split('-')
        print(list(enumerate(dash_split)))
        label = dash_split[0]
        model = dash_split[1].split('/')[-1]
        method = dash_split[2]
        dimenson = int(dash_split[3])
        lr = float(dash_split[4])
        ws = int(dash_split[5])
        epoch = int(dash_split[6].split('_')[0])
        p, r, f1 = float(dash_split[8]), float(dash_split[10]), float(dash_split[12].strip())
        writer.writerow([label, model, method, dimenson, lr, ws, epoch, p, r, f1])
