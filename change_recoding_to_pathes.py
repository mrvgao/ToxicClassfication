import csv


with open('results_recoding.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['label', 'method', 'dimension', 'lr', 'ws', 'epoch', 'p', 'r', 'f1'])

    for line in open('path_train_recoding.txt'):
        dash_split = line.split('-')
        print(list(enumerate(dash_split)))
        label = dash_split[0]
        method = dash_split[1].split('/')[-1]
        dimenson = int(dash_split[2])
        lr = float(dash_split[3])
        ws = int(dash_split[4])
        epoch = int(dash_split[5].split('_')[0])
        p, r, f1 = float(dash_split[7]), float(dash_split[9]), float(dash_split[11].strip())
        writer.writerow([label, method, dimenson, lr, ws, epoch, p, r, f1])

