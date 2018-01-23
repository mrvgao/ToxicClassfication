import csv


def parse_file(src, tgt):
    with open(tgt, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'model', 'method', 'dimension', 'lr', 'ws', 'epoch', 't-p', 't-r', 't-f1', 'd-p', 'd-r', 'd-f1'])

        for line in open(src):
            dash_split = line.split('-')
            label = dash_split[0]
            model = dash_split[1].split('/')[-1]
            method = dash_split[2]
            dimenson = int(dash_split[3])
            lr = float(dash_split[4])
            ws = int(dash_split[5])
            epoch = int(dash_split[6].split('_')[0])
            t_p, t_r, t_f1 = float(dash_split[8]), float(dash_split[10]), float(dash_split[12].strip())
            d_p, d_r, d_f1 = float(dash_split[14]), float(dash_split[16]), float(dash_split[18].strip())
            writer.writerow([label, model, method, dimenson, lr, ws, epoch, t_p, t_r, t_f1, d_p, d_r, d_f1])
