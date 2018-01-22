def precision(lables, predict):
    return sum([1 for l, p in zip(lables, predict) if l == p[0]]) / len(lables)


def recall(lables, predicate, right_label):
    right_num = sum([1 for l, p in zip(lables, predicate) if l == p[0] and l == right_label])
    samples_num = sum([1 for l in lables if l == right_label])
    return right_num / samples_num


def evaluation(clf, dev_file):
    labels, def_text = [], []

    for line in open(dev_file, encoding='utf-8'):
        label, text = line.split()[0], ' '.join(line.split()[1:])
        labels.append(label)
        def_text.append(text)

    labels_hat = clf.predict(def_text)

    p, r = precision(labels, labels_hat), recall(labels, labels_hat, '__label__1')

    f1 = 2 * (p * r) / (p + r)

    return p, r, f1
