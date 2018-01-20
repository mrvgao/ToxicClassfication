def evaluation(clf, dev_file):
    def_text = [' '.join(line.split()[1:]) for line in open(dev_file, encoding='utf-8')]
    labels = clf.predict(def_text)
    print(labels)

