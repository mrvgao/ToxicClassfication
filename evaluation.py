def evaluation(clf, dev_file):
    def_text = [' '.join(line.split()[1:]) for line in open(dev_file)]
    labels = clf.predict(def_text)
    print(labels)

