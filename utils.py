from statistics import mode

def classify_time_series(ts, clf, cut):
    serie = []
    for w in range(0, ts.shape[0], cut):
        crop = ts[w:(w + cut)]
        try:
            serie.append(clf.predict(crop))
        except:
            pass
    return mode(serie)
