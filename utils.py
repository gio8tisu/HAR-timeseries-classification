from statistics import mode

import numpy as np


def classify_time_series(ts, clf, cut, metadata=None):
    serie = []
    for w in range(0, ts.shape[0], cut):
        crop = ts[w:(w + cut)]
        if metadata is not None:
            crop = np.hstack([crop, metadata])
        serie.append(clf.predict(crop))
    return mode(serie)
