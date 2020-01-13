import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.pipeline
import sklearn.compose
from sklearn.decomposition import PCA
import sklearn.svm
from sklearn.metrics import classification_report

from dataset import HARDatasetCrops
from transforms import FourierTransform
from utils import classify_time_series
from dataset import HARDataset


if __name__ == '__main__':
    dataset = HARDatasetCrops('motionsense-dataset/train', 256, 50, 50,
                          metadata_file='motionsense-dataset/data_subjects_info.csv')

    X = np.array([np.hstack([np.linalg.norm(sample[:, -3:], axis=1)] + metadata)
                  for sample, _, metadata in dataset])

    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(list(dataset.CLASSES.keys()))
    y = label_encoder.transform([cls for _, cls, _ in dataset])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    print('Number of training examples:', y_train.shape[0])
    print('Number of testing examples:', y_test.shape[0])

    pca = sklearn.pipeline.make_pipeline(
        FourierTransform(),
        PCA(n_components=32)
    )
    features = sklearn.compose.ColumnTransformer([
        ('fft_pca', pca, slice(0, -4)),
        ('metadata', 'passthrough', slice(-4, X.shape[1]))
    ])

    clf = sklearn.pipeline.make_pipeline(
        features,
        sklearn.svm.SVC()
    )
    clf.fit(X_train, y_train)

    dataset_test = HARDataset('motionsense-dataset/test', metadata_file='motionsense-dataset/data_subjects_info.csv')

    y_pred = [classify_time_series(np.linalg.norm(x[:, -3:], axis=1), clf, 256, meta)
              for x, _, meta in dataset_test]
    y_true = label_encoder.transform([cls for _, cls, _ in dataset_test])

    report = classification_report(y_true, y_pred)
    print(report)
