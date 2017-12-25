import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


def preprocess():
    dataset = pd.read_csv("./dataset/winequality-red-dot.csv")
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(dataset)
    a = pd.DataFrame(x_scaled, columns=dataset.columns)
    a['quality'] = dataset['quality']
    # del a['fixed acidity']
    # del a['free sulfur dioxide']
    # del a['density']
    train, test = train_test_split(a, test_size=0.1, random_state=42)
    return train, test


train_data, test_data = preprocess()

train_y = train_data['quality']
del train_data['quality']
test_y = test_data['quality']
del test_data['quality']

clf = LogisticRegression(multi_class='multinomial', penalty='l2', solver='sag', tol=0.01)
clf.fit(train_data.values, train_y)
score = clf.score(test_data.values, test_y)
print("lr score:", score)

knn = KNeighborsClassifier()
knn.fit(train_data.values, train_y)
knn_score = knn.score(test_data.values, test_y)
print("knn score:", knn_score)

kmean = KMeans(n_clusters=6)
kmean.fit(train_data.values, train_y)
kmean_score = kmean.score(test_data, test_y)
print("kmean score:", kmean_score)

#

