import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# get data
dataset = load_digits()
dataset.keys()
dataset.data[0]
# reshape data for first digit as 8x8 matrix
dataset.data[0].reshape(8,8)
# plot first digit
plt.matshow(dataset.data[0].reshape(8,8))

# get unqiue values in target
np.unique(dataset.target)

# create df with data
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# describe df, get stats for each col
df.describe()

# create X and y
X = df
y = dataset.target

# scale data
ss = StandardScaler()
X_scaled = ss.fit_transform(X)
X_scaled

# split train and test dfs, use X_scaled
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=.2)

# fit logistic regression
logr = LogisticRegression()
logr.fit(X_train, y_train)
logr.score(X_test, y_test)

# Principal component analysis
# set n components to transform into
pca_2f = PCA(n_components=2)
X_pca_2f = pca_2f.fit_transform(X)
X_pca_2f.shape
pca_2f.explained_variance_ratio_()

# PCA 2. Retain 95% of useful features
pca_095f = PCA(0.95)
X_pca_095f = pca_095f.fit_transform(X)
X_pca_095f.shape
pca_095f.explained_variance_ratio_()

# split train and test dfs, use X_pca_095f
X_train, X_test, y_train, y_test = train_test_split(X_pca_095f, y, test_size=.2)

# fit logistic regression
logr = LogisticRegression()
logr.fit(X_train, y_train)
logr.score(X_test, y_test)

