from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, Y = make_blobs(n_samples=200, n_features=3,centers=5, cluster_std=1.0,random_state=1)
# model2 = MyGMM(5, 100)
# model2.fit(X)
# result = model2.predict(X)
# plt.scatter(X[:,0],X[:,1],c=Y)
# plt.scatter(X[:,0],X[:,1],c=result)

clf = GaussianMixture(n_components=5)
clf.fit(X)
result = clf.predict(X)
plt.scatter(X[:,0],X[:,1], c=Y)
plt.show()
plt.scatter(X[:,0],X[:,1], c=result)
plt.show()