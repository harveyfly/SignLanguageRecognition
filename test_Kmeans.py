### 用k-means算法处理手写数字
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

# 读取数据
np_data = np.load("data\SLR_dataset\processed\SLR_S0_E20_body_data.npy", allow_pickle=True)
np_label = np.load("data\SLR_dataset\processed\SLR_S0_E20_body_label.npy", allow_pickle=True)

# 种类数
n_clusters = np_data.shape[1]
data_len = np_data.shape[0]
np_data = np_data.reshape(data_len, -1)

kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(np_data)
#可视化10类中的中心点——最具有代表性的10个数字
# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
# plt.show()

#计算分类的准确率
from sklearn.metrics import accuracy_score
print(accuracy_score(np_label, clusters))
