import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成随机数据
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# 初始化K-means模型
kmeans = KMeans(n_clusters=4, random_state=42)

# 训练模型
kmeans.fit(X)

# 获取簇中心和标签
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.8, label='data point')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='core of cluster')
plt.title('K-means result')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend()
plt.show()
