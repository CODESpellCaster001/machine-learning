import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 生成100个二维随机点
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # 简单的分类规则，根据点的位置分类

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化KNN分类器
knn = KNeighborsClassifier(n_neighbors=3)

# 训练模型
knn.fit(X_train, y_train)

# 预测测试集
y_pred = knn.predict(X_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f'准确率: {accuracy}')

# 可视化训练集和测试集
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='text set', alpha=0.7)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='text set real lable')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='o', label='text set prodict label')
plt.title('KNN result')
plt.xlabel('feature1')
plt.ylabel('feature2')
plt.legend()
plt.show()
