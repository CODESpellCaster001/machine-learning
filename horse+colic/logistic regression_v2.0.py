import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# 从文本文件加载数据集
data = np.loadtxt('aimi-cn-horseColicTraining.txt', delimiter='\t')  # 假设数据以制表符分隔

# 划分特征和标签
X = data[:, :-1]  # 特征
y = data[:, -1]   # 标签

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 从文本文件加载测试集
test_data = np.loadtxt('aimi-cn-horseColicTest.txt', delimiter='\t')  # 假设测试集以制表符分隔

# 划分特征和标签
X_test = test_data[:, :-1]  # 测试集特征
y_test = test_data[:, -1]   # 测试集标签

# 创建Logistic回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_pca, y)

# 使用训练好的模型进行预测
X_test_pca = pca.transform(X_test)
y_pred = model.predict(X_test_pca)

# 计算模型的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy * 100:.2f}%")

# 绘制数据集的散点图
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Data set after PCA dimensionality reduction')

# 绘制测试集的分类结果
plt.figure()
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Test set after PCA dimensionality reduction')

plt.show()
