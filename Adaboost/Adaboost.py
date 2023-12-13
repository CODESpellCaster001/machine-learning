import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        m = len(X)
        weights = np.ones(m) / m
        for _ in range(self.n_estimators):
            # 创建弱分类器（决策树）
            model = DecisionTreeClassifier(max_depth=1)
            # 使用加权样本拟合分类器
            model.fit(X, y, sample_weight=weights)
            # 预测
            predictions = model.predict(X)
            # 计算误差
            error = np.sum(weights * (predictions != y))
            # 计算分类器权重
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
            # 更新样本权重
            weights = weights * np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)
            # 保存分类器和权重
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        # 对每个分类器进行预测
        predictions = np.array([model.predict(X) for model in self.models])
        # 计算加权投票
        weighted_votes = np.dot(self.alphas, predictions)
        # 返回最终预测
        return np.sign(weighted_votes)

# 生成一个简单的二分类数据集
X, y = make_classification(n_samples=999, n_features=23, n_informative=10, n_clusters_per_class=2, random_state=45)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Adaboost分类器
adaboost = AdaBoostClassifier(n_estimators=50)

# 训练Adaboost分类器
adaboost.fit(X_train, y_train)

# 预测测试集
y_pred = adaboost.predict(X_test)

# 评估分类器性能
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

def plot_decision_boundary(X, y, model, title):
    h = .02  # 步长
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

# 使用两个特征的数据集
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Adaboost分类器
adaboost = AdaBoostClassifier(n_estimators=50)

# 训练Adaboost分类器
adaboost.fit(X_train, y_train)

# 预测测试集
y_pred = adaboost.predict(X_test)

# 可视化训练集的决策边界
plot_decision_boundary(X_train, y_train, adaboost, "Adaboost - Training Set")

# 可视化测试集的决策边界
plot_decision_boundary(X_test, y_test, adaboost, "Adaboost - Test Set")

plt.show()