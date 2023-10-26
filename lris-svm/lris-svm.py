import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 从CSV文件中读取数据，假设文件名为 'iris_dataset.csv'
data = pd.read_csv('iris.csv')

# 提取特征和标签
X = data.iloc[:, 1:5].values
y = data.iloc[:, 5].values

# 将类别标签编码为数字
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 创建SVM分类器
classifier = SVC(kernel='linear', C=1.0, random_state=42)

# 拟合模型
classifier.fit(X_train, y_train)

# 预测
y_pred = classifier.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 可视化分类结果
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred)
plt.xlabel('length')
plt.ylabel('wigth')
plt.title('SVM result')
plt.show()
