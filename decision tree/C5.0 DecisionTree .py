import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import warnings
from sklearn.tree import export_graphviz
import graphviz
import pydotplus




# 1. 准备数据
# 请替换为你自己的数据文件路径或数据获取方式
data = pd.read_csv("german.csv")

# 2. 拆分特征和标签
X = data.iloc[:, :-1]  # 选择所有列除了最后一列作为特征
y = data.iloc[:, -1]   # 选择最后一列作为标签

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 使用独热编码处理非数字特征
# 这里假设你的数据中有非数字特征，你需要指定它们的列名
categorical_features = ["Status of existing checking account", "Credit history","Purpose","Savings account/bonds","Present employment since","Personal status and sex","Other debtors / guarantors","Property","Other installment plans","Housing","Job","Telephone","foreign worker"]

# 创建一个转换器，用于独热编码非数字特征
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

# 创建一个列转换器，将独热编码应用于指定的列
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# 5. 创建C5.0决策树分类器
model = DecisionTreeClassifier()

# 6. 创建一个管道，将数据预处理和模型拟合结合起来
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])

# 7. 训练模型
clf.fit(X_train, y_train)

# 8. 预测测试数据
y_pred = clf.predict(X_test)

# 9. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
accuracy_percentage = accuracy * 100
print(f"Accuracy: {accuracy_percentage:.2f}%")

