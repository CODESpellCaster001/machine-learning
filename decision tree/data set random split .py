import random

# 读取原始文件
with open('german.csv', 'r') as file:
    data = file.readlines()

# 随机打乱数据
random.shuffle(data)

# 计算分割点
split_index = int(len(data) * 0.2)

# 分割数据
test_data = data[:split_index]
train_data = data[split_index:]

# 写入分割后的文件
with open('test_data.csv', 'w') as file:
    file.writelines(test_data)

with open('train_data.csv', 'w') as file:
    file.writelines(train_data)

print("数据已成功分割为 test_data.txt 和 train_data.txt 文件。")
