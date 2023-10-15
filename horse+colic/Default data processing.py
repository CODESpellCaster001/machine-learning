import os

# 定义文件路径
file_path = 'horse-colic.test'
output_file_path = 'horse-colic-clean.test'

# 打开文件，读取内容
with open(file_path, 'r') as file:
    data = file.read()

# 替换所有的'?'为0
data = data.replace('?', '0')

# 用制表符分隔数据
data = data.replace(' ', '\t')

# 用换行符分隔行
data = data.replace('\n', '\n')

# 用空格分隔数字并删除超过999的数据
lines = data.split('\n')
output_lines = []

for line in lines:
    items = line.split('\t')
    filtered_items = []
    for item in items:
        try:
            num = float(item)
            if num <= 999:
                filtered_items.append(str(num))
        except ValueError:
            filtered_items.append(item)
    output_lines.append('\t'.join(filtered_items))

# 将处理后的数据写入新文件
with open(output_file_path, 'w') as output_file:
    output_file.write('\n'.join(output_lines))

print(f'文件已处理并保存到{output_file_path}')
