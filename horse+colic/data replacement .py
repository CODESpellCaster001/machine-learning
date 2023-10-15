# 定义输入文件路径和输出文件路径
input_file_path = 'horse-colic-clean.test'
output_file_path = 'horse-colic-clean-v1.0.test'

# 打开输入文件，读取内容
with open(input_file_path, 'r') as input_file:
    data = input_file.read()

# 用换行符分隔行
lines = data.split('\n')
output_lines = []

for line in lines:
    items = line.strip().split('\t')  # 用制表符分隔数据，并去掉首尾空格
    if len(items) >= 22:
        # 保留每行的前21个数据，再加上第23个数据
        output_line = '\t'.join(items[:21] + [items[22]])  # 拼接前21个数据和第23个数据
        output_lines.append(output_line)

# 将处理后的数据写入输出文件
with open(output_file_path, 'w') as output_file:
    output_file.write('\n'.join(output_lines))

print(f'数据已处理并保存到{output_file_path}')
