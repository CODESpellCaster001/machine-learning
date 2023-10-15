def convert_labels(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    converted_lines = []

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) > 0:
            # 获取原始标签
            original_label = parts[-1]
            # 转换标签为整数类型
            try:
                converted_label = int(float(original_label))
                # 重新构建行并将其添加到转换后的行列表中
                converted_line = '\t'.join(parts[:-1] + [str(converted_label)])
                converted_lines.append(converted_line)
            except ValueError:
                # 如果无法转换为整数，则跳过该行
                print(f"无法将标签转换为整数: {original_label}")
    
    with open(output_file, 'w') as output:
        output.write('\n'.join(converted_lines))

if __name__ == '__main__':
    input_file = 'horse-colic-clean-v1.0.test'  # 输入文件
    output_file = 'horse-colic-clean-v1.0-converted.test'  # 输出文件

    convert_labels(input_file, output_file)

