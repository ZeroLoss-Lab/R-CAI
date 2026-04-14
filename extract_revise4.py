import json
import os

def filter_jsonl(input_filename, output_filename, key, required_value):
    """
    读取 JSONL 文件，根据指定的键值对进行筛选，并将符合条件的行写入新的 JSONL 文件。

    :param input_filename: 输入的 .jsonl 文件名
    :param output_filename: 输出的 .jsonl 文件名
    :param key: 要筛选的键 (例如: 'refinement_status')
    :param required_value: 键的必需值 (例如: 'Success')
    """
    successful_count = 0
    total_count = 0
    
    # 检查输入文件是否存在
    if not os.path.exists(input_filename):
        print(f"错误：找不到输入文件 '{input_filename}'。请检查文件名是否正确。")
        return

    print(f"开始处理文件：{input_filename}")
    print(f"筛选条件：'{key}' == '{required_value}'")

    with open(input_filename, 'r', encoding='utf-8') as infile, \
         open(output_filename, 'w', encoding='utf-8') as outfile:
        
        for line_number, line in enumerate(infile, 1):
            total_count += 1
            try:
                # 1. 解析每一行为一个 JSON 对象 (字典)
                record = json.loads(line)
                
                # 2. 检查筛选条件
                if record.get(key) == required_value:
                    # 3. 如果符合条件，将该记录写入输出文件
                    # 注意：在写入 JSONL 文件时，每行必须是一个 JSON 对象，且以换行符结束
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    successful_count += 1
                    
            except json.JSONDecodeError:
                print(f"警告：第 {line_number} 行不是有效的 JSON 格式，已跳过。")
            except Exception as e:
                print(f"警告：处理第 {line_number} 行时发生错误：{e}")

    print("-" * 30)
    print(f"处理完成。")
    print(f"总记录数：{total_count}")
    print(f"符合条件的记录数：{successful_count}")
    print(f"结果已写入文件：{output_filename}")


# --- 使用示例 ---
# 请将 'input.jsonl' 替换为您实际的文件名
input_file = 'refined4.jsonl' 
output_file = 'final_refined.jsonl'
filter_key = 'refinement_status'
filter_value = 'Success'

filter_jsonl(input_file, output_file, filter_key, filter_value)