import json
import random
import os
from typing import List, Dict, Any

# --- 配置部分 ---
INPUT_FILE_PATH = "final_refined.jsonl"  # 你的输入文件路径
OUTPUT_FILE_PATH = "sft_data_8k.jsonl" # 抽样后的输出文件路径
SAMPLE_SIZE = 8000                   # 要抽取的样本数量
# =================

def sample_jsonl_data(input_path: str, output_path: str, sample_size: int):
    """
    从 JSONL 文件中读取所有数据，随机抽取指定数量的样本，并保存到新的 JSONL 文件。
    """
    
    # 1. 读取所有数据
    print(f"正在读取文件: {input_path}...")
    data_list: List[Dict[str, Any]] = []
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data_list.append(item)
                    except json.JSONDecodeError:
                        print(f"警告：第 {line_num + 1} 行解析失败，跳过。")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_path}。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    total_count = len(data_list)
    print(f"文件读取完成。总共找到 {total_count} 条数据。")

    if total_count == 0:
        print("错误：文件中没有有效数据。")
        return
        
    # 2. 随机抽样
    if total_count <= sample_size:
        print(f"总数据量 ({total_count}) 小于或等于所需样本量 ({sample_size})。将输出所有数据。")
        sampled_data = data_list
    else:
        print(f"正在随机抽取 {sample_size} 条数据...")
        # 使用 random.sample 进行无放回抽样，高效且随机
        sampled_data = random.sample(data_list, sample_size)
        print("抽样完成。")

    # 3. 写入新的 JSONL 文件
    print(f"正在将抽样结果写入文件: {output_path}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in sampled_data:
                # 写入 JSONL 格式 (每行一个 JSON 对象)
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"写入成功！最终文件 {output_path} 包含 {len(sampled_data)} 条数据。")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")


if __name__ == "__main__":
    # 设置随机种子以保证结果可复现 (可选)
    # random.seed(42) 
    
    sample_jsonl_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH, SAMPLE_SIZE)