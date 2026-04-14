import json
from typing import List, Dict

def process_and_save_data(input_data: List[Dict], output_file: str) -> None:
    """
    处理输入数据（仅保留extracted_human_prompts的第一个元素）并保存为JSONL格式
    
    Args:
        input_data: 输入的字典列表数据
        output_file: 输出的JSONL文件路径
    """
    try:
        processed_count = 0  # 记录成功处理的条目数
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in input_data:
                if not isinstance(item, dict):
                    print(f"警告：跳过非字典类型的数据 - {item}")
                    continue
                
                # 处理extracted_human_prompts，只保留第一个元素
                if "extracted_human_prompts" in item:
                    prompts = item["extracted_human_prompts"]
                    # 确保是列表且不为空，否则保留原结构
                    if isinstance(prompts, list) and len(prompts) > 0:
                        item["extracted_human_prompts"] = [prompts[0]]  # 保留第一个元素（仍为列表格式）
                    else:
                        print(f"警告：条目 {item.get('dialog_index')} 的extracted_human_prompts格式异常，未处理")
                
                # 写入处理后的对象
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                processed_count += 1
        
        print(f"数据已成功写入 {output_file}，共处理 {processed_count} 条记录")
    
    except Exception as e:
        print(f"处理数据时出错：{str(e)}")

if __name__ == "__main__":
    
    # 从外部文件读取数据（取消注释并修改路径即可）
    with open('extracted_human_prompts.json', 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    
    # 处理并保存
    process_and_save_data(sample_data, 'output_data.jsonl')