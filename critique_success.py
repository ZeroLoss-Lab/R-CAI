import json
from typing import Dict, Any

# --- 配置部分 ---
# 请根据您的实际文件路径进行调整
INPUT_PATH = "critique_response.jsonl"  # 您的批判阶段输出文件路径
OUTPUT_PATH = "successful_critique_for_refinement.jsonl"  # 筛选后用于精炼的新文件
# ------------------

def filter_critique_data(input_path: str, output_path: str):
    """
    读取 JSONL 文件，筛选出 'critique_status' 为 'Success' 
    且 'model_critique' 不是默认拒绝回复的样本，并保存到新文件。
    """
    successful_data_count = 0
    total_data_count = 0
    failed_data_count = 0
    
    # 常见的拒绝回复或失败标记 (统一为小写，进行稳健匹配)
    common_rejections = [
        "i can't fulfill this request.", 
        "i cannot fulfill this request.",
        "i'm sorry, i cannot provide assistance on that topic.",
        "i am unable to provide instructions or information on that topic.",
        "api_failed_after_",  # 匹配 API_FAILED_AFTER_3_RETRIES 等失败信息
        "runtime error:",
        "skipped: no response"
    ]

    print(f"开始从 {input_path} 读取数据并进行筛选...")

    try:
        # 逐行读取 JSONL 文件
        with open(input_path, "r", encoding="utf-8") as infile, \
             open(output_path, "w", encoding="utf-8") as outfile:
            
            for line in infile:
                total_data_count += 1
                if not line.strip():
                    continue
                
                try:
                    item: Dict[str, Any] = json.loads(line)
                    
                    critique = item.get("model_critique", "").strip().lower()
                    status = item.get("critique_status", "N/A").strip()
                    
                    # 1. 检查状态是否成功
                    is_success_status = (status == "Success")
                    
                    # 2. 检查批判内容是否为拒绝回复或失败标记
                    is_rejection_or_failure = False
                    if not critique or len(critique) < 10: # 长度小于10基本可认为是空或拒绝
                        is_rejection_or_failure = True
                    else:
                        for rejection in common_rejections:
                            if rejection in critique:
                                is_rejection_or_failure = True
                                break
                    
                    # 只有当状态为 Success 且批判内容不是拒绝回复时，才视为成功样本
                    if is_success_status and not is_rejection_or_failure:
                        # 确保写入时使用 UTF-8 编码，并且关闭 ensure_ascii 避免中文乱码
                        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                        successful_data_count += 1
                    else:
                        failed_data_count += 1
                        
                except json.JSONDecodeError:
                    print(f"警告: 发现格式错误的行，已跳过。行内容: {line.strip()[:100]}...")
                    failed_data_count += 1
                
        print("\n--- 筛选结果 ---")
        print(f"总数据条数: {total_data_count}")
        print(f"成功筛选并保存的数据条数: {successful_data_count}")
        print(f"被跳过/删除的数据条数: {failed_data_count}")
        print(f"成功的结果已保存至 {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_path}。请检查路径。")
    except Exception as e:
        print(f"处理过程中发生意外错误: {e}")

if __name__ == "__main__":
    filter_critique_data(INPUT_PATH, OUTPUT_PATH)