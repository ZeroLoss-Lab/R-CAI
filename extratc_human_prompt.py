import json
from typing import List, Dict

def extract_human_content(input_json_path: str, output_json_path: str) -> None:
    """
    从包含红队对话数据的JSON文件中提取human的内容，保存到新的JSON文件中。
    提取逻辑符合《红队数据集.pdf》中“红队Prompt为人类对话输入”的定义，仅保留诱导模型生成回复的人类文本。
    
    Args:
        input_json_path: 输入JSON文件的路径（需包含原始对话数据，如你提供的transcript格式）
        output_json_path: 输出JSON文件的路径（用于保存提取的human内容）
    """
    # 1. 读取输入JSON文件（支持单个JSON对象或JSON数组格式）
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            # 尝试解析为JSON数组（若输入是多个对话对象的集合）
            try:
                data: List[Dict] = json.load(f)
            except json.JSONDecodeError:
                # 若解析失败，尝试解析为单个JSON对象（单条对话）
                f.seek(0)  # 重置文件指针到开头
                data: List[Dict] = [json.load(f)]
    except FileNotFoundError:
        raise FileNotFoundError(f"输入文件不存在：{input_json_path}")
    except Exception as e:
        raise RuntimeError(f"读取输入文件时出错：{str(e)}")

    # 2. 提取每个对话中的human内容（按《红队数据集.pdf》Prompt格式筛选）
    extracted_human_data: List[Dict] = []
    for idx, dialog in enumerate(data, 1):
        # 检查对话数据是否包含transcript字段（你的数据中对话内容存储在该字段）
        if "transcript" not in dialog:
            print(f"警告：第{idx}条数据缺少'transcript'字段，已跳过")
            continue
        
        transcript: str = dialog["transcript"]
        # 按行分割transcript，提取以"Human:"开头的内容（红队Prompt的核心格式）
        human_lines: List[str] = []
        for line in transcript.strip().split("\n"):
            line_stripped = line.strip()
            if line_stripped.startswith("Human:"):
                # 提取"Human:"后的纯文本内容（去除前缀，保留诱导性输入）
                human_content = line_stripped[len("Human:"):].strip()
                if human_content:  # 跳过空内容
                    human_lines.append(human_content)
        
        # 若当前对话中存在有效human内容，整理为结构化数据（保留原始数据的关键标识，便于追溯）
        if human_lines:
            extracted_item = {
                "dialog_index": idx,  # 对话在原始数据中的序号
                "original_task_description": dialog.get("task_description", "未标注"),  # 原始任务描述（来自你的数据字段）
                "extracted_human_prompts": human_lines,  # 提取的所有human内容（红队Prompt列表）
                "original_harmlessness_score": dialog.get("min_harmlessness_score_transcript", "未标注")  # 原始有害性评分（便于筛选高诱导性Prompt）
            }
            extracted_human_data.append(extracted_item)

    # 3. 将提取的human内容保存到输出JSON文件
    if not extracted_human_data:
        raise ValueError("未从输入文件中提取到任何human内容，请检查输入数据格式是否符合要求（需包含'transcript'字段且存在'Human:'开头的行）")
    
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            # 缩进格式化输出，便于后续查看和使用
            json.dump(extracted_human_data, f, ensure_ascii=False, indent=4)
        print(f"提取完成！共提取{len(extracted_human_data)}条对话中的human内容，已保存到：{output_json_path}")
    except Exception as e:
        raise RuntimeError(f"保存输出文件时出错：{str(e)}")

# ------------------- 配置参数（请根据你的文件路径修改） -------------------
INPUT_JSON_PATH = "red_team_attempts.jsonl"    # 你的原始JSON文件路径（如包含transcript的文件）
OUTPUT_JSON_PATH = "extracted_human_prompts.json"  # 输出提取结果的JSON文件路径

# ------------------- 执行提取 -------------------
if __name__ == "__main__":
    try:
        extract_human_content(INPUT_JSON_PATH, OUTPUT_JSON_PATH)
    except Exception as e:
        print(f"程序执行失败：{str(e)}")