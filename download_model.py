# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Llama-3-8B-Instruct是指令微调（对话）版本，如果你需要预训练版本，模型名称会有不同
# model_id = "meta-llama/Meta-Llama-3-8B" 

# # 指定一个本地路径用于存放模型文件（可选，默认会存到缓存）
local_dir = "./llama3-8b-local" 

# tokenizer = AutoTokenizer.from_pretrained(model_id) #, local_dir=local_dir)
# model = AutoModelForCausalLM.from_pretrained(model_id) #, local_dir=local_dir)

# # 此时模型文件已下载/缓存到你的环境中。
# print("Llama-3-8B模型已下载/缓存完成。")

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct" 

# 1. 修改 AutoTokenizer 的调用
tokenizer = AutoTokenizer.from_pretrained(
    model_id, 
    local_dir=local_dir,
    # 添加 force_download=True 强制重新下载 Tokenizer 文件
    force_download=True 
) 

# 2. 修改 AutoModelForCausalLM 的调用
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    local_dir=local_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # 添加 force_download=True 强制重新下载模型权重文件
    force_download=True 
)

print("已使用 force_download=True 尝试重新下载模型和分词器。")