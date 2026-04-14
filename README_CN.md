# R-CAI

[English](README.md) | [简体中文](README_CN.md)

R-CAI 是一个用于 LLM 安全压力测试的 ACL 项目。  
当前仓库已按你的要求精简为四类核心代码：

- 初始回复生成
- Critique / Revision
- 基于 LLaMA-Factory 的 SFT / RM / PPO 训练
- 毒性 / 逻辑性 / 多样性评估

## 仓库结构

```text
.
├── harmful_data_generation.py     # 初始回复生成
├── critical.py                    # critique
├── revise.py                      # revision
├── critique_refine.py             # 多轮 critique+revision
├── prepare_lf_data.py             # 转换 LLaMA-Factory 数据集
├── train_sft_lf.sh                # SFT 训练脚本（LLaMA-Factory）
├── train_rm_lf.sh                 # RM 训练脚本（LLaMA-Factory）
├── train_ppo_lf.sh                # PPO 训练脚本（LLaMA-Factory）
├── evaluate_scores.py             # 毒性/逻辑性/多样性评估
├── data/examples/                 # 仅保留小样例
├── llama_8b.sh / llama_70b.sh     # 本地 vLLM 启动脚本
├── requirements.txt
└── README.md / README_CN.md
```

## 环境准备

```bash
pip install -r requirements.txt
```

可选环境变量：

```bash
export OPENAI_API_KEY="your_key_if_needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="llama3_8b"
export VLLM_API_URL="http://localhost:8000"
export VLLM_MODEL_NAME="llama3_8b"
```

## 主流程

1. 生成初始回复

```bash
python harmful_data_generation.py
```

2. 执行 critique 与 revision

```bash
python critical.py
python revise.py
# 或多轮:
python critique_refine.py
```

3. 生成 LLaMA-Factory 训练数据

```bash
python prepare_lf_data.py --input_jsonl refined4.jsonl --output_dir data
```

4. 使用 LLaMA-Factory 训练

```bash
bash train_sft_lf.sh
bash train_rm_lf.sh
bash train_ppo_lf.sh
```

5. 执行评估打分

```bash
python evaluate_scores.py \
  --input_jsonl refined4.jsonl \
  --output_jsonl evaluation_scores.jsonl \
  --summary_json evaluation_summary.json
```

## 数据策略

- 仓库中已移除完整数据集。
- 仅在 `data/examples/` 保留小样例文件。
- 如需完整实验，请替换为你有权限使用的完整数据。

## 说明

- `train_*_lf.sh` 依赖 `llamafactory-cli`。
- `prepare_lf_data.py` 会输出 `data/dataset_info.json` 和三个数据文件：
  - `lf_sft.json`
  - `lf_rm.json`
  - `lf_ppo_prompt.json`

## 安全声明

本仓库仅用于防御性 AI 安全研究。
