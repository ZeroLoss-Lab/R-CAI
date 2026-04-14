# R-CAI: 反向宪法式 AI 的可控毒性数据生成框架

[English](README.md) | [简体中文](README_CN.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange)](#)

R-CAI 是一个面向 LLM 安全压力测试的 ACL 研究项目。  
与传统“单条越狱提示词搜索”不同，R-CAI 将红队任务重构为**结构化对抗数据合成**：

- 反向宪法（Constitution of Toxicity）
- 多轮 Critique-Revision 自举生成
- 带概率截断（Probability Clamping）的 RLAIF 稳定优化

## 最新进展

- 2026-04：公开首版代码与实验过程文件。

## 核心特性

- Reverse Constitutional AI：将无害对齐原则反转为可控对抗目标。
- 自举式多轮生成：在无人工逐条标注下构建结构化对抗样本。
- 概率截断优化：缓解奖励饱和与语义退化。
- 完整脚本链路：从 prompt 提取、迭代精炼到采样和可视化。

## 目录

1. [方法概览](#方法概览)
2. [仓库结构](#仓库结构)
3. [安装](#安装)
4. [快速开始](#快速开始)
5. [主要实验设置](#主要实验设置)
6. [结果概览](#结果概览)
7. [安全声明](#安全声明)
8. [引用](#引用)
9. [联系方式](#联系方式)

## 方法概览

R-CAI 采用两阶段流程：

1. **自举式数据合成**
   在四类毒性原则（法律伦理、社会偏见、行为后果、信任欺骗）引导下进行多轮 critique-revision。
2. **约束强化学习**
   使用 RLAIF 并引入概率截断（默认 `[0.4, 0.6]`），抑制 reward hacking 与语义坍塌。

## 仓库结构

```text
.
├── README.md / README_CN.md
├── INDEX.md / INDEX_CN.md
├── requirements.txt
├── llama_8b.sh / llama_70b.sh                  # 启动本地 vLLM 服务
├── extratc_human_prompt.py                      # 从红队原始对话提取 human prompt
├── extract_first.py                             # 每条对话仅保留第一个 prompt
├── harmful_data_generation.py                   # 生成初始有害回复
├── critical.py                                  # 单轮 critique
├── revise.py                                    # 单轮 refinement
├── critique_refine.py                           # 多轮 critique+refinement 主流程
├── critique_success.py                          # 筛选有效 critique
├── extract_revise4.py                           # 筛选有效 refined 样本
├── random_extract.py                            # 随机采样（如 8K）
├── sft_data_process.py                          # 转换为 SFT 训练格式
├── add_plot.py / add_plot2.py                   # 作图脚本
├── ablation_clamping.pdf / iteration_dynamics.pdf
└── red_team_attempts.jsonl / extracted_human_prompts.json / sftdata.json
```

## 安装

### 环境要求

- Python 3.10+
- 若本地跑大模型，需 CUDA 环境
- 可选：vLLM（用于 OpenAI 兼容接口服务）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 可选：启动本地 vLLM

```bash
bash llama_8b.sh
# 或
bash llama_70b.sh
```

## 快速开始

当前脚本的数据路径主要通过文件内常量配置，同时支持环境变量配置接口与模型参数。  
建议先配置：

```bash
export OPENAI_API_KEY="your_key_if_needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="llama3_8b"
export VLLM_API_URL="http://localhost:8000"
export VLLM_MODEL_NAME="llama3_8b"
```

1. 从红队对话中提取 prompt

```bash
python extratc_human_prompt.py
```

2. （可选）每条对话保留首个 prompt

```bash
python extract_first.py
```

3. 基于 OpenAI 兼容接口生成初始回复

```bash
python harmful_data_generation.py
```

4. 执行多轮 critique-revision

```bash
python critique_refine.py
```

5. 筛选并采样用于 SFT

```bash
python extract_revise4.py
python random_extract.py
python sft_data_process.py
```

6. 绘图

```bash
python add_plot.py
python add_plot2.py
```

## 主要实验设置

- Base policy / Reward model：**Llama-3-8B**
- Judge model：**Llama-3-70B**
- Prompt 数量：**30,000**
- Critique-Revision 轮数：**4**
- LoRA：rank **32**, alpha **64**
- 概率截断区间：**[0.4, 0.6]**

## 结果概览

根据论文草稿结果：

- 概率截断在不显著损失毒性强度的前提下，使连贯性提升约 **15%**。
- 相比无截断优化，多样性提升约 **42.6%**。
- R-CAI 在多维毒性目标上表现出更强、且更可控的对抗行为。

仓库中已包含主要图表文件：

- `ablation_clamping.pdf` / `ablation_clamping.png`
- `iteration_dynamics.pdf` / `iteration_dynamics.png`

## 安全声明

本仓库仅用于**防御性安全研究**。

- 禁止将有害内容用于真实攻击或传播。
- 仅限授权场景下的红队评测与鲁棒性分析。
- 请遵守当地法律、机构伦理规范与平台安全政策。

## 引用

如使用本仓库，请引用：

```bibtex
@article{fang2026rcai,
  title   = {Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF},
  author  = {Fang, Yuan and Luo, Yiming and Zhou, Aimin and Tan, Fei},
  year    = {2026},
  journal = {ACL (submission/manuscript)}
}
```

## 联系方式

- Fei Tan: `ftan@mail.ecnu.edu.cn`
