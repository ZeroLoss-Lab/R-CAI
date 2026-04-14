# R-CAI 文档导航

[English](INDEX.md) | [简体中文](INDEX_CN.md)

本页用于快速定位仓库内容。

## 建议先读

- 项目总览（英文）：[README.md](README.md)
- 项目总览（中文）：[README_CN.md](README_CN.md)

## 脚本流程导航

1. `extratc_human_prompt.py`
2. `extract_first.py`（可选）
3. `harmful_data_generation.py`
4. `critique_refine.py`（或 `critical.py` + `revise.py`）
5. `extract_revise4.py`
6. `random_extract.py`
7. `sft_data_process.py`
8. `add_plot.py` 与 `add_plot2.py`

## 辅助文件

- vLLM 启动脚本：`llama_8b.sh`、`llama_70b.sh`
- 图表文件：`ablation_clamping.pdf`、`iteration_dynamics.pdf`
- 数据与过程文件：
  - `red_team_attempts.jsonl`
  - `extracted_human_prompts.json`
  - `sftdata.json`

## 使用说明

- 大部分脚本通过文件顶部常量配置运行参数。
- 运行前请先修改数据路径，并设置接口/模型环境变量（如 `OPENAI_BASE_URL`、`OPENAI_MODEL_NAME`、`VLLM_API_URL`）。
- 本仓库仅用于授权场景下的防御性 AI 安全研究。
