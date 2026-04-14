# R-CAI Documentation Index

[English](INDEX.md) | [简体中文](INDEX_CN.md)

This page provides a quick navigation map for this repository.

## Start Here

- Project overview: [README.md](README.md)
- Chinese overview: [README_CN.md](README_CN.md)

## Script Workflow

1. `extratc_human_prompt.py`
2. `extract_first.py` (optional)
3. `harmful_data_generation.py`
4. `critique_refine.py` (or `critical.py` + `revise.py`)
5. `extract_revise4.py`
6. `random_extract.py`
7. `sft_data_process.py`
8. `add_plot.py` and `add_plot2.py`

## Supporting Files

- vLLM launch scripts: `llama_8b.sh`, `llama_70b.sh`
- Figures: `ablation_clamping.pdf`, `iteration_dynamics.pdf`
- Data/process artifacts:
  - `red_team_attempts.jsonl`
  - `extracted_human_prompts.json`
  - `sftdata.json`

## Notes

- Most scripts are configured through constants at the top of each file.
- Before running, update data paths and set endpoint/model environment variables (for example: `OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`, `VLLM_API_URL`).
- Use this code only for authorized defensive AI safety research.
