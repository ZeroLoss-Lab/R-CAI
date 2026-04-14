# R-CAI

[English](README.md) | [简体中文](README_CN.md)

R-CAI is an ACL project on automated adversarial data synthesis for LLM safety stress testing.
This repository now keeps only the core code you requested:

- initial response generation,
- critique and revision,
- LLaMA-Factory training (SFT / RM / PPO),
- evaluation for toxicity, coherence, and diversity.

## Repository Structure

```text
.
├── harmful_data_generation.py     # initial response generation
├── critical.py                    # critique
├── revise.py                      # revision
├── critique_refine.py             # multi-round critique+revision pipeline
├── prepare_lf_data.py             # convert outputs to LLaMA-Factory datasets
├── train_sft_lf.sh                # SFT training script (LLaMA-Factory)
├── train_rm_lf.sh                 # RM training script (LLaMA-Factory)
├── train_ppo_lf.sh                # PPO training script (LLaMA-Factory)
├── evaluate_scores.py             # toxicity/coherence/diversity evaluation
├── llama_8b.sh / llama_70b.sh     # local vLLM launch scripts
├── requirements.txt
└── README.md / README_CN.md
```

## Environment

```bash
pip install -r requirements.txt
```

Optional environment variables:

```bash
export OPENAI_API_KEY="your_key_if_needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="llama3_8b"
export VLLM_API_URL="http://localhost:8000"
export VLLM_MODEL_NAME="llama3_8b"
```

## Pipeline

1. Generate initial responses

```bash
python harmful_data_generation.py
```

2. Critique and revise

```bash
python critical.py
python revise.py
# or multi-round:
python critique_refine.py
```

3. Prepare LLaMA-Factory datasets

```bash
python prepare_lf_data.py --input_jsonl refined4.jsonl --output_dir data
```

4. Train with LLaMA-Factory

```bash
bash train_sft_lf.sh
bash train_rm_lf.sh
bash train_ppo_lf.sh
```

5. Evaluate scores

```bash
python evaluate_scores.py \
  --input_jsonl refined4.jsonl \
  --output_jsonl evaluation_scores.jsonl \
  --summary_json evaluation_summary.json
```

## Notes

- `train_*_lf.sh` expects `llamafactory-cli` in your environment.
- `prepare_lf_data.py` writes `data/dataset_info.json` and three datasets:
  - `lf_sft.json`
  - `lf_rm.json`
  - `lf_ppo_prompt.json`

## Safety

This repository is for defensive AI safety research only.
