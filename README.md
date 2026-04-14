# R-CAI: Reverse Constitutional AI for Controllable Toxic Data Generation

[English](README.md) | [简体中文](README_CN.md)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research-orange)](#)

R-CAI is an ACL research project on **safety stress testing for large language models (LLMs)**.
Instead of searching for single jailbreak prompts, R-CAI reframes red teaming as **structured adversarial data synthesis** with:

- a reverse constitution ("constitution of toxicity"),
- iterative critique-revision data generation,
- probability-clamped RLAIF to reduce reward hacking.

## News

- 2026-04: Initial public code and experiment artifacts released.

## Key Features

- Reverse Constitutional AI: inverts harmless constitutions into controllable adversarial objectives.
- Self-Bootstrapped Critique-Revision: iterative synthetic data construction without human annotation in the loop.
- Probability-Clamped RLAIF: improves coherence while maintaining strong adversarial intensity.
- Reproducible script pipeline: prompt extraction, iterative refinement, sampling, and plotting.

## Table of Contents

1. [Method Overview](#method-overview)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Main Experimental Settings](#main-experimental-settings)
6. [Results Snapshot](#results-snapshot)
7. [Safety Notice](#safety-notice)
8. [Citation](#citation)
9. [Contact](#contact)

## Method Overview

R-CAI follows a two-stage design:

1. **Self-Bootstrapped Synthesis**
   Build adversarial samples through multi-round critique-revision under four toxicity principles:
   legal/ethical harm, social bias, behavioral consequences, and trust/deception.
2. **Constrained Reinforcement Learning**
   Train with RLAIF and apply probability clamping (default: `[0.4, 0.6]`) to mitigate reward saturation and semantic collapse.

## Repository Structure

```text
.
├── README.md / README_CN.md
├── INDEX.md / INDEX_CN.md
├── requirements.txt
├── llama_8b.sh / llama_70b.sh                  # vLLM server launch scripts
├── extratc_human_prompt.py                      # extract human prompts from raw red-team transcripts
├── extract_first.py                             # keep first prompt per dialog
├── harmful_data_generation.py                   # generate initial harmful responses
├── critical.py                                  # single-round critique generation
├── revise.py                                    # single-round refinement generation
├── critique_refine.py                           # multi-stage critique+refinement pipeline
├── critique_success.py                          # filter successful critique samples
├── extract_revise4.py                           # filter successful refined samples
├── random_extract.py                            # random sampling (e.g., 8K)
├── sft_data_process.py                          # convert to SFT training format
├── add_plot.py / add_plot2.py                   # figure scripts
├── ablation_clamping.pdf / iteration_dynamics.pdf
└── red_team_attempts.jsonl / extracted_human_prompts.json / sftdata.json
```

## Installation

### Requirements

- Python 3.10+
- CUDA environment if running large local models
- Optional: vLLM for OpenAI-compatible local serving

### Install

```bash
pip install -r requirements.txt
```

### Optional: start local vLLM service

```bash
bash llama_8b.sh
# or
bash llama_70b.sh
```

## Quick Start

The scripts use in-file constants for data paths and support environment variables for endpoint/model settings.
Recommended environment setup:

```bash
export OPENAI_API_KEY="your_key_if_needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL_NAME="llama3_8b"
export VLLM_API_URL="http://localhost:8000"
export VLLM_MODEL_NAME="llama3_8b"
```

1. Extract prompts from raw red-team transcripts

```bash
python extratc_human_prompt.py
```

2. (Optional) Keep first prompt per dialog

```bash
python extract_first.py
```

3. Generate initial responses (OpenAI-compatible endpoint)

```bash
python harmful_data_generation.py
```

4. Run iterative critique-revision

```bash
python critique_refine.py
```

5. Filter and sample for SFT

```bash
python extract_revise4.py
python random_extract.py
python sft_data_process.py
```

6. Plot figures

```bash
python add_plot.py
python add_plot2.py
```

## Main Experimental Settings

- Base policy / reward model: **Llama-3-8B**
- Judge model: **Llama-3-70B**
- Prompt pool: **30,000** harmful-inducing prompts
- Critique-revision rounds: **4**
- LoRA: rank **32**, alpha **64**
- Probability clamp bounds: **[0.4, 0.6]**

## Results Snapshot

From the paper draft:

- Probability clamping improves coherence by about **15%** while preserving adversarial strength.
- Diversity improves by about **42.6%** over unclamped optimization.
- R-CAI maintains stronger prompt-aligned toxic behavior than baseline/SFT-only variants.

Figures in this repo:

- `ablation_clamping.pdf` / `ablation_clamping.png`
- `iteration_dynamics.pdf` / `iteration_dynamics.png`

## Safety Notice

This repository is for **defensive AI safety research** only.

- Do not deploy generated harmful content in real-world systems.
- Use only in authorized red-teaming and robustness evaluation settings.
- Follow local laws, institutional review policies, and platform safety policies.

## Citation

If you use this repository, please cite:

```bibtex
@article{fang2026rcai,
  title   = {Reverse Constitutional AI: A Framework for Controllable Toxic Data Generation via Probability-Clamped RLAIF},
  author  = {Fang, Yuan and Luo, Yiming and Zhou, Aimin and Tan, Fei},
  year    = {2026},
  journal = {ACL (submission/manuscript)}
}
```

## Contact

- Fei Tan: `ftan@mail.ecnu.edu.cn`
