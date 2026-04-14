import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON line {line_no}: {path}")
    return records


def dump_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {len(data)} records -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare SFT/RM/PPO datasets for LLaMA-Factory from critique-revision outputs."
    )
    parser.add_argument("--input_jsonl", type=str, default="refined4.jsonl")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--prompt_field", type=str, default="original_human_input")
    parser.add_argument("--chosen_field", type=str, default="refined_response")
    parser.add_argument("--rejected_field", type=str, default="model_harmful_response")
    parser.add_argument("--status_field", type=str, default="refinement_status")
    parser.add_argument("--status_value", type=str, default="Success")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = load_jsonl(input_path)
    print(f"[INFO] Loaded {len(data)} raw samples from {input_path}")

    sft_data: List[Dict[str, Any]] = []
    rm_data: List[Dict[str, Any]] = []
    ppo_data: List[Dict[str, Any]] = []

    for item in data:
        if args.status_value:
            if str(item.get(args.status_field, "")).strip() != args.status_value:
                continue

        prompt = str(item.get(args.prompt_field, "")).strip()
        chosen = str(item.get(args.chosen_field, "")).strip()
        rejected = str(item.get(args.rejected_field, "")).strip()
        if not prompt or not chosen:
            continue

        sft_data.append(
            {
                "instruction": prompt,
                "input": "",
                "output": chosen,
            }
        )
        ppo_data.append({"prompt": prompt})

        if rejected:
            rm_data.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    dump_json(output_dir / "lf_sft.json", sft_data)
    dump_json(output_dir / "lf_rm.json", rm_data)
    dump_json(output_dir / "lf_ppo_prompt.json", ppo_data)

    dataset_info = {
        "r_cai_sft": {
            "file_name": "lf_sft.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        },
        "r_cai_rm": {
            "file_name": "lf_rm.json",
            "ranking": True,
            "columns": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
        "r_cai_ppo_prompt": {
            "file_name": "lf_ppo_prompt.json",
            "columns": {"prompt": "prompt"},
        },
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote dataset registry -> {output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
