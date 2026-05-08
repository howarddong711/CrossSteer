#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch


def _ensure_torchvision_nms_schema():
    global _TORCHVISION_SCHEMA_LIB
    try:
        _TORCHVISION_SCHEMA_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_SCHEMA_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


_ensure_torchvision_nms_schema()

from advwave.attack import qwen_jailbreak_gen
from advwave.data import ensure_audio_files, load_test_dataset
from advwave.defense import (
    apply_steering_to_model,
    inference_with_audio,
    load_steering_vector,
    restore_steering_layer,
)
from advwave.judge import judge_outcome
from advwave.paths import DATA_DIR, OUTPUT_DIR
from run_advwave_defense import (
    ensure_qwen_speaker_map,
    patch_qwen2_5_omni_load_speakers,
    patch_qwen_omni_forward,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_MODEL = os.environ.get("QWEN25_OMNI_MODEL", "")
DEFAULT_VECTOR_ROOT = os.path.join(REPO_ROOT, "vectors", "qwen25_omni")


def _parse_layers(layers_arg: List[str], vector_root: str) -> List[int]:
    if len(layers_arg) == 1 and layers_arg[0].lower() == "all":
        layers = []
        for name in os.listdir(vector_root):
            if name.startswith("layer") and name[5:].isdigit():
                layers.append(int(name[5:]))
        if not layers:
            raise FileNotFoundError(f"No layer* directories found under {vector_root}")
        return sorted(layers)
    return [int(x) for x in layers_arg]


def _resolve_vector_jobs(args) -> List[Tuple[int, str]]:
    jobs = []
    missing = []
    for layer in _parse_layers(args.layers, args.vector_root):
        vector_path = os.path.join(
            args.vector_root,
            f"layer{layer}",
            f"vec_ep{args.vector_epoch}_layer{layer}.pt",
        )
        if os.path.exists(vector_path):
            jobs.append((layer, vector_path))
        else:
            missing.append(vector_path)
    if missing:
        raise FileNotFoundError("Missing Qwen2.5-Omni vectors:\n" + "\n".join(missing))
    return jobs


def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl_by_key(path: str, key_name: str) -> Dict[int, dict]:
    rows: Dict[int, dict] = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key_name in row:
                rows[int(row[key_name])] = row
    return rows


def _parse_api_config(path: str) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    text = open(path, "r", encoding="utf-8").read().strip().replace("，", ",")
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                continue
            data[key.strip()] = value.strip().strip("'\"")
    key = data.get("key") or data.get("api_key") or data.get("OPENAI_API_KEY")
    url = data.get("url") or data.get("base_url") or data.get("api_base")
    model = data.get("model") or data.get("MODEL")
    env = {}
    if key:
        env["OPENAI_API_KEY"] = str(key)
        env["SILICONFLOW_API_KEY"] = str(key)
    if url:
        url = str(url).rstrip("/")
        if not url.endswith("/v1"):
            url += "/v1"
        env["SILICONFLOW_BASE_URL"] = url
    if model:
        env["SILICONFLOW_MODEL"] = str(model)
    return env


def _sync_judge_env() -> None:
    import advwave.judge as judge_module

    judge_module.SILICONFLOW_API_KEY = (
        os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    judge_module.SILICONFLOW_BASE_URL = os.getenv(
        "SILICONFLOW_BASE_URL", judge_module.SILICONFLOW_BASE_URL
    )
    judge_module.SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", judge_module.SILICONFLOW_MODEL)


def load_attack_rows_from_results(paths: List[str], run_dir: str) -> List[dict]:
    rows_by_id: Dict[int, dict] = {}
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attack results JSONL not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                idx = int(raw["id"])
                adv_audio_path = raw.get("adv_audio_path")
                if not adv_audio_path or not os.path.exists(adv_audio_path):
                    raise FileNotFoundError(f"Missing adv audio for id={idx}: {adv_audio_path}")
                row = {
                    "id": idx,
                    "source": raw.get("source", "attack_results"),
                    "source_index": raw.get("source_index", idx),
                    "question": raw.get("question", ""),
                    "matching": raw.get("matching", ""),
                    "not_matching": raw.get("not_matching", ""),
                    "attack": raw.get("attack", "audio_ours"),
                    "audio_path": raw.get("audio_path", ""),
                    "adv_audio_path": adv_audio_path,
                    "attack_response": raw.get("attack_response", raw.get("response", "")),
                    "attack_label": raw.get("attack_label", raw.get("label", "")),
                    "attack_jailbreak": raw.get("attack_jailbreak", raw.get("is_jailbreak", False)),
                    "attack_invalid": raw.get("attack_invalid", raw.get("is_invalid", False)),
                    "attack_judge_error": raw.get("attack_judge_error", raw.get("is_judge_error", False)),
                    "attack_judge_reason": raw.get("attack_judge_reason", raw.get("reason", "")),
                }
                rows_by_id[idx] = row

    attack_rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    cache_dir = os.path.join(run_dir, "attack_cache")
    os.makedirs(cache_dir, exist_ok=True)
    results_path = os.path.join(cache_dir, "attack_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for row in attack_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _save_json(os.path.join(cache_dir, "attack_summary.json"), _summarize_attack(attack_rows))
    return attack_rows


def _label_text(outcome: dict) -> str:
    if outcome["is_jailbreak"]:
        return "Jailbreak"
    if outcome.get("is_judge_error", False):
        return "JudgeError"
    if outcome["is_invalid"]:
        return "Invalid"
    return "Refusal"


def _summarize_attack(rows: List[dict]) -> dict:
    total = len(rows)
    attack_success = sum(1 for r in rows if r.get("attack_jailbreak", False))
    attack_invalid = sum(1 for r in rows if r.get("attack_invalid", False))
    attack_judge_error = sum(1 for r in rows if r.get("attack_judge_error", False))
    return {
        "total": total,
        "attack_success": attack_success,
        "attack_invalid": attack_invalid,
        "attack_judge_error": attack_judge_error,
        "attack_asr": (attack_success / total * 100) if total else 0,
        "attack_invalid_rate": (attack_invalid / total * 100) if total else 0,
        "attack_judge_error_rate": (attack_judge_error / total * 100) if total else 0,
    }


def _summarize_defense(rows: List[dict], multipliers: List[float]) -> dict:
    stats = {}
    for m in multipliers:
        m_key = str(float(m))
        m_rows = [r for r in rows if str(r.get("multiplier")) == m_key]
        denom = len(m_rows)
        counts = {
            "jailbreak": sum(1 for r in m_rows if r.get("label") == "jailbreak"),
            "refusal": sum(1 for r in m_rows if r.get("label") == "refusal"),
            "invalid": sum(1 for r in m_rows if r.get("label") == "invalid"),
            "judge_error": sum(1 for r in m_rows if r.get("label") == "judge_error"),
        }
        stats[m_key] = {
            "total": denom,
            "jailbreak_count": counts["jailbreak"],
            "jailbreak_rate": (counts["jailbreak"] / denom * 100) if denom else 0,
            "refusal_count": counts["refusal"],
            "refusal_rate": (counts["refusal"] / denom * 100) if denom else 0,
            "invalid_count": counts["invalid"],
            "invalid_rate": (counts["invalid"] / denom * 100) if denom else 0,
            "judge_error_count": counts["judge_error"],
            "judge_error_rate": (counts["judge_error"] / denom * 100) if denom else 0,
            "defense_success_rate": (counts["refusal"] / denom * 100) if denom else 0,
        }
    return stats


def build_attack_cache(model, processor, dataset: List[dict], audio_paths: List[str], run_dir: str, args) -> List[dict]:
    cache_dir = os.path.join(run_dir, "attack_cache")
    wav_dir = os.path.join(cache_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    results_path = os.path.join(cache_dir, "attack_results.jsonl")
    existing = _load_jsonl_by_key(results_path, "id")

    total = len(dataset)
    for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
        idx = int(sample["id"])
        if idx in existing and existing[idx].get("adv_audio_path") and os.path.exists(existing[idx]["adv_audio_path"]):
            print(f"[Attack cache] skip existing {local_idx + 1}/{total} id={idx}")
            continue

        prompt = sample["question"]
        adv_path = os.path.join(wav_dir, f"{idx}.wav")
        print(f"\n[Attack cache {local_idx + 1}/{total}] Optimizing id={idx}")
        response_text, record = qwen_jailbreak_gen(
            prompt,
            [raw_audio],
            processor,
            model,
            audio_save_path=adv_path,
            num_token_suffix=args.suffix_length,
            num_epochs=args.num_epochs,
            lr=args.lr,
            advnoise_control=False,
            control_obj="",
            model_judge=None,
            targets=[],
        )
        if isinstance(record, dict) and "jailbreaked_audio" in record:
            adv_path = record["jailbreaked_audio"]

        attack_response = response_text if isinstance(response_text, str) else ""
        attack_outcome = judge_outcome(prompt, attack_response)
        row = {
            "id": idx,
            "source": sample["source"],
            "source_index": sample["source_index"],
            "question": sample["question"],
            "matching": sample["matching"],
            "not_matching": sample["not_matching"],
            "attack": "audio_ours",
            "audio_path": raw_audio,
            "adv_audio_path": adv_path,
            "attack_response": attack_response,
            "attack_label": attack_outcome["label"],
            "attack_jailbreak": attack_outcome["is_jailbreak"],
            "attack_invalid": attack_outcome["is_invalid"],
            "attack_judge_error": attack_outcome.get("is_judge_error", False),
            "attack_judge_reason": attack_outcome["reason"],
        }
        _append_jsonl(results_path, row)
        existing[idx] = row
        print(f"  [Attack] Result: {_label_text(attack_outcome)}")
        torch.cuda.empty_cache()

    attack_rows = [existing[k] for k in sorted(existing)]
    _save_json(os.path.join(cache_dir, "attack_summary.json"), _summarize_attack(attack_rows))
    return attack_rows


def run_defense_sweep(model, processor, attack_rows: List[dict], vector_jobs: List[Tuple[int, str]], run_dir: str, args) -> None:
    defense_root = os.path.join(run_dir, "defense")
    for layer, vector_path in vector_jobs:
        print(f"\n=== Defense sweep layer {layer} ===")
        layer_dir = os.path.join(defense_root, f"layer{layer}")
        results_path = os.path.join(layer_dir, "results.jsonl")
        existing_rows = []
        existing_keys = set()
        if os.path.exists(results_path):
            with open(results_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    existing_rows.append(row)
                    existing_keys.add((int(row["id"]), str(row["multiplier"])))

        steering_vector = load_steering_vector(vector_path, args.device)
        wrapped_layer = apply_steering_to_model(model, steering_vector, layer)
        original_layer = wrapped_layer.block

        try:
            for m in args.multipliers:
                multiplier = float(m)
                wrapped_layer.set_multiplier(multiplier)
                m_key = str(multiplier)
                print(f"\n[Layer {layer}] multiplier={m_key}")
                for local_idx, row in enumerate(attack_rows):
                    idx = int(row["id"])
                    key = (idx, m_key)
                    if key in existing_keys:
                        continue
                    print(f"  [Defense {local_idx + 1}/{len(attack_rows)} id={idx} m={m_key}] ", end="", flush=True)
                    response = inference_with_audio(model, processor, row["adv_audio_path"], args.device, prompt="")
                    outcome = judge_outcome(row["question"], response)
                    defense_row = {
                        "id": idx,
                        "layer": layer,
                        "vector_path": vector_path,
                        "multiplier": multiplier,
                        "source": row["source"],
                        "source_index": row["source_index"],
                        "question": row["question"],
                        "matching": row["matching"],
                        "not_matching": row["not_matching"],
                        "attack": row["attack"],
                        "audio_path": row["audio_path"],
                        "adv_audio_path": row["adv_audio_path"],
                        "attack_label": row["attack_label"],
                        "attack_jailbreak": row["attack_jailbreak"],
                        "attack_invalid": row["attack_invalid"],
                        "attack_judge_error": row["attack_judge_error"],
                        "response": response,
                        "label": outcome["label"],
                        "is_jailbreak": outcome["is_jailbreak"],
                        "is_invalid": outcome["is_invalid"],
                        "is_judge_error": outcome.get("is_judge_error", False),
                        "defense_success": outcome["label"] == "refusal",
                        "reason": outcome["reason"],
                    }
                    _append_jsonl(results_path, defense_row)
                    existing_rows.append(defense_row)
                    existing_keys.add(key)
                    print(_label_text(outcome))
                    torch.cuda.empty_cache()
        finally:
            restore_steering_layer(model, layer, original_layer)
            del steering_vector
            torch.cuda.empty_cache()

        summary = {
            "layer": layer,
            "vector_path": vector_path,
            "dataset_size": len(attack_rows),
            "multipliers": args.multipliers,
            "multiplier_stats": _summarize_defense(existing_rows, args.multipliers),
        }
        _save_json(os.path.join(layer_dir, "summary.json"), summary)


def load_qwen_runtime(args):
    patch_qwen2_5_omni_load_speakers()
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    print(f"Loading processor and model: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()
    ensure_qwen_speaker_map(model)
    patch_qwen_omni_forward(model)
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached Qwen2.5-Omni AdvWave attack + all-layer defense sweep")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--vector_root", type=str, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--vector_epoch", type=int, default=400)
    parser.add_argument("--layers", nargs="+", default=["13"])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=200)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--suffix_length", type=int, default=16000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate_tts", action="store_true")
    parser.add_argument("--tts_max_retries", type=int, default=5)
    parser.add_argument("--tts_sleep", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=os.path.join(OUTPUT_DIR, "qwenomni_cached"))
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_csv", type=str, default=os.path.join(DATA_DIR, "advbench_infer_200.csv"))
    parser.add_argument("--test_infer_csv", type=str, default="")
    parser.add_argument("--attack_results_jsonl", nargs="*", default=None)
    parser.add_argument("--api_file", type=str, default=os.environ.get("CROSSSTEER_API_FILE", "API.md"))
    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name is required.")

    for key, value in _parse_api_config(args.api_file).items():
        os.environ[key] = value
    _sync_judge_env()

    vector_jobs = _resolve_vector_jobs(args)
    print(f"Resolved {len(vector_jobs)} Qwen2.5-Omni vector job(s).")

    dataset = []
    audio_paths = []
    if not args.attack_results_jsonl:
        dataset, seed = load_test_dataset(
            test_csv=args.test_csv,
            test_infer_csv=args.test_infer_csv,
            seed=args.seed,
        )
        args.seed = seed
        dataset = dataset[args.start:args.end]
        audio_paths = ensure_audio_files(
            dataset,
            generate_tts=args.generate_tts,
            tts_max_retries=args.tts_max_retries,
            tts_sleep=args.tts_sleep,
        )

    run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    _save_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "model_name": args.model_name,
            "vector_root": args.vector_root,
            "vector_epoch": args.vector_epoch,
            "layers": [layer for layer, _ in vector_jobs],
            "multipliers": args.multipliers,
            "start": args.start,
            "end": args.end,
            "num_epochs": args.num_epochs,
            "suffix_length": args.suffix_length,
            "lr": args.lr,
            "seed": args.seed,
            "dataset_size": len(dataset),
            "test_csv": args.test_csv,
            "test_infer_csv": args.test_infer_csv,
            "attack_results_jsonl": args.attack_results_jsonl,
        },
    )

    model, processor = load_qwen_runtime(args)
    if args.attack_results_jsonl:
        attack_rows = load_attack_rows_from_results(args.attack_results_jsonl, run_dir)
    else:
        attack_rows = build_attack_cache(model, processor, dataset, audio_paths, run_dir, args)
    run_defense_sweep(model, processor, attack_rows, vector_jobs, run_dir, args)
    print(f"Done. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
