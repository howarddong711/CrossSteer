#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

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
from advwave.paths import DATA_DIR
from run_advwave_defense import ensure_qwen_speaker_map, patch_qwen2_5_omni_load_speakers


DEFAULT_MODEL = os.environ.get("QWEN25_OMNI_MODEL", "")


def parse_api_config(path: str) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8").strip().replace("，", ",")
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
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


def loss_summary(losses):
    if not losses:
        return {}
    first = float(losses[0])
    last = float(losses[-1])
    best = float(min(losses))
    return {
        "steps": len(losses),
        "first": first,
        "last": last,
        "best": best,
        "delta": first - last,
        "relative_drop": ((first - last) / first) if first else 0.0,
        "last_10_mean": sum(float(x) for x in losses[-10:]) / min(10, len(losses)),
    }


def resolve_audio_paths(dataset_by_id: Dict[int, Dict], ids: List[int], audio_dir: str = "") -> List[str]:
    if not audio_dir:
        return ensure_audio_files([dataset_by_id[i] for i in ids], generate_tts=False)
    base = Path(audio_dir)
    paths = [str(base / f"{i}.wav") for i in ids]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing audio files: {missing[:10]}")
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset_csv", default=os.path.join(DATA_DIR, "advbench_infer_200.csv"))
    parser.add_argument("--audio_dir", default="")
    parser.add_argument("--ids", nargs="+", type=int, default=None)
    parser.add_argument("--id_start", type=int, default=None)
    parser.add_argument("--id_end", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--suffix_length", type=int, default=32000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=2124729650)
    parser.add_argument("--api_file", default=os.environ.get("CROSSSTEER_API_FILE", "API.md"))
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name is required.")

    if args.ids is None:
        if args.id_start is None or args.id_end is None:
            args.ids = [0, 1]
        else:
            args.ids = list(range(args.id_start, args.id_end + 1))
    args.ids = [int(i) for i in args.ids]

    for key, value in parse_api_config(args.api_file).items():
        os.environ[key] = value

    import advwave.judge as judge_module
    from advwave.judge import judge_outcome
    judge_module.SILICONFLOW_API_KEY = (
        os.getenv("SILICONFLOW_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    judge_module.SILICONFLOW_BASE_URL = os.getenv(
        "SILICONFLOW_BASE_URL", judge_module.SILICONFLOW_BASE_URL
    )
    judge_module.SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", judge_module.SILICONFLOW_MODEL)

    patch_qwen2_5_omni_load_speakers()

    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading QwenOmni model: {args.model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    ensure_qwen_speaker_map(model)
    model.eval()

    dataset, seed = load_test_dataset(args.dataset_csv, "", seed=args.seed, shuffle=args.shuffle)
    print(f"Dataset seed: {seed}", flush=True)
    dataset_by_id = {int(row["id"]): row for row in dataset}
    missing_ids = [i for i in args.ids if i not in dataset_by_id]
    if missing_ids:
        raise ValueError(f"Requested ids not found in dataset: {missing_ids}")
    audio_paths = resolve_audio_paths(dataset_by_id, args.ids, args.audio_dir)

    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for sample_id, raw_audio in zip(args.ids, audio_paths):
            sample = dataset_by_id[sample_id]
            adv_path = wav_dir / f"{sample_id}.wav"
            print(
                f"\n[Probe] id={sample_id} epochs={args.num_epochs} suffix={args.suffix_length} lr={args.lr}",
                flush=True,
            )
            response, record = qwen_jailbreak_gen(
                sample["question"],
                [raw_audio],
                processor,
                model,
                audio_save_path=str(adv_path),
                num_token_suffix=args.suffix_length,
                num_epochs=args.num_epochs,
                lr=args.lr,
                advnoise_control=False,
                control_obj="",
                model_judge=None,
                targets=[],
            )
            losses = record.get("losses", []) if isinstance(record, dict) else []
            outcome = judge_outcome(sample["question"], response or "")
            row = {
                "id": sample_id,
                "question": sample["question"],
                "audio_path": raw_audio,
                "adv_audio_path": str(adv_path),
                "num_epochs": args.num_epochs,
                "suffix_length": args.suffix_length,
                "lr": args.lr,
                "response": response,
                "label": outcome["label"],
                "is_jailbreak": outcome["is_jailbreak"],
                "is_invalid": outcome["is_invalid"],
                "is_judge_error": outcome.get("is_judge_error", False),
                "reason": outcome["reason"],
                "loss_summary": loss_summary(losses),
                "losses": losses,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[Probe] id={sample_id} label={row['label']} loss={row['loss_summary']}",
                flush=True,
            )
            torch.cuda.empty_cache()

    print(f"Done. Results: {results_path}", flush=True)


if __name__ == "__main__":
    main()
