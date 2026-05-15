#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from typing import List

import torch

from advwave.data import build_audio_paths, ensure_audio_files, load_test_dataset
from advwave.judge import judge_outcome
from advwave.llamaomni2 import (
    apply_steering_to_model,
    generate_universal_adversarial_audio,
    inference_with_audio,
    llamaomni_eval_gen,
    llamaomni_jailbreak_gen,
    load_llama_omni2_model,
    load_llama_omni2_tokenizer,
    load_steering_vector,
    resolve_transformer_layers,
    save_audio,
)
from advwave.paths import OUTPUT_DIR
from advwave.tts import prompt2audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_VECTOR_ROOT = os.path.join(REPO_ROOT, "vectors", "llamaomni2")

def _vector_path(vector_root: str, layer: int, epoch: int) -> str:
    return os.path.join(vector_root, f"layer{layer}", f"vec_ep{epoch}_layer{layer}.pt")

def _vector_tag(path: str) -> str:
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def _ensure_output_dir(base_output: str, vector_tag: str, attack: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_output, vector_tag, f"{attack}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def _normalize_text(text: str) -> str:
    return " ".join((text or "").replace("\r\n", "\n").replace("\r", "\n").split())

def _atomic_rewrite_jsonl(path: str, rows: List[dict]) -> None:
    tmp_path = path + ".tmp"
    _save_jsonl(tmp_path, rows)
    os.replace(tmp_path, path)

def _cache_key(row: dict) -> tuple:
    return ("question", _normalize_text(row.get("question", "")))

def _dedup_dataset_by_prompt(dataset: List[dict]) -> List[dict]:
    deduped = []
    seen = set()
    for sample in dataset:
        key = _cache_key(sample)
        if key in seen:
            continue
        item = dict(sample)
        item["id"] = len(deduped)
        deduped.append(item)
        seen.add(key)
    return deduped

def _load_dataset_csv(path: str, dedup_prompts: bool = False) -> List[dict]:
    dataset = []
    source = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dataset.append(
                {
                    "id": idx,
                    "question": row.get("question", ""),
                    "matching": row.get("matching", ""),
                    "not_matching": row.get("not_matching", ""),
                    "source": source,
                    "source_index": idx,
                }
            )
    if dedup_prompts:
        dataset = _dedup_dataset_by_prompt(dataset)
    return dataset

def _valid_cached_attack(row: dict) -> bool:
    adv_path = row.get("adv_audio_path")
    return bool(adv_path) and os.path.exists(str(adv_path)) and not row.get("skipped", False)

def _load_valid_attack_cache(attack_cache_dir: str | None) -> List[dict]:
    if not attack_cache_dir:
        return []
    rows = _load_jsonl(os.path.join(attack_cache_dir, "results.jsonl"))
    valid = []
    seen = set()
    for row in rows:
        key = _cache_key(row)
        if key in seen or not _valid_cached_attack(row):
            continue
        valid.append(row)
        seen.add(key)
    return valid

def _apply_cached_raw_audio_paths(dataset: List[dict], attack_cache_dir: str | None) -> None:
    cached_by_key = {
        _cache_key(row): row
        for row in _load_valid_attack_cache(attack_cache_dir)
    }
    for sample in dataset:
        cached = cached_by_key.get(_cache_key(sample))
        audio_path = cached.get("audio_path") if cached else None
        if audio_path and os.path.exists(str(audio_path)):
            sample["audio_path"] = audio_path

def _apply_existing_tts_audio_paths_by_prompt(dataset: List[dict], seed: int) -> None:
    existing_dataset, _ = load_test_dataset(seed=seed)
    audio_dir = build_audio_paths()
    audio_by_prompt = {}
    for sample in existing_dataset:
        audio_path = os.path.join(audio_dir, f"{sample['id']}.wav")
        if os.path.exists(audio_path):
            audio_by_prompt.setdefault(_cache_key(sample), audio_path)

    applied = 0
    for sample in dataset:
        if sample.get("audio_path") and os.path.exists(str(sample["audio_path"])):
            continue
        audio_path = audio_by_prompt.get(_cache_key(sample))
        if audio_path:
            sample["audio_path"] = audio_path
            applied += 1
    print(f"Reused existing TTS audio by prompt for {applied}/{len(dataset)} sample(s).")

def _stable_audio_path(sample: dict) -> str:
    audio_dir = os.path.join(build_audio_paths(), "stable_by_source")
    os.makedirs(audio_dir, exist_ok=True)
    source = str(sample.get("source", "unknown")).replace(os.sep, "_")
    source_index = int(sample.get("source_index", sample.get("id", -1)))
    return os.path.join(audio_dir, f"{source}_{source_index}.wav")

def _ensure_audio_files_stable(
    dataset: List[dict],
    generate_tts: bool = True,
    tts_max_retries: int = 5,
    tts_sleep: float = 0.5,
) -> List[str]:
    audio_paths = []
    for sample in dataset:
        audio_path = sample.get("audio_path") or _stable_audio_path(sample)
        if not os.path.exists(audio_path) and generate_tts:
            prompt2audio(
                sample["question"],
                audio_path,
                max_retries=tts_max_retries,
                throttle_seconds=tts_sleep,
            )
        audio_paths.append(audio_path)
    return audio_paths

def _prepare_attack_dataset(dataset: List[dict], args) -> List[dict]:
    target_count = len(dataset)
    cached_rows = _load_valid_attack_cache(args.attack_cache_dir)
    if not cached_rows:
        return dataset

    prepared = []
    used = set()
    for cached in cached_rows:
        if len(prepared) >= target_count:
            break
        sample = {
            "id": cached.get("id", len(prepared)),
            "source": cached.get("source", ""),
            "source_index": cached.get("source_index", -1),
            "question": cached.get("question", ""),
            "matching": cached.get("matching", ""),
            "not_matching": cached.get("not_matching", ""),
            "audio_path": cached.get("audio_path"),
        }
        prepared.append(sample)
        used.add(_cache_key(sample))

    for sample in dataset:
        if len(prepared) >= target_count:
            break
        key = _cache_key(sample)
        if key in used:
            continue
        item = dict(sample)
        item["id"] = len(prepared)
        item["audio_path"] = _stable_audio_path(item)
        prepared.append(item)
        used.add(key)

    print(f"Loaded {len(cached_rows)} cached attack(s); prepared {len(prepared)}/{target_count} sample(s).")
    return prepared

def _print_outcome(outcome: dict) -> None:
    if outcome["is_jailbreak"]:
        print("Jailbreak")
    elif outcome.get("is_judge_error", False):
        print("JudgeError")
    elif outcome["is_invalid"]:
        print("Invalid")
    else:
        print("Refusal")

def _write_summary(output_dir: str, results: List[dict], multipliers: List[float], args, extra_config=None):
    extra_config = extra_config or {}
    summary = {
        "config": {
            "model": args.model_name,
            "attack": "audio_ours",
            "multipliers": multipliers,
            "num_epochs": args.num_epochs,
            "suffix_length": args.suffix_length,
            "universal_size": args.universal_size,
            "lr": args.lr,
            "seed": args.seed,
            "dataset_size": len(results),
            "max_new_tokens": args.max_new_tokens,
            **extra_config,
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "attack_stats": {},
        "multiplier_stats": {},
    }

    total = len(results)
    attack_success = sum(1 for r in results if r.get("attack_jailbreak", False))
    attack_invalid = sum(1 for r in results if r.get("attack_invalid", False))
    attack_judge_error = sum(1 for r in results if r.get("attack_judge_error", False))
    summary["attack_stats"] = {
        "total": total,
        "attack_success": attack_success,
        "attack_invalid": attack_invalid,
        "attack_judge_error": attack_judge_error,
        "attack_asr": (attack_success / total * 100) if total else 0,
        "attack_invalid_rate": (attack_invalid / total * 100) if total else 0,
        "attack_judge_error_rate": (attack_judge_error / total * 100) if total else 0,
    }

    for m in multipliers:
        m_key = str(m)
        cnt_jailbreak = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "jailbreak"
        )
        cnt_refusal = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "refusal"
        )
        cnt_invalid = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "invalid"
        )
        cnt_judge_error = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "judge_error"
        )

        summary["multiplier_stats"][m_key] = {
            "jailbreak_count": cnt_jailbreak,
            "jailbreak_rate": (cnt_jailbreak / total * 100) if total else 0,
            "refusal_count": cnt_refusal,
            "refusal_rate": (cnt_refusal / total * 100) if total else 0,
            "invalid_count": cnt_invalid,
            "invalid_rate": (cnt_invalid / total * 100) if total else 0,
            "judge_error_count": cnt_judge_error,
            "judge_error_rate": (cnt_judge_error / total * 100) if total else 0,
            "defense_success_rate": (cnt_refusal / total * 100) if total else 0,
        }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary

def generate_audio_ours_attack_cache(
    dataset: List[dict],
    audio_paths: List[str],
    tokenizer,
    model,
    output_dir: str,
    args,
) -> List[dict]:
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.jsonl")
    existing = _load_jsonl(results_path)
    existing_by_key = {}
    for row in _load_valid_attack_cache(args.seed_attack_cache_dir):
        existing_by_key.setdefault(_cache_key(row), row)
    for row in existing:
        if _valid_cached_attack(row):
            existing_by_key[_cache_key(row)] = row
    if len(existing_by_key) == len(dataset):
        print(f"Reusing complete attack cache: {output_dir}")
        results = []
        for sample, raw_audio in zip(dataset, audio_paths):
            idx = sample["id"]
            adv_path = os.path.join(output_dir, f"{idx}.wav")
            cached = dict(existing_by_key[_cache_key(sample)])
            cached_adv_path = cached.get("adv_audio_path")
            if cached_adv_path and os.path.abspath(str(cached_adv_path)) != os.path.abspath(adv_path):
                if not os.path.exists(adv_path):
                    shutil.copy2(str(cached_adv_path), adv_path)
                cached["reused_from_adv_audio_path"] = cached_adv_path
                cached["adv_audio_path"] = adv_path
            cached.update(
                {
                    "id": idx,
                    "source": sample["source"],
                    "source_index": sample["source_index"],
                    "question": sample["question"],
                    "matching": sample["matching"],
                    "not_matching": sample["not_matching"],
                    "audio_path": raw_audio,
                    "attack": "audio_ours",
                    "defense_results": {},
                }
            )
            results.append(cached)
        _atomic_rewrite_jsonl(results_path, results)
        _write_summary(output_dir, results, [], args, {"stage": "attack_cache"})
        return results

    results = []
    total_samples = len(dataset)

    for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
        idx = sample["id"]
        prompt = sample["question"]
        adv_path = os.path.join(output_dir, f"{idx}.wav")
        cached = existing_by_key.get(_cache_key(sample))
        if cached is not None:
            print(f"\n[Attack {local_idx + 1}/{total_samples}] Reusing cached sample {idx}: {cached['adv_audio_path']}")
            result = dict(cached)
            cached_adv_path = result.get("adv_audio_path")
            if cached_adv_path and os.path.abspath(str(cached_adv_path)) != os.path.abspath(adv_path):
                if not os.path.exists(adv_path):
                    shutil.copy2(str(cached_adv_path), adv_path)
                result["reused_from_adv_audio_path"] = cached_adv_path
                result["adv_audio_path"] = adv_path
            result.update(
                {
                    "id": idx,
                    "source": sample["source"],
                    "source_index": sample["source_index"],
                    "question": sample["question"],
                    "matching": sample["matching"],
                    "not_matching": sample["not_matching"],
                    "attack": "audio_ours",
                    "audio_path": raw_audio,
                    "defense_results": {},
                }
            )
            results.append(result)
            _atomic_rewrite_jsonl(results_path, results)
            continue

        print(f"\n[Attack {local_idx + 1}/{total_samples}] Optimizing sample {idx}")

        response_text, record = llamaomni_jailbreak_gen(
            prompt,
            [raw_audio],
            tokenizer,
            model,
            audio_save_path=adv_path,
            num_token_suffix=args.suffix_length,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device,
        )
        if isinstance(record, dict) and "jailbreaked_audio" in record:
            adv_path = record["jailbreaked_audio"]

        if adv_path is None or not os.path.exists(adv_path):
            print(f"Skipping id={idx} - adv_audio not generated")
            result = {
                "id": idx,
                "source": sample["source"],
                "source_index": sample["source_index"],
                "question": sample["question"],
                "matching": sample["matching"],
                "not_matching": sample["not_matching"],
                "attack": "audio_ours",
                "audio_path": raw_audio,
                "adv_audio_path": adv_path,
                "skipped": True,
                "skip_reason": "adv_audio not generated",
                "defense_results": {},
            }
        else:
            attack_response = response_text if isinstance(response_text, str) else llamaomni_eval_gen(
                [adv_path],
                tokenizer,
                model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            attack_outcome = judge_outcome(prompt, attack_response)
            result = {
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
                "defense_results": {},
            }
            print("  [Attack] Result: ", end="")
            _print_outcome(attack_outcome)

        results.append(result)
        _atomic_rewrite_jsonl(results_path, results)

    _write_summary(output_dir, results, [], args, {"stage": "attack_cache"})
    return results

def run_cached_defense_for_layer(
    layer: int,
    attack_cache_dir: str,
    attack_results: List[dict],
    tokenizer,
    model,
    set_multiplier,
    output_dir: str,
    args,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    results_path = os.path.join(output_dir, "results.jsonl")
    if os.path.exists(results_path):
        os.remove(results_path)

    total_samples = len(attack_results)
    for local_idx, cached in enumerate(attack_results):
        idx = cached["id"]
        prompt = cached["question"]
        adv_path = cached.get("adv_audio_path")
        print(f"\n[Layer {layer} Defense {local_idx + 1}/{total_samples}] sample {idx}")
        result = dict(cached)
        result["attack_cache_dir"] = attack_cache_dir
        result["defense_layer"] = layer
        result["defense_results"] = {}

        if cached.get("skipped") or not adv_path or not os.path.exists(str(adv_path)):
            result["skipped_defense"] = True
            result["skip_defense_reason"] = "missing cached adversarial audio"
        else:
            for m in args.multipliers:
                set_multiplier(m)
                print(f"  [Defense m={m}] Generating response...", end=" ", flush=True)
                defense_response = inference_with_audio(
                    model,
                    tokenizer,
                    str(adv_path),
                    args.device,
                    prompt="",
                    max_new_tokens=args.max_new_tokens,
                )
                defense_outcome = judge_outcome(prompt, defense_response)
                result["defense_results"][str(m)] = {
                    "response": defense_response,
                    "label": defense_outcome["label"],
                    "is_jailbreak": defense_outcome["is_jailbreak"],
                    "is_invalid": defense_outcome["is_invalid"],
                    "is_judge_error": defense_outcome.get("is_judge_error", False),
                    "defense_success": defense_outcome["label"] == "refusal",
                    "reason": defense_outcome["reason"],
                }
                _print_outcome(defense_outcome)

        results.append(result)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    _write_summary(
        output_dir,
        results,
        args.multipliers,
        args,
        {"stage": "cached_defense", "layer": layer, "attack_cache_dir": attack_cache_dir},
    )
    print(f"Layer {layer} cached-defense results saved to {output_dir}")

def run_for_attack(
    attack: str,
    dataset: List[dict],
    audio_paths: List[str],
    tokenizer,
    model,
    set_multiplier,
    multipliers: List[float],
    output_dir: str,
    args,
):
    results = []
    total_samples = len(dataset)

    if attack == "audio_ours_universal":
        set_multiplier(0.0)
        optim_audio_paths = audio_paths[: args.universal_size]
        optim_texts = [d["question"] for d in dataset[: args.universal_size]]
        uni_save_path = os.path.join(output_dir, "universal_adv.wav")
        suffix_path, record = generate_universal_adversarial_audio(
            optim_audio_paths,
            optim_texts,
            tokenizer,
            model,
            uni_save_path,
            num_token_suffix=args.suffix_length,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=args.device,
        )
        print(f"Universal suffix saved: {suffix_path}")

        import librosa
        import numpy as np

        suffix_audio, _ = librosa.load(suffix_path, sr=16000)

        for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
            idx = sample["id"]
            prompt = sample["question"]

            print(f"\n[{local_idx + 1}/{total_samples}] Evaluating sample {idx} (universal)")
            base_audio, _ = librosa.load(raw_audio, sr=16000)
            adv_audio = np.concatenate([base_audio, suffix_audio])
            adv_audio_path = os.path.join(output_dir, f"{idx}_adv.wav")
            save_audio(adv_audio_path, torch.tensor(adv_audio), 16000)

            result = {
                "id": idx,
                "source": sample["source"],
                "source_index": sample["source_index"],
                "question": sample["question"],
                "matching": sample["matching"],
                "not_matching": sample["not_matching"],
                "attack": attack,
                "audio_path": raw_audio,
                "adv_audio_path": adv_audio_path,
                "defense_results": {},
            }

            set_multiplier(0.0)
            print("  [Attack] Generating response (m=0.0)...", end=" ", flush=True)
            attack_response = llamaomni_eval_gen(
                [adv_audio_path],
                tokenizer,
                model,
                prompt="",
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            attack_outcome = judge_outcome(prompt, attack_response)
            result["attack_response"] = attack_response
            result["attack_label"] = attack_outcome["label"]
            result["attack_jailbreak"] = attack_outcome["is_jailbreak"]
            result["attack_invalid"] = attack_outcome["is_invalid"]
            result["attack_judge_error"] = attack_outcome.get("is_judge_error", False)
            result["attack_judge_reason"] = attack_outcome["reason"]

            if attack_outcome["is_jailbreak"]:
                print("Jailbreak")
            elif attack_outcome.get("is_judge_error", False):
                print("JudgeError")
            elif attack_outcome["is_invalid"]:
                print("Invalid")
            else:
                print("Refusal")

            for m in multipliers:
                set_multiplier(m)
                print(f"  [Defense m={m}] Generating response...", end=" ", flush=True)
                defense_response = inference_with_audio(
                    model,
                    tokenizer,
                    adv_audio_path,
                    args.device,
                    prompt="",
                    max_new_tokens=args.max_new_tokens,
                )
                defense_outcome = judge_outcome(prompt, defense_response)
                result["defense_results"][str(m)] = {
                    "response": defense_response,
                    "label": defense_outcome["label"],
                    "is_jailbreak": defense_outcome["is_jailbreak"],
                    "is_invalid": defense_outcome["is_invalid"],
                    "is_judge_error": defense_outcome.get("is_judge_error", False),
                    "defense_success": defense_outcome["label"] == "refusal",
                    "reason": defense_outcome["reason"],
                }

                if defense_outcome["is_jailbreak"]:
                    print("Jailbreak")
                elif defense_outcome.get("is_judge_error", False):
                    print("JudgeError")
                elif defense_outcome["is_invalid"]:
                    print("Invalid")
                else:
                    print("Refusal")

            results.append(result)
            with open(os.path.join(output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    elif attack == "audio_ours":
        for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
            idx = sample["id"]
            prompt = sample["question"]
            adv_path = os.path.join(output_dir, f"{idx}.wav")

            print(f"\n[{local_idx + 1}/{total_samples}] Processing sample {idx} (audio_ours)")

            set_multiplier(0.0)
            response_text, record = llamaomni_jailbreak_gen(
                prompt,
                [raw_audio],
                tokenizer,
                model,
                audio_save_path=adv_path,
                num_token_suffix=args.suffix_length,
                num_epochs=args.num_epochs,
                lr=args.lr,
                device=args.device,
            )

            if isinstance(record, dict) and "jailbreaked_audio" in record:
                adv_path = record["jailbreaked_audio"]

            if adv_path is None or not os.path.exists(adv_path):
                print(f"Skipping id={idx} - adv_audio not generated for audio_ours")
                result = {
                    "id": idx,
                    "source": sample["source"],
                    "source_index": sample["source_index"],
                    "question": sample["question"],
                    "matching": sample["matching"],
                    "not_matching": sample["not_matching"],
                    "attack": attack,
                    "audio_path": raw_audio,
                    "adv_audio_path": adv_path,
                    "skipped": True,
                    "skip_reason": "adv_audio not generated for audio_ours",
                    "defense_results": {},
                }
                results.append(result)
                with open(os.path.join(output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                continue

            attack_response = response_text if isinstance(response_text, str) else llamaomni_eval_gen(
                [adv_path],
                tokenizer,
                model,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            attack_outcome = judge_outcome(prompt, attack_response)

            result = {
                "id": idx,
                "source": sample["source"],
                "source_index": sample["source_index"],
                "question": sample["question"],
                "matching": sample["matching"],
                "not_matching": sample["not_matching"],
                "attack": attack,
                "audio_path": raw_audio,
                "adv_audio_path": adv_path,
                "attack_response": attack_response,
                "attack_label": attack_outcome["label"],
                "attack_jailbreak": attack_outcome["is_jailbreak"],
                "attack_invalid": attack_outcome["is_invalid"],
                "attack_judge_error": attack_outcome.get("is_judge_error", False),
                "attack_judge_reason": attack_outcome["reason"],
                "defense_results": {},
            }

            if attack_outcome["is_jailbreak"]:
                print("  [Attack] Result: Jailbreak")
            elif attack_outcome.get("is_judge_error", False):
                print("  [Attack] Result: JudgeError")
            elif attack_outcome["is_invalid"]:
                print("  [Attack] Result: Invalid")
            else:
                print("  [Attack] Result: Refusal")

            for m in multipliers:
                set_multiplier(m)
                print(f"  [Defense m={m}] Generating response...", end=" ", flush=True)
                defense_response = inference_with_audio(
                    model,
                    tokenizer,
                    adv_path,
                    args.device,
                    prompt="",
                    max_new_tokens=args.max_new_tokens,
                )
                defense_outcome = judge_outcome(prompt, defense_response)
                result["defense_results"][str(m)] = {
                    "response": defense_response,
                    "label": defense_outcome["label"],
                    "is_jailbreak": defense_outcome["is_jailbreak"],
                    "is_invalid": defense_outcome["is_invalid"],
                    "is_judge_error": defense_outcome.get("is_judge_error", False),
                    "defense_success": defense_outcome["label"] == "refusal",
                    "reason": defense_outcome["reason"],
                }

                if defense_outcome["is_jailbreak"]:
                    print("Jailbreak")
                elif defense_outcome.get("is_judge_error", False):
                    print("JudgeError")
                elif defense_outcome["is_invalid"]:
                    print("Invalid")
                else:
                    print("Refusal")

            results.append(result)
            with open(os.path.join(output_dir, "results.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported attack: {attack}")

    _save_jsonl(os.path.join(output_dir, "results.jsonl"), results)

    summary = {
        "config": {
            "model": args.model_name,
            "attack": attack,
            "multipliers": multipliers,
            "num_epochs": args.num_epochs,
            "suffix_length": args.suffix_length,
            "universal_size": args.universal_size,
            "lr": args.lr,
            "seed": args.seed,
            "dataset_size": len(dataset),
            "max_new_tokens": args.max_new_tokens,
        },
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "attack_stats": {},
        "multiplier_stats": {},
    }

    total = len(results)
    attack_success = sum(1 for r in results if r.get("attack_jailbreak", False))
    attack_invalid = sum(1 for r in results if r.get("attack_invalid", False))
    attack_judge_error = sum(1 for r in results if r.get("attack_judge_error", False))
    summary["attack_stats"] = {
        "total": total,
        "attack_success": attack_success,
        "attack_invalid": attack_invalid,
        "attack_judge_error": attack_judge_error,
        "attack_asr": (attack_success / total * 100) if total else 0,
        "attack_invalid_rate": (attack_invalid / total * 100) if total else 0,
        "attack_judge_error_rate": (attack_judge_error / total * 100) if total else 0,
    }

    for m in multipliers:
        m_key = str(m)
        cnt_jailbreak = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "jailbreak"
        )
        cnt_refusal = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "refusal"
        )
        cnt_invalid = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "invalid"
        )
        cnt_judge_error = sum(
            1
            for r in results
            if m_key in r.get("defense_results", {})
            and r["defense_results"][m_key].get("label") == "judge_error"
        )

        summary["multiplier_stats"][m_key] = {
            "jailbreak_count": cnt_jailbreak,
            "jailbreak_rate": (cnt_jailbreak / total * 100) if total else 0,
            "refusal_count": cnt_refusal,
            "refusal_rate": (cnt_refusal / total * 100) if total else 0,
            "invalid_count": cnt_invalid,
            "invalid_rate": (cnt_invalid / total * 100) if total else 0,
            "judge_error_count": cnt_judge_error,
            "judge_error_rate": (cnt_judge_error / total * 100) if total else 0,
            "defense_success_rate": (cnt_refusal / total * 100) if total else 0,
        }

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {output_dir}")
    return summary

def parse_args():
    parser = argparse.ArgumentParser(description="AdvWave defense runner for LLaMA-Omni2")
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.environ.get("LLAMA_OMNI2_MODEL", ""),
    )
    parser.add_argument("--vector_root", type=str, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--vector_epoch", type=int, default=300)
    parser.add_argument("--layers", nargs="+", type=int, default=[12])
    parser.add_argument("--attacks", nargs="+", default=["audio_ours"])
    parser.add_argument("--multipliers", nargs="+", type=float, default=[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--suffix_length", type=int, default=16000)
    parser.add_argument("--universal_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate_tts", action="store_true")
    parser.add_argument("--tts_max_retries", type=int, default=5)
    parser.add_argument("--tts_sleep", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=os.path.join(OUTPUT_DIR, "llamaomni2"))
    parser.add_argument("--attack_cache_dir", type=str, default=None)
    parser.add_argument("--seed_attack_cache_dir", type=str, default=None)
    parser.add_argument("--dataset_csv", type=str, default=None)
    parser.add_argument("--dedup_prompts", action="store_true")
    parser.add_argument("--reuse_existing_tts_by_prompt", action="store_true")
    parser.add_argument("--attack_only", action="store_true")
    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name is required.")
    return args

def main():
    args = parse_args()
    if args.device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device.split(":")[-1])
    if args.seed is None:
        args.seed = 0

    if args.dataset_csv:
        dataset = _load_dataset_csv(args.dataset_csv, dedup_prompts=args.dedup_prompts)
        print(f"Loaded dataset_csv={args.dataset_csv}, samples={len(dataset)}")
    else:
        dataset, seed = load_test_dataset(seed=args.seed)
        args.seed = seed
        if args.dedup_prompts:
            dataset = _dedup_dataset_by_prompt(dataset)
    if args.end is None:
        args.end = len(dataset)

    dataset = dataset[args.start : args.end]
    _apply_cached_raw_audio_paths(dataset, args.seed_attack_cache_dir)
    if args.reuse_existing_tts_by_prompt:
        _apply_existing_tts_audio_paths_by_prompt(dataset, args.seed)
    audio_paths = _ensure_audio_files_stable(
        dataset,
        generate_tts=args.generate_tts,
        tts_max_retries=args.tts_max_retries,
        tts_sleep=args.tts_sleep,
    )

    print("Loading tokenizer and model...")
    tokenizer = load_llama_omni2_tokenizer(args.model_name)
    model, resolved_model_name = load_llama_omni2_model(args.model_name, torch.bfloat16, args.device)
    model.eval()
    print(f"Using model: {resolved_model_name}")

    if args.attacks != ["audio_ours"]:
        raise ValueError("Cached all-layer runner currently supports only --attacks audio_ours")

    attack_cache_dir = args.attack_cache_dir
    if attack_cache_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attack_cache_dir = os.path.join(
            args.output_dir,
            "attack_cache",
            f"audio_ours_ep{args.num_epochs}_n{len(dataset)}_{timestamp}",
        )

    print(f"\n=== Stage 1/2: optimize attack audio once for {len(dataset)} sample(s) ===")
    attack_results = generate_audio_ours_attack_cache(
        dataset,
        audio_paths,
        tokenizer,
        model,
        attack_cache_dir,
        args,
    )
    if args.attack_only:
        print(f"Attack-only cache saved to {attack_cache_dir}")
        del model
        torch.cuda.empty_cache()
        return

    print(f"\n=== Stage 2/2: reuse cached attack audio for {len(args.layers)} layer(s) ===")
    layers = resolve_transformer_layers(model)
    for layer in args.layers:
        vector_path = os.path.abspath(_vector_path(args.vector_root, layer, args.vector_epoch))
        vector_tag = _vector_tag(vector_path)
        print(f"\n=== Running layer {layer}, vector {vector_tag} ===")

        steering_vector = load_steering_vector(vector_path, args.device)
        original_layer = layers[layer]
        wrapped_layer = apply_steering_to_model(model, steering_vector, layer, args.device)

        def set_multiplier(multiplier: float):
            wrapped_layer.set_multiplier(multiplier)

        output_dir = _ensure_output_dir(args.output_dir, vector_tag, "audio_ours_cached")
        run_cached_defense_for_layer(
            layer,
            attack_cache_dir,
            attack_results,
            tokenizer,
            model,
            set_multiplier,
            output_dir,
            args,
        )

        layers[layer] = original_layer
        del steering_vector
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
