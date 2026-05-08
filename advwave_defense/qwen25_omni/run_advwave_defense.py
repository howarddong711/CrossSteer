#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import torch

from advwave.attack import (
    generate_universal_adversarial_audio,
    qwen_eval_gen,
    qwen_jailbreak_gen,
)
from advwave.data import load_test_dataset, ensure_audio_file
from advwave.defense import (
    apply_steering_to_model,
    inference_with_audio,
    load_steering_vector,
)
from advwave.judge import judge_outcome
from advwave.paths import OUTPUT_DIR, TRANSFORMERS_DIR


use_third_party = False
try:
    from transformers import Qwen2_5OmniForConditionalGeneration
except Exception:
    use_third_party = True
if use_third_party and TRANSFORMERS_DIR not in sys.path:
    sys.path.insert(0, TRANSFORMERS_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_MODEL = os.environ.get("QWEN25_OMNI_MODEL", "")
DEFAULT_VECTORS = [
    os.path.join(REPO_ROOT, "vectors", "qwen25_omni", "layer13", "vec_ep400_layer13.pt"),
]


def patch_qwen2_5_omni_load_speakers() -> None:

    try:
        import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as qwen_mod
    except Exception as err:
        print(f"⚠️  Unable to import qwen2_5_omni modeling for patch: {err}")
        return

    def patched_load_speakers(self, *args, **kwargs):
        print("⚠️  Skipped load_speakers (PyTorch<2.6 workaround)")
        return None

    try:
        qwen_mod.Qwen2_5OmniForConditionalGeneration.load_speakers = patched_load_speakers
        print("✅ Patched Qwen2.5-Omni load_speakers")
    except Exception as err:
        print(f"⚠️  Failed to patch load_speakers: {err}")


def ensure_qwen_speaker_map(model, default_speaker: str = "Chelsie") -> None:

    if not hasattr(model, "speaker_map"):
        return
    speaker_map = getattr(model, "speaker_map", None)
    if speaker_map is None:
        model.speaker_map = {default_speaker: None}
        return
    try:
        if len(speaker_map) == 0:
            model.speaker_map = {default_speaker: None}
        elif default_speaker not in speaker_map:
            try:
                sample_val = next(iter(speaker_map.values()))
            except StopIteration:
                sample_val = None
            speaker_map[default_speaker] = sample_val
    except TypeError:
        model.speaker_map = {default_speaker: None}

def patch_qwen_omni_forward(model) -> None:

    needs_embed_override = False
    if not hasattr(model, "get_input_embeddings"):
        needs_embed_override = True
    else:
        try:
            _ = model.get_input_embeddings()
        except Exception:
            needs_embed_override = True

    if needs_embed_override:
        def _get_input_embeddings():
            if hasattr(model, "thinker") and hasattr(model.thinker, "get_input_embeddings"):
                return model.thinker.get_input_embeddings()
            if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "embed_tokens"):
                return model.thinker.model.embed_tokens
            return None

        def _set_input_embeddings(value):
            if hasattr(model, "thinker") and hasattr(model.thinker, "set_input_embeddings"):
                return model.thinker.set_input_embeddings(value)
            if hasattr(model, "thinker") and hasattr(model.thinker, "model") and hasattr(model.thinker.model, "embed_tokens"):
                model.thinker.model.embed_tokens = value
                return

        model.get_input_embeddings = _get_input_embeddings
        model.set_input_embeddings = _set_input_embeddings

    if getattr(model.__class__, "forward", None) is torch.nn.Module.forward and hasattr(model, "thinker"):
        def _forward(*args, **kwargs):
            return model.thinker(*args, **kwargs)
        model.forward = _forward


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
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  Skipping malformed jsonl line in {path}")
    return rows


def _build_resume_dataset(
    base_dataset: List[dict],
    existing_results: List[dict],
) -> Tuple[List[dict], Set[int]]:
    total = len(base_dataset)
    record_by_key = {(d.get("source"), d.get("source_index")): d for d in base_dataset}
    final: List[Optional[dict]] = [None] * total
    used_keys: Set[Tuple[Optional[str], Optional[int]]] = set()
    processed_ids: Set[int] = set()

    for row in existing_results:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not isinstance(rid, int):
            continue
        if rid < 0 or rid >= total:
            print(f"⚠️  Skipping out-of-range id={rid} in existing results")
            continue
        key = (row.get("source"), row.get("source_index"))
        record = record_by_key.get(key)
        if record is None:
            record = {
                "question": row.get("question", ""),
                "matching": row.get("matching", ""),
                "not_matching": row.get("not_matching", ""),
                "source": row.get("source", ""),
                "source_index": row.get("source_index", -1),
            }
        final[rid] = record
        used_keys.add(key)
        processed_ids.add(rid)

    remaining = [
        rec
        for rec in base_dataset
        if (rec.get("source"), rec.get("source_index")) not in used_keys
    ]
    remaining_iter = iter(remaining)
    for i in range(total):
        if final[i] is None:
            try:
                final[i] = next(remaining_iter)
            except StopIteration:
                break

    dataset: List[dict] = []
    for i, rec in enumerate(final):
        if rec is None:
            continue
        dataset.append(
            {
                "id": i,
                "question": rec.get("question", ""),
                "matching": rec.get("matching", ""),
                "not_matching": rec.get("not_matching", ""),
                "source": rec.get("source", ""),
                "source_index": rec.get("source_index", -1),
            }
        )
    return dataset, processed_ids


def run_for_attack(
    attack: str,
    dataset: List[dict],
    processor,
    model,
    set_multiplier,
    multipliers: List[float],
    output_dir: str,
    args,
    existing_results: Optional[List[dict]] = None,
    processed_ids: Optional[Set[int]] = None,
):
    results = list(existing_results) if existing_results else []
    total_samples = len(dataset)
    audio_cache = {}
    processed_ids = set(processed_ids) if processed_ids else set()

    def get_audio_path(sample: dict) -> str:
        sample_id = sample["id"]
        if sample_id in audio_cache:
            return audio_cache[sample_id]
        audio_path = ensure_audio_file(
            sample,
            generate_tts=args.generate_tts,
            tts_max_retries=args.tts_max_retries,
            tts_sleep=args.tts_sleep,
        )
        audio_cache[sample_id] = audio_path
        return audio_path

    if attack == "audio_ours_universal":
        set_multiplier(0.0)
        optim_audio_paths = [get_audio_path(sample) for sample in dataset[: args.universal_size]]
        optim_texts = [d["question"] for d in dataset[: args.universal_size]]
        uni_save_path = os.path.join(output_dir, "universal_adv.wav")
        suffix_path, record = generate_universal_adversarial_audio(
            optim_audio_paths,
            optim_texts,
            processor,
            model,
            uni_save_path,
            num_token_suffix=args.suffix_length,
            num_epochs=args.num_epochs,
            lr=args.lr,
        )
        print(f"Universal suffix saved: {suffix_path}")

        import librosa
        import numpy as np
        import torchaudio

        suffix_audio, _ = librosa.load(suffix_path, sr=16000)

        for local_idx, sample in enumerate(dataset):
            raw_audio = get_audio_path(sample)
            idx = sample["id"]
            prompt = sample["question"]

            print(f"\n[{local_idx + 1}/{total_samples}] Evaluating sample {idx} (universal)")
            if idx in processed_ids:
                print(f"Skipping id={idx} - already in results.jsonl")
                continue
            base_audio, _ = librosa.load(raw_audio, sr=16000)
            adv_audio = np.concatenate([base_audio, suffix_audio])
            adv_audio_path = os.path.join(output_dir, f"{idx}_adv.wav")
            torchaudio.save(adv_audio_path, torch.tensor(adv_audio).unsqueeze(0), 16000)

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
            attack_response = qwen_eval_gen([adv_audio_path], processor, model, prompt="")
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
                defense_response = inference_with_audio(model, processor, adv_audio_path, "cuda", prompt="")
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
        for local_idx, sample in enumerate(dataset):
            raw_audio = get_audio_path(sample)
            idx = sample["id"]
            prompt = sample["question"]
            adv_path = os.path.join(output_dir, f"{idx}.wav")

            print(f"\n[{local_idx + 1}/{total_samples}] Processing sample {idx} (audio_ours)")
            if idx in processed_ids:
                print(f"Skipping id={idx} - already in results.jsonl")
                continue

            set_multiplier(0.0)
            response_text, record = qwen_jailbreak_gen(
                prompt,
                [raw_audio],
                processor,
                model,
                audio_save_path=adv_path,
                num_token_suffix=args.suffix_length,
                num_epochs=args.num_epochs,
                advnoise_control=False,
                control_obj="",
                model_judge=None,
                targets=[],
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

            attack_response = response_text if isinstance(response_text, str) else qwen_eval_gen(
                [adv_path], processor, model
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
                defense_response = inference_with_audio(model, processor, adv_path, "cuda", prompt="")
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
            "attack": attack,
            "multipliers": multipliers,
            "num_epochs": args.num_epochs,
            "suffix_length": args.suffix_length,
            "universal_size": args.universal_size,
            "lr": args.lr,
            "seed": args.seed,
            "dataset_size": len(dataset),
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


def main():
    parser = argparse.ArgumentParser(description="AdvWave defense runner (audio_ours, audio_ours_universal)")
    parser.add_argument(
        "--vectors",
        nargs="+",
        default=DEFAULT_VECTORS,
    )
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["audio_ours", "audio_ours_universal"],
    )
    parser.add_argument("--multipliers", nargs="+", type=float, default=[-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--suffix_length", type=int, default=16000)
    parser.add_argument("--universal_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generate_tts", action="store_true")
    parser.add_argument("--tts_max_retries", type=int, default=5)
    parser.add_argument("--tts_sleep", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--test_csv", type=str, default=None)
    parser.add_argument("--test_infer_csv", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)

    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name is required.")

    device = "cuda:0"

    resume_results: Optional[List[dict]] = None
    processed_ids: Set[int] = set()

    if args.resume_dir is not None:
        if len(args.vectors) != 1 or len(args.attacks) != 1:
            raise ValueError("--resume_dir requires a single vector and single attack")
        if not os.path.isdir(args.resume_dir):
            raise FileNotFoundError(f"Resume directory not found: {args.resume_dir}")
        resume_path = os.path.join(args.resume_dir, "results.jsonl")
        resume_results = _load_jsonl(resume_path)
        base_dataset, seed = load_test_dataset(
            test_csv=args.test_csv,
            test_infer_csv=args.test_infer_csv,
            seed=args.seed,
            shuffle=args.seed is not None,
        )
        args.seed = seed
        dataset, processed_ids = _build_resume_dataset(base_dataset, resume_results)
        print(f"Resuming from {resume_path} ({len(processed_ids)} entries)")
    else:
        dataset, seed = load_test_dataset(
            test_csv=args.test_csv,
            test_infer_csv=args.test_infer_csv,
            seed=args.seed,
        )
        args.seed = seed
    if args.end is None:
        args.end = len(dataset)

    dataset = dataset[args.start : args.end]
    patch_qwen2_5_omni_load_speakers()
    from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration

    for vector_path in args.vectors:
        vector_path = os.path.abspath(vector_path)
        vector_tag = _vector_tag(vector_path)
        print(f"\n=== Running vector {vector_tag} ===")

        print("Loading processor and model...")
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

        steering_vector = load_steering_vector(vector_path, device)
        wrapped_layer = apply_steering_to_model(model, steering_vector, args.layer)

        def set_multiplier(multiplier: float):
            wrapped_layer.set_multiplier(multiplier)

        for attack in args.attacks:
            if args.resume_dir is not None:
                output_dir = args.resume_dir
            else:
                output_dir = _ensure_output_dir(args.output_dir, vector_tag, attack)
            run_for_attack(
                attack,
                dataset,
                processor,
                model,
                set_multiplier,
                args.multipliers,
                output_dir,
                args,
                existing_results=resume_results,
                processed_ids=processed_ids,
            )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
