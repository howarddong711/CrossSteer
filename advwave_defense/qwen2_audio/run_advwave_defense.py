#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

import torch

from advwave.attack import (
    generate_universal_adversarial_audio,
    qwen_eval_gen,
    qwen_jailbreak_gen,
)
from advwave.data import load_test_dataset, ensure_audio_files
from advwave.defense import (
    apply_steering_to_model,
    inference_with_audio,
    load_steering_vector,
)
from advwave.judge import judge_outcome
from advwave.paths import OUTPUT_DIR, TRANSFORMERS_DIR

if TRANSFORMERS_DIR not in sys.path:
    sys.path.insert(0, TRANSFORMERS_DIR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DEFAULT_MODEL = os.environ.get("QWEN2_AUDIO_MODEL", "")
DEFAULT_VECTORS = [
    os.path.join(REPO_ROOT, "vectors", "qwen2_audio", "vec_ep100_layer15.pt"),
]

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

def run_for_attack(
    attack: str,
    dataset: List[dict],
    audio_paths: List[str],
    processor,
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

        for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
            idx = sample["id"]
            prompt = sample["question"]

            print(f"\n[{local_idx + 1}/{total_samples}] Evaluating sample {idx} (universal)")
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
        for local_idx, (sample, raw_audio) in enumerate(zip(dataset, audio_paths)):
            idx = sample["id"]
            prompt = sample["question"]
            adv_path = os.path.join(output_dir, f"{idx}.wav")

            print(f"\n[{local_idx + 1}/{total_samples}] Processing sample {idx} (audio_ours)")

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
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=["audio_ours", "audio_ours_universal"],
    )
    parser.add_argument("--multipliers", nargs="+", type=float, default=[-2.0, -1.0, 0.0])
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
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)

    args = parser.parse_args()
    if not args.model_name:
        parser.error("--model_name is required.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "cuda:0"

    dataset, seed = load_test_dataset(seed=args.seed)
    args.seed = seed
    if args.end is None:
        args.end = len(dataset)

    dataset = dataset[args.start : args.end]
    audio_paths = ensure_audio_files(
        dataset,
        generate_tts=args.generate_tts,
        tts_max_retries=args.tts_max_retries,
        tts_sleep=args.tts_sleep,
    )

    from transformers.models.qwen2_audio.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
    from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor

    for vector_path in args.vectors:
        vector_path = os.path.abspath(vector_path)
        vector_tag = _vector_tag(vector_path)
        print(f"\n=== Running vector {vector_tag} ===")

        print("Loading processor and model...")
        processor = Qwen2AudioProcessor.from_pretrained(args.model_name)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        model.eval()

        steering_vector = load_steering_vector(vector_path, device)
        wrapped_layer = apply_steering_to_model(model, steering_vector, args.layer)

        def set_multiplier(multiplier: float):
            wrapped_layer.set_multiplier(multiplier)

        for attack in args.attacks:
            output_dir = _ensure_output_dir(args.output_dir, vector_tag, attack)
            run_for_attack(
                attack,
                dataset,
                audio_paths,
                processor,
                model,
                set_multiplier,
                args.multipliers,
                output_dir,
                args,
            )

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
