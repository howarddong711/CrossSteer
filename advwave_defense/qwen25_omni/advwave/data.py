import os
import random
from typing import List, Tuple, Dict, Optional

import pandas as pd

from .paths import DATA_DIR, OUTPUT_DIR
from .tts import prompt2audio


def load_test_dataset(
    test_csv: Optional[str] = None,
    test_infer_csv: Optional[str] = None,
    seed: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[List[Dict], int]:
    def _normalize(path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        if isinstance(path, str):
            cleaned = path.strip()
            if cleaned == "" or cleaned.lower() in {"none", "null"}:
                return None
            return cleaned
        return path

    test_csv = _normalize(test_csv)
    test_infer_csv = _normalize(test_infer_csv)
    if test_csv is None and test_infer_csv is None:
        test_csv = os.path.join(DATA_DIR, "advbench_infer_200.csv")
        test_infer_csv = None

    dfs = []
    for path, source in ((test_csv, "test"), (test_infer_csv, "test_infer")):
        if path is None:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        df = pd.read_csv(path)
        df = df.copy()
        df["source"] = source
        df["source_index"] = list(range(len(df)))
        dfs.append(df)
    if not dfs:
        raise ValueError("No dataset files provided to load_test_dataset")

    merged = pd.concat(dfs, ignore_index=True)
    records = merged.to_dict(orient="records")

    if shuffle:
        if seed is None:
            seed = random.SystemRandom().randint(0, 2 ** 31 - 1)
        rng = random.Random(seed)
        rng.shuffle(records)
    else:
        if seed is None:
            seed = 0

    dataset = []
    for idx, row in enumerate(records):
        dataset.append(
            {
                "id": idx,
                "question": row.get("question", ""),
                "matching": row.get("matching", ""),
                "not_matching": row.get("not_matching", ""),
                "source": row.get("source", ""),
                "source_index": row.get("source_index", -1),
            }
        )

    return dataset, seed


def build_audio_paths() -> str:
    audio_dir = os.path.join(OUTPUT_DIR, "audio", "crosssteer_test", "0", "1")
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir


def ensure_audio_files(
    dataset: List[Dict],
    generate_tts: bool = True,
    tts_max_retries: int = 5,
    tts_sleep: float = 0.5,
) -> List[str]:
    audio_dir = build_audio_paths()
    audio_paths: List[str] = []

    for item in dataset:
        audio_path = os.path.join(audio_dir, f"{item['id']}.wav")
        if generate_tts:
            prompt2audio(
                item["question"],
                audio_path,
                max_retries=tts_max_retries,
                throttle_seconds=tts_sleep,
            )
        audio_paths.append(audio_path)

    return audio_paths


def ensure_audio_file(
    item: Dict,
    generate_tts: bool = True,
    tts_max_retries: int = 5,
    tts_sleep: float = 0.5,
) -> str:
    audio_dir = build_audio_paths()
    audio_path = os.path.join(audio_dir, f"{item['id']}.wav")
    if generate_tts:
        prompt2audio(
            item["question"],
            audio_path,
            max_retries=tts_max_retries,
            throttle_seconds=tts_sleep,
        )
    return audio_path
