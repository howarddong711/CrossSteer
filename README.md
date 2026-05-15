# CrossSteer

Code and artifacts for **CrossSteer: Cross-Modal Safety Steering for Audio-Language Models**.

CrossSteer is a representation-level safety defense for audio-language models. It learns a residual-stream steering direction from text-only safety preference triples and applies the same direction inside the shared language-model backbone during audio inference. The goal is to shift harmful-request generation from unsafe compliance toward safe refusal while preserving benign utility.

This repository contains steering-vector training code, AdvWave attack code, defense runners, AdvBench splits, and trained steering vectors for three audio-capable model backbones.

## Setup

```bash
pip install -r requirements.txt
```

The AdvWave TTS and judge components require an API key:

```bash
export SILICONFLOW_API_KEY=...
```

External experiment logging is disabled by default. Pass `--report_to swanlab` to the training scripts only if SwanLab logging is needed.

## Repository Layout

- `vector_training/qwen2_audio/`: Qwen2-Audio CrossSteer vector training.
- `vector_training/qwen25_omni/`: Qwen2.5-Omni CrossSteer vector training.
- `vector_training/llamaomni2/`: LLaMA-Omni2 CrossSteer vector training.
- `advwave_defense/qwen2_audio/`: Qwen2-Audio AdvWave attack and CrossSteer defense.
- `advwave_defense/qwen25_omni/`: Qwen2.5-Omni AdvWave attack and CrossSteer defense.
- `advwave_defense/llamaomni2/`: LLaMA-Omni2 AdvWave attack and CrossSteer defense.
- `vectors/`: trained `.pt` steering vectors used in the main runs.
- `data/`: AdvBench text split files used for training and inference.
- `scripts/check_vectors.py`: steering-vector loading check.


## Training

```bash
cd vector_training/qwen2_audio
python train_multimodal.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 15 --num_train_epochs 100

cd ../qwen25_omni
python train_multimodal_qwenomni.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 13 --num_train_epochs 400

cd ../llamaomni2
python train_multimodal_llamaomni2.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 12 --num_train_epochs 300
```

## Defense Runners

```bash
cd advwave_defense/qwen2_audio
python run_advwave_defense.py --model_name <MODEL_PATH> --attacks audio_ours --layer 15

cd ../qwen25_omni
python run_advwave_defense_qwenomni_cached.py --model_name <MODEL_PATH> --layers 13 --vector_epoch 400

cd ../llamaomni2
python run_advwave_defense_llamaomni2.py --model_name <MODEL_PATH> --layers 12 --vector_epoch 300
```

Generated audio caches, logs, and experiment outputs are excluded from the repository.
