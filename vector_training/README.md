# CrossSteer Vector Training

This directory contains the training scripts for CrossSteer residual-stream steering vectors. The scripts train one vector per model while keeping the original audio-language model parameters frozen.

## Model Scripts

- `qwen2_audio/train_multimodal.py`: Qwen2-Audio-7B-Instruct vector training.
- `qwen25_omni/train_multimodal_qwenomni.py`: Qwen2.5-Omni-7B vector training.
- `llamaomni2/train_multimodal_llamaomni2.py`: LLaMA-Omni2-7B vector training.

## Example Commands

```bash
cd vector_training/qwen2_audio
python train_multimodal.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 15 --num_train_epochs 100

cd ../qwen25_omni
python train_multimodal_qwenomni.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 13 --num_train_epochs 400

cd ../llamaomni2
python train_multimodal_llamaomni2.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 12 --num_train_epochs 300
```

The shared AdvBench split is stored at `../data/advbench_train_300.csv` and `../data/advbench_infer_200.csv`.

External experiment logging is disabled by default. Pass `--report_to swanlab` only when SwanLab logging is needed.
