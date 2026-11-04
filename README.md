# CrossSteer: Cross-Modal Safety Steering for Audio-Language Models

**Houde Dong, Weifei Jin, Yuxin Cao, Wei Song, Derui Wang, and Jie Hao**

NeurIPS 2026

CrossSteer is a representation-level defense against jailbreaks in audio-language models (ALMs). It learns a safety steering vector from text-only preference triples and transfers the intervention to spoken inputs through the model's shared language backbone. The model weights remain frozen: at inference, a scaled vector is added to the residual stream at a selected layer to encourage safe refusal.

![CrossSteer pipeline: text preference optimization and cross-modal inference](assets/crosssteer_pipeline.png)

In the paper, audio-derived activations are more diffuse than text-derived activations, while semantically matched text and audio become aligned in middle backbone layers. Learning the steering direction from text exploits this alignment without requiring audio preference training. Across three evaluated backbones, the paper reports average audio attack success rate (ASR) falling from **84.3% to 6.8%**, and text ASR from **60.5% to 29.9%**. These are paper results under the reported experimental settings, not a guarantee against adaptive attacks.

![Comparison of text-trained and audio-trained steering vectors](assets/modality_comparison.png)

The experiments cover Qwen2-Audio, Qwen2.5-Omni, and LLaMA-Omni2, plus a LLaMA-Omni2-14B scale setting. See the paper for the full evaluation, experimental protocol, and limitations.

## Repository Contents

- `vector_training/`: text preference-based vector training for the three backbones.
- `advwave_defense/`: AdvWave attack and CrossSteer defense runners.
- `vectors/`: released steering vectors.
- `data/`: 320/200 JSONL splits and the 520-example text and audio dataset.
- `scripts/check_vectors.py`: inspect the released vectors.

## Setup

Install the dependencies in a suitable Python environment with PyTorch and GPU support:

```bash
pip install -r requirements.txt
```

Download the model weights separately and pass the local model path to the relevant runner. Some AdvWave TTS and judge workflows use an API key:

```bash
export SILICONFLOW_API_KEY="<your-key>"
```

Never commit API keys. External experiment logging is disabled by default; pass `--report_to swanlab` to a training script to enable SwanLab.

To inspect the supplied vectors:

```bash
python scripts/check_vectors.py
```

## Dataset

The released AdvBench data contains harmful prompts and unsafe example responses for safety research. Handle it accordingly.

| Path | Contents |
| --- | --- |
| `data/AdvBench_Text.jsonl` | 520 text preference records: `prompt`, `jailbreak_response`, `refusal_response` |
| `data/AdvBench_Audio.jsonl` | 520 corresponding records with an additional `audio` filename |
| `data/advbench_audio/` | 520 audio files referenced by the audio JSONL |
| `data/advbench_train_320.jsonl` | 320 text preference records for training |
| `data/advbench_infer_200.jsonl` | 200 text preference records for evaluation |

Audio paths are relative to `data/advbench_audio/`. For example, a record with `"audio": "0.wav"` refers to `data/advbench_audio/0.wav`. The training and evaluation files are disjoint subsets of `AdvBench_Text.jsonl`. The corresponding audio records can be found in `AdvBench_Audio.jsonl` by matching the prompt.

The released audio files have a `.wav` suffix, but their contents are MP3-encoded. Use an audio decoder that detects the actual format and resample as required by the model. This release does not contain 16 kHz PCM WAV files.

## Steering Vectors

| Model | Layer | Epoch | Path |
| --- | ---: | ---: | --- |
| Qwen2-Audio-7B-Instruct | 15 | 100 | `vectors/qwen2_audio/vec_ep100_layer15.pt` |
| Qwen2.5-Omni-7B | 13 | 400 | `vectors/qwen25_omni/layer13/vec_ep400_layer13.pt` |
| LLaMA-Omni2-7B | 12 | 300 | `vectors/llamaomni2/layer12/vec_ep300_layer12.pt` |

## Training

Run from the repository root. Replace `<MODEL_PATH>` with the corresponding local model path:

```bash
cd vector_training/qwen2_audio
python train_multimodal.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 15 --num_train_epochs 100

cd ../qwen25_omni
python train_multimodal_qwenomni.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 13 --num_train_epochs 400

cd ../llamaomni2
python train_multimodal_llamaomni2.py --model_name_or_path <MODEL_PATH> --input_modality text --behavior jailbreak --layer 12 --num_train_epochs 300
```

These commands use the 320-example training split. Consult each script's arguments and model-specific dependencies before running.

## Defense Runners

Run from the repository root, with model weights and any required attack inputs prepared:

```bash
cd advwave_defense/qwen2_audio
python run_advwave_defense.py --model_name <MODEL_PATH> --attacks audio_ours --layer 15

cd ../qwen25_omni
python run_advwave_defense_qwenomni_cached.py --model_name <MODEL_PATH> --layers 13 --vector_epoch 400

cd ../llamaomni2
python run_advwave_defense_llamaomni2.py --model_name <MODEL_PATH> --layers 12 --vector_epoch 300
```

Generated attack caches, logs, and experiment outputs are excluded from the repository. The included WAV files are the dataset inputs, not generated attack outputs.

## Citation

```bibtex
@inproceedings{dong2026crosssteer,
  title     = {CrossSteer: Cross-Modal Safety Steering for Audio-Language Models},
  author    = {Dong, Houde and Jin, Weifei and Cao, Yuxin and Song, Wei and Wang, Derui and Hao, Jie},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```
