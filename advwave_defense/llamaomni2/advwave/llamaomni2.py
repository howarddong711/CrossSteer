import os
import sys
from io import BytesIO
from urllib.request import urlopen

import librosa
import torch
import torchaudio
import whisper
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForConditionalGeneration
except ImportError:
    AutoModelForConditionalGeneration = AutoModelForCausalLM

from .paths import DATA_DIR, OUTPUT_DIR


LLAMA_OMNI2_REPO = os.environ.get("LLAMA_OMNI2_REPO", "")
if LLAMA_OMNI2_REPO and LLAMA_OMNI2_REPO not in sys.path and os.path.exists(LLAMA_OMNI2_REPO):
    sys.path.insert(0, LLAMA_OMNI2_REPO)

try:
    from llama_omni2.constants import DEFAULT_SPEECH_TOKEN, SPEECH_TOKEN_INDEX
except Exception:
    DEFAULT_SPEECH_TOKEN = "<speech>"
    SPEECH_TOKEN_INDEX = -200


SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."


class BlockWrapper(torch.nn.Module):
    def __init__(self, block, steering_vector=None, multiplier=0.0):
        super().__init__()
        self.block = block
        self.steering_vector = steering_vector
        self.multiplier = multiplier

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if self.steering_vector is None or self.multiplier == 0.0:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            steering_vec = self._prep_vector(hidden_states)
            hidden_states = hidden_states + (self.multiplier * steering_vec)
            return (hidden_states,) + output[1:]

        steering_vec = self._prep_vector(output)
        return output + (self.multiplier * steering_vec)

    def _prep_vector(self, target_tensor):
        steering_vec = self.steering_vector.view(1, 1, -1)
        return steering_vec.to(dtype=target_tensor.dtype, device=target_tensor.device)

    def set_multiplier(self, multiplier):
        self.multiplier = float(multiplier)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)


def ensure_llama_omni2_registered():
    try:
        from llama_omni2.model.language_model.omni2_speech2s_qwen2 import (
            Omni2Speech2SQwen2ForCausalLM,
        )
        from llama_omni2.model.language_model.omni2_speech_qwen2 import (
            Omni2SpeechQwen2ForCausalLM,
        )
        return True
    except Exception as err:
        print(f"Failed to import llama_omni2 registry hooks: {err}")
        return False


def resolve_model_name_or_path(model_name):
    if os.path.exists(model_name):
        return model_name
    hf_home = os.environ.get("HF_HOME_DIR", "")
    local_candidate = os.path.join(hf_home, model_name) if hf_home else ""
    if os.path.exists(local_candidate):
        return local_candidate
    return model_name


def load_llama_omni2_model(model_name, dtype, device):
    if not ensure_llama_omni2_registered():
        raise RuntimeError("Failed to register LLaMA-Omni2 classes.")

    resolved_model_name = resolve_model_name_or_path(model_name)
    config = AutoConfig.from_pretrained(resolved_model_name, trust_remote_code=True)
    if hasattr(config, "speech_encoder") and config.speech_encoder:
        speech_encoder = str(config.speech_encoder)
        if not os.path.isabs(speech_encoder) and not os.path.exists(speech_encoder):
            candidates = [
                os.path.join(resolved_model_name, speech_encoder),
                os.path.join(os.environ.get("CROSSSTEER_PURE_ROOT", ""), speech_encoder),
                os.path.join(LLAMA_OMNI2_REPO, speech_encoder) if LLAMA_OMNI2_REPO else "",
            ]
            resolved_speech_encoder = next((p for p in candidates if os.path.exists(p)), None)
            if resolved_speech_encoder:
                config.speech_encoder = resolved_speech_encoder
    if hasattr(config, "speech_generator"):
        config.speech_generator = None

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "config": config,
        "low_cpu_mem_usage": False,
    }

    try:
        from llama_omni2.model.language_model.omni2_speech_qwen2 import Omni2SpeechQwen2ForCausalLM

        model = Omni2SpeechQwen2ForCausalLM.from_pretrained(resolved_model_name, **load_kwargs)
    except Exception as err:
        print(f"Omni2SpeechQwen2ForCausalLM load failed: {err}")
        try:
            model = AutoModelForCausalLM.from_pretrained(resolved_model_name, **load_kwargs)
        except Exception as auto_err:
            print(f"AutoModelForCausalLM load failed: {auto_err}")
            model = AutoModelForConditionalGeneration.from_pretrained(resolved_model_name, **load_kwargs)

    if not hasattr(model, "speech_generator"):
        model.speech_generator = None
    model = model.to(device)
    model.requires_grad_(False)
    return model, resolved_model_name


def load_llama_omni2_tokenizer(model_name):
    resolved_model_name = resolve_model_name_or_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def resolve_transformer_layers(model):
    candidates = [
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.layers,
    ]
    for getter in candidates:
        try:
            return getter(model)
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers on the LLaMA-Omni2 model.")


def load_steering_vector(vector_path, device, dtype=torch.bfloat16):
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found: {vector_path}")
    try:
        vector = torch.load(vector_path, map_location=device, weights_only=True)
    except TypeError:
        vector = torch.load(vector_path, map_location=device)
    return vector.to(dtype=dtype)


def apply_steering_to_model(model, steering_vector, layer_idx, device):
    layers = resolve_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} is out of bounds. Model has {len(layers)} layers.")
    wrapped_layer = BlockWrapper(layers[layer_idx], steering_vector=steering_vector, multiplier=0.0)
    wrapped_layer = wrapped_layer.to(device)
    layers[layer_idx] = wrapped_layer
    return wrapped_layer


def _load_audio_from_path(audio_path, target_sr=16000):
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        waveform = torch.from_numpy(data)
        if waveform.ndim > 1:
            waveform = waveform.mean(-1)
    if waveform.ndim > 1:
        waveform = waveform.mean(0)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.detach().cpu()


def _load_audio_from_url(audio_url, target_sr=16000):
    if audio_url.startswith("file:"):
        return _load_audio_from_path(audio_url.replace("file:", ""), target_sr).numpy()
    return librosa.load(BytesIO(urlopen(audio_url).read()), sr=target_sr)[0]


def save_audio(path, waveform, sample_rate=16000):
    import soundfile as sf

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu()
        if waveform.ndim > 1:
            waveform = waveform.squeeze(0)
        waveform = waveform.numpy()
    sf.write(path, waveform, sample_rate)


def waveform_to_speech_features(waveform, device):
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform)
    waveform = waveform.to(device=device, dtype=torch.float32)
    waveform = whisper.pad_or_trim(waveform)
    return whisper.log_mel_spectrogram(waveform, n_mels=128).permute(1, 0)


def build_conversation(prompt="", include_audio=True):
    prompt = (prompt or "").strip()
    if include_audio and prompt:
        user_content = f"{DEFAULT_SPEECH_TOKEN}\n{prompt}"
    elif include_audio:
        user_content = DEFAULT_SPEECH_TOKEN
    else:
        user_content = prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prepare_prompt_ids(tokenizer, prompt, device):
    conversation = build_conversation(prompt=prompt, include_audio=True)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
    )[0].to(dtype=torch.long)
    speech_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_TOKEN)
    if speech_token_id is not None and speech_token_id >= 0:
        input_ids[input_ids == speech_token_id] = SPEECH_TOKEN_INDEX
    return input_ids.unsqueeze(0).to(device)


def llamaomni_eval_gen(audio_list, tokenizer, model, prompt="", device="cuda:0", max_new_tokens=200):
    if len(audio_list) > 1:
        raise ValueError("Only single audio clip supported")

    waveform = _load_audio_from_path(audio_list[0], 16000)
    speech_features = waveform_to_speech_features(waveform, device)
    speech_tensor = speech_features.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
    speech_lengths = torch.LongTensor([speech_features.shape[0]]).to(device)
    input_ids = prepare_prompt_ids(tokenizer, prompt, device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            speech=speech_tensor,
            speech_lengths=speech_lengths,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def inference_with_audio(model, tokenizer, audio_path, device, prompt="", max_new_tokens=200):
    return llamaomni_eval_gen([audio_path], tokenizer, model, prompt=prompt, device=device, max_new_tokens=max_new_tokens)


def _load_target_text(prompt):
    import pandas as pd

    for name in ("advbench_train_300.csv", "advbench_infer_200.csv"):
        csv_path = os.path.join(DATA_DIR, name)
        if not os.path.exists(csv_path):
            continue
        data = pd.read_csv(csv_path)
        for _, row in data.iterrows():
            if row.get("question") == prompt:
                return row.get("matching", "Sure, here are the step-by-step instructions")
    return "Sure, here are the step-by-step instructions"


def _build_attack_batch(tokenizer, prompt, target_text, device):
    prompt_ids = prepare_prompt_ids(tokenizer, prompt="", device=device)[0]
    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0].to(device)
    input_ids = torch.cat([prompt_ids, target_ids], dim=0).unsqueeze(0)
    labels = torch.full_like(input_ids, -100)
    labels[0, -target_ids.shape[0] :] = target_ids
    return input_ids, labels


def llamaomni_jailbreak_gen(
    ori_prompt,
    audio_list,
    tokenizer,
    model,
    audio_save_path,
    num_token_suffix,
    num_epochs,
    lr=5e-3,
    noise_initial=None,
    device="cuda:0",
):
    if noise_initial is None:
        noise_initial = os.path.join(OUTPUT_DIR, "cache", "carhorn_initial.wav")
    if len(audio_list) > 1:
        raise ValueError("Only single audio clip supported")

    target_text = _load_target_text(ori_prompt)
    print(f"Target response: {target_text}")

    base_audio = _load_audio_from_path(audio_list[0], 16000).to(device)
    std = 0.01
    if noise_initial and os.path.exists(noise_initial):
        adv_audio_suffix = _load_audio_from_path(noise_initial, 16000)[:num_token_suffix].to(device)
        if adv_audio_suffix.numel() < num_token_suffix:
            pad = torch.randn(num_token_suffix - adv_audio_suffix.numel(), device=device) * std
            adv_audio_suffix = torch.cat([adv_audio_suffix, pad], dim=0)
    else:
        adv_audio_suffix = torch.randn([num_token_suffix], device=device) * std
    adv_audio_suffix.requires_grad_(True)

    input_ids, labels = _build_attack_batch(tokenizer, ori_prompt, target_text, device)
    optimizer = torch.optim.Adam([adv_audio_suffix], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    losses = []
    early_stop_threshold = 0.15
    patience = 50
    low_loss_count = 0

    from tqdm import tqdm

    pbar = tqdm(range(num_epochs))
    for _ in pbar:
        optimizer.zero_grad()
        full_audio = torch.cat([base_audio, adv_audio_suffix], dim=0)
        speech_features = waveform_to_speech_features(full_audio, device)
        speech_tensor = speech_features.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        speech_lengths = torch.LongTensor([speech_features.shape[0]]).to(device)

        output = model(
            input_ids=input_ids,
            speech=speech_tensor,
            speech_lengths=speech_lengths,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.detach().cpu().item()
        losses.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.4f}")

        if loss_val < early_stop_threshold:
            low_loss_count += 1
            if low_loss_count >= patience:
                pbar.set_description(f"Early stop at loss: {loss_val:.4f}")
                break
        else:
            low_loss_count = 0

    final_audio = torch.cat([base_audio.detach(), adv_audio_suffix.detach()], dim=0)
    save_audio(audio_save_path, final_audio, 16000)
    response = llamaomni_eval_gen([audio_save_path], tokenizer, model, device=device)
    record = {"original_audio": audio_list, "jailbreaked_audio": audio_save_path, "losses": losses}
    torch.cuda.empty_cache()
    return response, record


def generate_universal_adversarial_audio(
    audio_paths,
    text_list,
    tokenizer,
    model,
    save_path,
    num_token_suffix=16000,
    num_epochs=1000,
    lr=5e-3,
    device="cuda:0",
):
    print(f"Universal attack: optimizing suffix on {len(audio_paths)} audio(s) for {num_epochs} epochs")

    std = 0.01
    adv_audio_suffix = torch.randn([num_token_suffix], device=device) * std
    adv_audio_suffix.requires_grad_(True)
    optimizer = torch.optim.Adam([adv_audio_suffix], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    base_audio_cache = [_load_audio_from_path(path, 16000).to(device) for path in audio_paths]
    target_cache = []
    for prompt in text_list:
        target_text = _load_target_text(prompt)
        target_cache.append(_build_attack_batch(tokenizer, prompt, target_text, device))

    from tqdm import tqdm

    losses = []
    pbar = tqdm(range(num_epochs), desc="Universal optimization")
    for epoch in pbar:
        sample_idx = epoch % len(audio_paths)
        optimizer.zero_grad()
        full_audio = torch.cat([base_audio_cache[sample_idx], adv_audio_suffix], dim=0)
        speech_features = waveform_to_speech_features(full_audio, device)
        speech_tensor = speech_features.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        speech_lengths = torch.LongTensor([speech_features.shape[0]]).to(device)
        input_ids, labels = target_cache[sample_idx]

        output = model(
            input_ids=input_ids,
            speech=speech_tensor,
            speech_lengths=speech_lengths,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.detach().cpu().item()
        losses.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.4f} (sample {sample_idx})")

    suffix_path = save_path.replace(".wav", "_suffix.wav")
    save_audio(suffix_path, adv_audio_suffix, 16000)

    final_audio = torch.cat([base_audio_cache[0].detach(), adv_audio_suffix.detach()], dim=0)
    save_audio(save_path, final_audio, 16000)
    return suffix_path, {"losses": losses}
