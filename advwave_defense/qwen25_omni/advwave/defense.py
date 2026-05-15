import os
from io import BytesIO
from urllib.request import urlopen

import librosa
import torch
import torchaudio

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

def resolve_transformer_layers(model):

    candidates = [
        ("thinker.model.layers", lambda m: m.thinker.model.layers),
        ("language_model.model.layers", lambda m: m.language_model.model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.model.layers),
        ("model.layers", lambda m: m.model.layers),
        ("layers", lambda m: m.layers),
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
            return path, layers
        except AttributeError:
            continue
    raise AttributeError("Could not locate transformer layers on the model.")

def load_steering_vector(vector_path, device, dtype=torch.float16):
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found: {vector_path}")
    if not vector_path.endswith(".pt"):
        raise ValueError(f"Steering vector must be .pt: {vector_path}")
    try:
        vector = torch.load(vector_path, map_location=device, weights_only=True)
    except TypeError:
        vector = torch.load(vector_path, map_location=device)
    return vector.to(dtype=dtype)

def _processor_call_with_audio(processor, *, text, audio, sampling_rate):
    import inspect

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    if isinstance(audio, (list, tuple)):
        audio = [_to_numpy(a) for a in audio]
    else:
        audio = _to_numpy(audio)

    use_audio = False
    try:
        params = inspect.signature(processor.__call__).parameters
        use_audio = "audio" in params
    except Exception:
        use_audio = False

    if use_audio:
        return processor(
            text=text,
            audio=audio,
            return_tensors="pt",
            padding=True,
            sampling_rate=sampling_rate,
        )
    return processor(
        text=text,
        audios=audio,
        return_tensors="pt",
        padding=True,
        sampling_rate=sampling_rate,
    )

def apply_steering_to_model(model, steering_vector, layer_idx):
    _, layers = resolve_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} is out of bounds. Model has {len(layers)} layers.")
    target_layer = layers[layer_idx]
    wrapped_layer = BlockWrapper(target_layer, steering_vector=steering_vector, multiplier=0.0)
    wrapped_layer = wrapped_layer.to(next(model.parameters()).device)
    layers[layer_idx] = wrapped_layer
    return wrapped_layer

def restore_steering_layer(model, layer_idx, original_layer):
    _, layers = resolve_transformer_layers(model)
    layers[layer_idx] = original_layer

def inference_with_audio(model, processor, audio_path, device, prompt="", max_new_tokens=200):
    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    audio_url = "file:" + audio_path
    content = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    content.append({"type": "audio", "audio_url": audio_url})
    conversation = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio_url = ele["audio_url"]
                    if audio_url.startswith("file:"):
                        path = audio_url.replace("file:", "")
                        audios.append(librosa.load(path, sr=target_sr, mono=True)[0])
                    else:
                        audios.append(
                            librosa.load(
                                BytesIO(urlopen(audio_url).read()),
                                sr=target_sr,
                            )[0]
                        )

    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device=device)
    audios = torch.cat(audios).unsqueeze_(0)

    inputs = _processor_call_with_audio(
        processor,
        text=text,
        audio=audios,
        sampling_rate=16000,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    try:
        generate_ids = model.generate(
            **inputs,
            max_length=1024,
            do_sample=False,
            temperature=0.0,
            top_p=0,
            top_k=0,
            speaker="Chelsie",
            return_audio=False,
        )
    except (TypeError, ValueError) as exc:
        if hasattr(model, "thinker"):
            try:
                generate_ids = model.thinker.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            except Exception:
                raise exc
        else:
            raise
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response
