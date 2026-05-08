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


def load_steering_vector(vector_path, device, dtype=torch.float16):
    if not os.path.exists(vector_path):
        raise FileNotFoundError(f"Steering vector not found: {vector_path}")
    if not vector_path.endswith(".pt"):
        raise ValueError(f"Steering vector must be .pt: {vector_path}")
    vector = torch.load(vector_path, map_location=device)
    return vector.to(dtype=dtype)


def apply_steering_to_model(model, steering_vector, layer_idx):
    target_layer = model.language_model.model.layers[layer_idx]
    wrapped_layer = BlockWrapper(target_layer, steering_vector=steering_vector, multiplier=0.0)
    wrapped_layer = wrapped_layer.to(next(model.parameters()).device)
    model.language_model.model.layers[layer_idx] = wrapped_layer
    return wrapped_layer


def inference_with_audio(model, processor, audio_path, device, prompt="", max_new_tokens=200):
    audio_url = "file:" + audio_path
    content = [{"type": "audio", "audio_url": audio_url}]
    if prompt:
        content.append({"type": "text", "text": prompt})
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
                        wav, sr = torchaudio.load(path)
                        if wav.ndim > 1:
                            wav = wav.mean(0)
                        if sr != processor.feature_extractor.sampling_rate:
                            wav = torchaudio.functional.resample(
                                wav, sr, processor.feature_extractor.sampling_rate
                            )
                        audios.append(wav.detach().cpu().numpy())
                    else:
                        audios.append(
                            librosa.load(
                                BytesIO(urlopen(audio_url).read()),
                                sr=processor.feature_extractor.sampling_rate,
                            )[0]
                        )

    for idx in range(len(audios)):
        audios[idx] = torch.tensor(audios[idx], device=device)
    audios = torch.cat(audios).unsqueeze_(0)

    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=16000)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    generate_ids = model.generate(
        **inputs,
        max_length=1024,
        do_sample=False,
        temperature=0.0,
        top_p=0,
        top_k=0,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response
