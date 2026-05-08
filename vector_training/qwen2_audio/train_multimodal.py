import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ.setdefault('HF_ENDPOINT', 'https://huggingface.co')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


os.environ['HF_DATASETS_IN_MEMORY_MAX_SIZE'] = '0'
os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import warnings
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")

import gc
import random
import numpy as np
import torch
import logging
import pandas as pd
import psutil
import swanlab
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from datasets import Dataset, concatenate_datasets
from transformers import AutoProcessor, AutoTokenizer, Qwen2AudioForConditionalGeneration, HfArgumentParser
from trl import DPOConfig
from trl_text_trainer import CrossSteerTextTrainer


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("training_multimodal.log")
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CROSSSTEER_DATA_DIR = os.path.join(REPO_ROOT, "data")


def resolve_advbench_split(train=True):
    filename = "advbench_train_300.csv" if train else "advbench_infer_200.csv"
    return os.path.join(CROSSSTEER_DATA_DIR, filename)

class BlockWrapper(torch.nn.Module):








    def __init__(self, block, hidden_size):
        super().__init__()
        self.block = block

        self.vec = torch.nn.Parameter(torch.zeros(hidden_size))

        self.register_buffer('initial_vec', torch.zeros(hidden_size))

        self.multiplier_value = 0.0
        self._use_initial_vec = False

    @property
    def multiplier(self):

        return self.multiplier_value

    def forward(self, *args, **kwargs):

        output = self.block(*args, **kwargs)


        if self.multiplier_value == 0.0:
            return output


        active_vec = self.initial_vec if self._use_initial_vec else self.vec


        if isinstance(output, tuple):
            hidden_states = output[0]

            steering_vec = active_vec.view(1, 1, -1).to(hidden_states.dtype)

            modified_hidden_states = hidden_states + (self.multiplier_value * steering_vec)
            return (modified_hidden_states,) + output[1:]
        else:
            steering_vec = active_vec.view(1, 1, -1).to(output.dtype)
            modified_output = output + (self.multiplier_value * steering_vec)
            return modified_output

    def set_multiplier(self, multiplier):

        self.multiplier_value = float(multiplier)

    def get_multiplier(self):

        return self.multiplier_value

    def use_reference_mode(self, enable=True):






        self._use_initial_vec = enable

    def save_initial_vec(self):





        with torch.no_grad():
            self.initial_vec.copy_(self.vec.data)

    def __getattr__(self, name):

        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def extra_repr(self):
        return f'multiplier={self.multiplier_value:.4f}, vec_shape={self.vec.shape}'


def print_detailed_memory(message=""):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    ram_percent = psutil.virtual_memory().percent
    logger.debug(f"{message} 💾 RAM: {ram_gb:.2f} GB ({ram_percent}%)")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.debug(f"{message} 🎮 GPU {i}: {allocated:.2f}/{total:.2f} GB allocated, {reserved:.2f} GB reserved")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seed set to {seed}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    trainable_percentage = 100 * trainable_params / all_param
    logger.debug(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable %: {trainable_percentage:.4f}%"
    )


def get_text_dataset(behavior='power-seeking', train=True):

    dataset_type = "train" if train else "test"
    logger.debug(f"📊 Loading text data: behavior={behavior}, type={dataset_type}")
    print_detailed_memory(f"{dataset_type}数据加载前 ")

    data_file = resolve_advbench_split(train)

    try:
        df = pd.read_csv(data_file)
        df = df.fillna('')
        df = df.rename(columns={
            'question': 'prompt',
            'matching': 'matching',
            'not_matching': 'not_matching'
        })
        if 'prompt' not in df.columns:
            df['prompt'] = ''
        df['audio_path'] = ''
        df['modal_type'] = 'text'

        dataset = Dataset.from_pandas(df.loc[:, ['prompt', 'matching', 'not_matching', 'audio_path', 'modal_type']])

    except FileNotFoundError as e:
        logger.error(f"FATAL: Dataset file not found at {data_file}.")
        logger.error(f"Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"FATAL: Failed to load CSV at {data_file}. Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    logger.debug(f"✅ {dataset_type.capitalize()} dataset loaded (text), samples: {len(dataset)}")
    print_detailed_memory(f"{dataset_type}文本读取完毕 ")
    return dataset


def get_audio_dataset(behavior='jailbreak', train=True, force_empty_prompt=False):

    raise ValueError("CrossSteer vector training uses the shared text AdvBench split. Run with --input_modality text.")


def build_multimodal_dataset(text_behavior, audio_behavior, train=True):

    text_ds = get_text_dataset(text_behavior, train)
    audio_ds = get_audio_dataset(audio_behavior, train)
    combined = concatenate_datasets([text_ds, audio_ds])
    logger.debug(
        f"✅ Combined multimodal dataset built: text={len(text_ds)} + audio={len(audio_ds)} => total={len(combined)}"
    )
    return combined


def determine_precision():

    if not torch.cuda.is_available():
        return {"bf16": False, "fp16": False}, torch.float32

    device_index = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device_index)
    bf16_supported = major >= 8

    if bf16_supported:
        return {"bf16": True, "fp16": False}, torch.bfloat16


    return {"bf16": False, "fp16": True}, torch.float16


def resolve_model_load_mode(script_args, precision_flags, default_dtype):

    valid_modes = {"auto", "fp32", "fp16", "bf16", "8bit", "4bit"}
    mode = (script_args.model_load_mode or "auto").lower()
    if mode not in valid_modes:
        raise ValueError(f"Unsupported model_load_mode '{mode}'. Valid options: {sorted(valid_modes)}")

    total_mem_gb = None
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_mem_gb = props.total_memory / 1024 ** 3

    if mode == "auto":
        if total_mem_gb is not None and total_mem_gb <= script_args.auto_quantization_threshold_gb:
            mode = "8bit"
        elif precision_flags["bf16"]:
            mode = "bf16"
        elif precision_flags["fp16"]:
            mode = "fp16"
        else:
            mode = "fp32"

    load_kwargs = {"low_cpu_mem_usage": True}
    chosen_dtype = default_dtype

    current_device = torch.cuda.current_device() if torch.cuda.is_available() else None

    if mode == "4bit":
        chosen_dtype = torch.float16
        precision_flags["bf16"] = False
        precision_flags["fp16"] = True
        load_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": chosen_dtype,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "device_map": {"": current_device or 0},
            }
        )
    elif mode == "8bit":
        chosen_dtype = torch.float16
        precision_flags["bf16"] = False
        precision_flags["fp16"] = True
        load_kwargs.update(
            {
                "load_in_8bit": True,
                "device_map": {"": current_device or 0},
            }
        )
    elif mode == "bf16":
        chosen_dtype = torch.bfloat16
        precision_flags["bf16"] = True
        precision_flags["fp16"] = False
        load_kwargs["torch_dtype"] = chosen_dtype
        if current_device is not None:
            load_kwargs["device_map"] = {"": current_device}
    elif mode == "fp16":
        chosen_dtype = torch.float16
        precision_flags["bf16"] = False
        precision_flags["fp16"] = True
        load_kwargs["torch_dtype"] = chosen_dtype
        if current_device is not None:
            load_kwargs["device_map"] = {"": current_device}
    elif mode == "fp32":
        chosen_dtype = torch.float32
        precision_flags["bf16"] = False
        precision_flags["fp16"] = False
        load_kwargs["torch_dtype"] = chosen_dtype
        if current_device is not None:
            load_kwargs["device_map"] = {"": current_device}

    return load_kwargs, chosen_dtype, precision_flags, mode, total_mem_gb


@dataclass
class ScriptArguments:
    beta: Optional[float] = field(default=0.1)
    model_name_or_path: Optional[str] = field(default="")
    learning_rate: Optional[float] = field(default=1e-4)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    warmup_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.05)
    optimizer_type: Optional[str] = field(default="adamw_torch")
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    gradient_checkpointing: Optional[bool] = field(default=True)
    max_prompt_length: Optional[int] = field(default=512)
    max_length: Optional[int] = field(default=1024)
    num_train_epochs: Optional[int] = field(default=100)
    logging_steps: Optional[int] = field(default=10)
    behavior: Optional[str] = field(default="jailbreak", metadata={"help": "Dataset behavior: power-seeking | jailbreak | helpful-ethical"})
    layer: Optional[int] = field(default=15)
    output_dir: Optional[str] = field(default="./output_text_steer")
    report_to: Optional[str] = field(default="none")
    num_proc: Optional[int] = field(default=1)
    model_load_mode: Optional[str] = field(default="fp16")
    auto_quantization_threshold_gb: Optional[float] = field(default=36.0)
    input_modality: Optional[str] = field(default="text", metadata={"help": "text"})
    text_behavior: Optional[str] = field(default=None, metadata={"help": "Override text dataset behavior"})
    audio_behavior: Optional[str] = field(default=None, metadata={"help": "Override audio dataset behavior"})

    resume_from_epoch: Optional[int] = field(
        default=None,
        metadata={"help": "从指定epoch的vector恢复训练。例如：40表示加载vec_ep40_layer10.pt并从epoch 41开始"}
    )


if __name__ == "__main__":
    logger.debug("🎬 Starting unified training script...")
    logger.debug(f"🖥️  System Info: CPU cores={psutil.cpu_count()}, RAM={psutil.virtual_memory().total/1024**3:.1f}GB")

    torch.cuda.empty_cache()
    gc.collect()
    print_detailed_memory("程序启动后 ")

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    if not script_args.model_name_or_path:
        raise ValueError("--model_name_or_path is required.")
    logger.debug(f"CUDA available: {torch.cuda.is_available()}")
    set_seed(seed=11)
    logger.debug(f"[Behavior]: {script_args.behavior}, [Layer]: {script_args.layer}, [Model]: {script_args.model_name_or_path}")

    modality = (script_args.input_modality or "text").lower()
    valid_modalities = {"text"}
    if modality not in valid_modalities:
        raise ValueError(f"Unsupported input_modality '{modality}'. Choose from {sorted(valid_modalities)}")
    text_behavior = script_args.text_behavior or script_args.behavior
    audio_behavior = script_args.audio_behavior or script_args.behavior
    if script_args.output_dir == "./output_text_steer":
        script_args.output_dir = f"./output_{modality}_steer"
    vector_tag = f"{script_args.behavior}_qwen2-audio-{modality}"
    logger.debug(f"📥 Input modality: {modality} (text_behavior={text_behavior}, audio_behavior={audio_behavior})")

    precision_flags, target_dtype = determine_precision()
    load_kwargs, target_dtype, precision_flags, load_mode, total_mem_gb = resolve_model_load_mode(
        script_args, precision_flags, target_dtype
    )



    logger.debug(f"🔧 Original precision: bf16={precision_flags['bf16']}, fp16={precision_flags['fp16']}")
    precision_flags['fp16'] = False
    precision_flags['bf16'] = False
    logger.debug(f"🔧 Modified precision: bf16={precision_flags['bf16']}, fp16={precision_flags['fp16']} (using FP32 for training)")
    if total_mem_gb is not None:
        logger.debug(
            f"🧮 Precision settings – dtype: {target_dtype}, bf16: {precision_flags['bf16']}, "
            f"fp16: {precision_flags['fp16']}, gpu_mem: {total_mem_gb:.2f} GB"
        )
    else:
        logger.debug(
            f"🧮 Precision settings – dtype: {target_dtype}, bf16: {precision_flags['bf16']}, "
            f"fp16: {precision_flags['fp16']}"
        )
    logger.debug(f"🧲 Model load mode resolved to '{load_mode}' (threshold={script_args.auto_quantization_threshold_gb} GB)")
    if precision_flags["bf16"] is False and precision_flags["fp16"] is False and torch.cuda.is_available():
        logger.warning("Current GPU does not support bf16/fp16; training will run in full precision.")

    logger.debug("🤖 Loading processors/tokenizers based on modality...")
    print_detailed_memory("加载Tokenizer/Processor前 ")

    processor = None
    if modality == "text":
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    else:
        processor = AutoProcessor.from_pretrained(script_args.model_name_or_path)
        tokenizer = processor.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token



    use_swanlab = (script_args.report_to or "").lower() == "swanlab"
    if use_swanlab:
        safe_model_name = script_args.model_name_or_path.replace("/", "_")
        swanlab.init(
            project=f"CrossSteer-{safe_model_name}",
            experiment_name=f"{script_args.behavior}-{modality}-layer{script_args.layer}",
            description=f"CrossSteer training for {script_args.model_name_or_path} ({modality}) on {script_args.behavior} dataset",
            config={
                "model": script_args.model_name_or_path,
                "mode": modality,
                "behavior": script_args.behavior,
                "layer": script_args.layer,
                "learning_rate": script_args.learning_rate,
                "batch_size": script_args.per_device_train_batch_size,
                "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
                "num_epochs": script_args.num_train_epochs,
            },
        )
        logger.debug("✅ SwanLab initialized")


    gc.collect()
    torch.cuda.empty_cache()

    print_detailed_memory("加载主模型前 ")

    logger.debug("Loading main model to GPU 0...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        **load_kwargs,
    )
    if load_kwargs.get("device_map") is None:
        model = model.to("cuda:0")
    model.config.torch_dtype = target_dtype


    model.config.use_cache = False

    print_detailed_memory("加载主模型后 ")
    hidden_size = model.config.text_config.hidden_size
    logger.debug(f"📐 Hidden size: {hidden_size}")

    logger.debug(f"🔧 Wrapping layer {script_args.layer} with BlockWrapper...")
    try:
        target_layer = model.language_model.model.layers[script_args.layer]
        wrapped_layer = BlockWrapper(target_layer, hidden_size=hidden_size)

        wrapped_layer = wrapped_layer.to("cuda:0")



        wrapped_layer.vec.data = wrapped_layer.vec.data.to(torch.float32)
        wrapped_layer.initial_vec = wrapped_layer.initial_vec.to(torch.float32)


        wrapped_layer.save_initial_vec()
        logger.debug(f"💾 Saved initial steering vector as reference (all zeros, FP32)")


        if script_args.resume_from_epoch is not None:

            vec_dir = f"./vector/{vector_tag}"
            vec_filename = f"vec_ep{script_args.resume_from_epoch}_layer{script_args.layer}.pt"
            vec_path = os.path.join(vec_dir, f"layer{script_args.layer}", vec_filename)

            logger.debug(f"🔄 Attempting to resume from epoch {script_args.resume_from_epoch}")
            logger.debug(f"   Vector path: {vec_path}")

            if os.path.exists(vec_path):
                try:

                    loaded_vec = torch.load(vec_path, map_location="cpu")
                    logger.debug(f"   Loaded vector shape: {loaded_vec.shape}, dtype: {loaded_vec.dtype}")


                    wrapped_layer.vec.data.copy_(loaded_vec.to(dtype=wrapped_layer.vec.dtype, device=wrapped_layer.vec.device))


                    wrapped_layer.initial_vec.copy_(loaded_vec.to(dtype=torch.float32, device=wrapped_layer.vec.device))

                    logger.debug(f"✅ Successfully restored steering vector from epoch {script_args.resume_from_epoch}")
                    logger.debug(f"   Vector norm: {wrapped_layer.vec.norm().item():.4f}")
                    logger.debug(f"   Vector preview: {wrapped_layer.vec.data[:10].cpu()}")
                except Exception as e:
                    logger.error(f"❌ Failed to load vector: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                logger.error(f"❌ Vector file not found: {vec_path}")
                logger.error(f"   Please check if the file exists and the epoch number is correct")


                layer_dir = os.path.join(vec_dir, f"layer{script_args.layer}")
                if os.path.exists(layer_dir):
                    logger.error(f"   Available vectors in {layer_dir}:")
                    vec_files = [f for f in sorted(os.listdir(layer_dir)) if f.startswith('vec_ep') and f.endswith('.pt')]
                    for f in vec_files:

                        try:
                            epoch_num = int(f.split('_ep')[1].split('_')[0])
                            logger.error(f"     - Epoch {epoch_num}: {f}")
                        except:
                            logger.error(f"     - {f}")
                else:
                    logger.error(f"   Directory does not exist: {layer_dir}")

                raise FileNotFoundError(f"Cannot find vector file for epoch {script_args.resume_from_epoch}")

        model.language_model.model.layers[script_args.layer] = wrapped_layer
        logger.debug(f"✅ Layer {script_args.layer} wrapped successfully")
        logger.debug(f"   Device: {next(wrapped_layer.parameters()).device}")
    except IndexError:
        logger.error(f"Layer {script_args.layer} is out of bounds. Model has {len(model.language_model.model.layers)} layers.")
        exit(1)
    except AttributeError as e:
        logger.error(f"AttributeError: {e}")
        logger.error("Could not find the layers. Please double-check the model structure path.")
        exit(1)

    model.config.use_cache = False
    if load_kwargs.get("device_map") is None:
        model = model.to("cuda:0")
    print_detailed_memory("包装BlockWrapper后 ")

    logger.debug("✅ 优化：不加载独立的参考模型，使用 multiplier=0 的主模型代替")
    logger.debug("   这将节省约 50% 的显存占用！")
    model_ref = None

    logger.debug("🔒 Freezing non-target parameters in the main model...")
    target_param_name = f'language_model.model.layers.{script_args.layer}.vec'
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if name == target_param_name:
            param.requires_grad = True
            trainable_count += 1
            logger.debug(f"   ✅ Trainable: {name}")
        else:
            param.requires_grad = False
            frozen_count += 1
    logger.debug(f"   Frozen: {frozen_count} params, Trainable: {trainable_count} params")
    logger.debug('✅ Model loading and preparation complete.')
    print_detailed_memory("模型准备完成 ")


    def load_dataset_by_modality(target_modality, train_flag=True):
        if target_modality == "text":
            return get_text_dataset(text_behavior, train_flag)
        if target_modality == "audio":
            return get_audio_dataset(audio_behavior, train_flag, force_empty_prompt=True)
        return build_multimodal_dataset(text_behavior, audio_behavior, train_flag)

    logger.debug("📚 Loading training dataset...")
    print_detailed_memory("加载训练集前 ")
    train_data = load_dataset_by_modality(modality, True)
    print_detailed_memory("加载训练集后 ")

    logger.debug("📚 Loading test dataset...")
    print_detailed_memory("加载测试集前 ")
    test_data = load_dataset_by_modality(modality, False)
    print_detailed_memory("加载测试集后 ")


    logger.debug(f"Train dataset first example: {train_data[0]}")
    logger.debug(f"Test dataset first example: {test_data[0]}")

    logger.debug("⚙️ Initializing training arguments...")
    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=script_args.learning_rate,
        eval_strategy="epoch",
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        max_grad_norm=1.0,
        remove_unused_columns=False,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,

        local_rank=-1,

        precompute_ref_log_probs=False,
    )

    logger.debug("🏃 Initializing CrossSteer text trainer...")
    print_detailed_memory("初始化trainer前 ")


    gc.collect()
    torch.cuda.empty_cache()


    logger.debug("🔧 设置 CUDA_VISIBLE_DEVICES='0' 以防止 DataParallel 复制主模型到 GPU 1")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    import resource

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        logger.debug(f"📊 当前内存限制: soft={soft/(1024**3):.1f}GB, hard={hard/(1024**3):.1f}GB")

        if soft == resource.RLIM_INFINITY:
            resource.setrlimit(resource.RLIMIT_AS, (100 * 1024**3, hard))
            logger.debug("✅ 设置内存软限制为 100GB")
    except Exception as e:
        logger.warning(f"⚠️  无法设置内存限制: {e}")



    dpo_trainer = CrossSteerTextTrainer(
        model,
        ref_model=model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_data,
        eval_dataset={"test_dataset_add": test_data, "test_dataset_sub": test_data},
        processing_class=tokenizer,
        behavior=script_args.behavior,
        layer=script_args.layer,
        name=f"qwen2-audio-{modality}",
        input_modality=modality,
        audio_processor=processor,
        precompute_ref_log_probs=False,
    )

    logger.debug("✅ Trainer initialized successfully")
    print_detailed_memory("初始化trainer后 ")


    if script_args.resume_from_epoch is not None:
        dpo_trainer.epoch_for_saving_vec = script_args.resume_from_epoch
        logger.debug(f"✅ Set trainer starting epoch to {script_args.resume_from_epoch}")
        logger.debug(f"   Next vector will be saved as: vec_ep{script_args.resume_from_epoch + 1}_layer{script_args.layer}.pt")
        logger.debug(f"   Training will continue until epoch {script_args.num_train_epochs}")

    logger.debug("🎯 Starting training...")
    print_trainable_parameters(model)
    print_detailed_memory("训练开始前 ")


    torch.cuda.empty_cache()
    gc.collect()

    try:
        dpo_trainer.train()
        logger.debug("🎉 Training complete!")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        print_detailed_memory("训练报错时 ")
        import traceback
        traceback.print_exc()
        raise e

    logger.debug("💾 Saving the trained steering vector...")
    try:
        steering_vector = dpo_trainer.model.language_model.model.layers[script_args.layer].vec.detach().cpu()
        vector_save_path = os.path.join(
            script_args.output_dir,
            f"steering_vector_layer_{script_args.layer}_{script_args.behavior}.pt"
        )
        os.makedirs(script_args.output_dir, exist_ok=True)
        torch.save(steering_vector, vector_save_path)
        logger.debug(f"✅ Steering vector successfully saved to: {vector_save_path}")
        logger.debug(f"   Vector shape: {steering_vector.shape}")
        logger.debug(f"   Vector norm: {steering_vector.norm().item():.4f}")
    except Exception as e:
        logger.error(f"❌ Failed to save steering vector: {e}")
        import traceback
        traceback.print_exc()


    if use_swanlab:
        swanlab.finish()
        logger.debug("🏁 SwanLab session finished")
    logger.debug("🏁 Script execution completed!")
