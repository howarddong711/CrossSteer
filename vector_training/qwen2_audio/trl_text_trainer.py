
import os

os.environ['HF_DATASETS_DISABLE_PROGRESS_BARS'] = '1'

import time
import json
import math
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from huggingface_hub.utils._deprecation import _deprecate_arguments
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from transformers.debug_utils import DebugOption

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.dpo_config import DPOConfig, FDivergenceConstants, FDivergenceType
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    RunningMoments,
    cap_exp,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class CrossSteerTextTrainer(Trainer):







    _tag_names = ["trl", "dpo", "crosssteer", "text-only"]

    @_deprecate_arguments(
        version="1.0.0",
        deprecated_args=[
            "beta",
            "label_smoothing",
            "loss_type",
            "label_pad_token_id",
            "padding_value",
            "truncation_mode",
            "max_length",
            "max_prompt_length",
            "max_target_length",
            "is_encoder_decoder",
            "disable_dropout",
            "generate_during_eval",
            "precompute_ref_log_probs",
            "dataset_num_proc",
            "model_init_kwargs",
            "ref_model_init_kwargs",
            "model_adapter_name",
            "ref_adapter_name",
            "reference_free",
            "force_use_ref_model",
        ],
        custom_message="Deprecated positional argument(s) used in DPOTrainer, please use the DPOConfig to set these arguments instead.",
    )
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: Literal["sigmoid", "hinge", "ipo", "bco_pair", "robust", "aot", "aot_pair"] = "sigmoid",
        args: Optional[DPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        behavior: str = "power-seeking",
        layer: int = 15,
        name: Optional[str] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
        input_modality: Literal["text", "audio", "multimodal"] = "text",
        audio_processor: Optional[Any] = None,
    ):

        tokenizer = processing_class if processing_class is not None else tokenizer

        if model_init_kwargs is not None:
            warnings.warn(
                "You passed `model_init_kwargs` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_init_kwargs = model_init_kwargs

        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_init_kwargs to the CrossSteerTextTrainer/DPOConfig, but your model is already instantiated."
            )
        else:
            model_init_kwargs = args.model_init_kwargs
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
            model_init_kwargs["torch_dtype"] = torch_dtype

        if ref_model_init_kwargs is not None:
            warnings.warn(
                "You passed `ref_model_init_kwargs` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_model_init_kwargs = ref_model_init_kwargs

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_init_kwargs to the CrossSteerTextTrainer/DPOConfig, but your ref_model is already instantiated."
            )
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            torch_dtype = ref_model_init_kwargs.get("torch_dtype")
            if torch_dtype is not None:
                if isinstance(torch_dtype, str) and torch_dtype != "auto":
                    torch_dtype = getattr(torch, torch_dtype)
                if torch_dtype != "auto" and not isinstance(torch_dtype, torch.dtype):
                    raise ValueError(
                        f"Invalid `torch_dtype` passed to the DPOConfig. Expected a string with either `torch.dtype` or 'auto', but got {torch_dtype}."
                    )
            ref_model_init_kwargs["torch_dtype"] = torch_dtype

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the CrossSteerTextTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the CrossSteerTextTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        if force_use_ref_model:
            warnings.warn(
                "You passed `force_use_ref_model` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.force_use_ref_model = force_use_ref_model

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not args.force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in CrossSteerTextTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True

        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval:
            warnings.warn(
                "You passed `generate_during_eval` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.generate_during_eval = generate_during_eval
        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if is_encoder_decoder is not None:
            warnings.warn(
                "You passed `is_encoder_decoder` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.is_encoder_decoder = is_encoder_decoder
        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder to the CrossSteerTextTrainer/DPOConfig."
            )
        else:
            self.is_encoder_decoder = args.is_encoder_decoder


        self.input_modality = input_modality
        self.is_vision_model = False
        self.is_audio_model = input_modality in {"audio", "multimodal"}
        self.processor = audio_processor if self.is_audio_model else None
        if self.is_audio_model and self.processor is None:
            raise ValueError("Audio or multimodal mode requires a valid audio processor.")
        self.tokenizer = tokenizer

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if model_adapter_name is not None:
            warnings.warn(
                "You passed `model_adapter_name` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.model_adapter_name = model_adapter_name
        self.model_adapter_name = args.model_adapter_name

        if ref_adapter_name is not None:
            warnings.warn(
                "You passed `ref_adapter_name` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.ref_adapter_name = ref_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if reference_free:
            warnings.warn(
                "You passed `reference_free` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.reference_free = reference_free
        self.reference_free = args.reference_free

        if precompute_ref_log_probs is not None:
            warnings.warn(
                "You passed `precompute_ref_log_probs` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.precompute_ref_log_probs = precompute_ref_log_probs



        if ref_model is not None:

            self.ref_model = ref_model
        elif self.is_peft_model:

            self.ref_model = None
        elif ref_model is None:

            self.ref_model = None
        else:


            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")

        if max_length is not None:
            warnings.warn(
                "You passed `max_length` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_length = max_length
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_length = 512

        if max_prompt_length is not None:
            warnings.warn(
                "You passed `max_prompt_length` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_prompt_length = max_prompt_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_prompt_length = 128

        if max_target_length is not None:
            warnings.warn(
                "You passed `max_target_length` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.max_target_length = max_target_length
        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            args.max_target_length = 128

        if label_pad_token_id != -100:
            warnings.warn(
                "You passed `label_pad_token_id` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_pad_token_id = label_pad_token_id
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=self.tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if not disable_dropout:
            warnings.warn(
                "You passed `disable_dropout` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.disable_dropout = disable_dropout
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = args.max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        if padding_value is not None:
            warnings.warn(
                "You passed `padding_value` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.padding_value = padding_value
        self.padding_value = args.padding_value if padding_value is not None else self.tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        if truncation_mode != "keep_end":
            warnings.warn(
                "You passed `truncation_mode` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.truncation_mode = truncation_mode
        self.truncation_mode = args.truncation_mode
        self.behavior = behavior
        self.layer = layer
        self.name = name
        if self.name is None:
            self.vec_dir = f"./vector/{self.behavior}"
        else:
            self.vec_dir = f"./vector/{self.behavior}_{self.name}"
        if not os.path.exists(self.vec_dir):
            os.makedirs(self.vec_dir)
            print('Create vector dir: ', self.vec_dir)
        else:
            print('vector dir: ', self.vec_dir)
        self.max_target_length = args.max_target_length
        self.precompute_ref_log_probs = args.precompute_ref_log_probs

        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type != "sigmoid":
            warnings.warn(
                "You passed `loss_type` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.loss_type = loss_type
        if label_smoothing != 0:
            warnings.warn(
                "You passed `label_smoothing` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.label_smoothing = label_smoothing
        if args.loss_type in ["hinge", "ipo", "bco_pair"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )
        if args.loss_type == "kto_pair":
            raise ValueError("Support for kto_pair has been removed in DPOTrainer. Please use KTOTrainer.")

        if beta != 0.1:
            warnings.warn(
                "You passed `beta` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.beta = beta
        self.beta = args.beta
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.aux_loss_enabled = getattr(model.config, "output_router_logits", False)

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        self.f_divergence_type = args.f_divergence_type
        self.f_divergence_params = {FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY: args.f_alpha_divergence_coef}

        if dataset_num_proc is not None:
            warnings.warn(
                "You passed `dataset_num_proc` to the CrossSteerTextTrainer, the value you passed will override the one in the `DPOConfig`."
            )
            args.dataset_num_proc = dataset_num_proc
        self.dataset_num_proc = args.dataset_num_proc

        self.epoch_for_saving_vec = 0
        self.multiplier_counts = {-1.0: 0, 1.0: 0}
        self._last_saved_vec = None


        print("🔄 开始预处理数据集（避免 Trainer 内部处理）...")
        train_dataset = self._preprocess_dataset(train_dataset, "训练集")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                processed_eval = {}
                for key, dataset in eval_dataset.items():
                    processed_eval[key] = self._preprocess_dataset(dataset, f"验证集-{key}")
                eval_dataset = processed_eval
            else:
                eval_dataset = self._preprocess_dataset(eval_dataset, "验证集")
        print("✅ 数据集预处理完成，现在初始化 Trainer...")



        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


        if hasattr(self, 'model') and isinstance(self.model, torch.nn.DataParallel):
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("🔧 检测到 DataParallel 包装，正在移除...")
            self.model = self.model.module
            logger.info(f"✅ 模型已解包，设备: {next(self.model.parameters()).device}")


        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )


        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:


            if not (self.is_peft_model or self.precompute_ref_log_probs):
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    "⚠️  No reference model provided. Using single-model optimization "
                    "(multiplier=0 mode will be used as reference)."
                )
            if args.sync_ref_model:
                raise ValueError(
                    "You currently cannot use `ref_model=None` with TR-DPO method. Please provide `ref_model`."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model.eval()
                for param in self.ref_model.parameters():
                    param.requires_grad = False

                import logging
                logger = logging.getLogger(__name__)
                ref_device = next(self.ref_model.parameters()).device
                logger.info(f"✅ Reference model on {ref_device}, skipping accelerator.prepare_model")

        if args.sync_ref_model:
            if precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with TR-DPO method. Please set `precompute_ref_log_probs=False`."
                )

            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))
        if self.loss_type == "bco_pair":
            self.running = RunningMoments(self.accelerator)

    def _preprocess_dataset(self, dataset: Dataset, name: str = "数据集") -> Dataset:










        import gc
        from datasets import concatenate_datasets

        dataset_size = len(dataset)
        if self.is_audio_model:
            batch_size = 32
            writer_batch_size = 4
        else:
            batch_size = min(250, dataset_size)
            writer_batch_size = 10

        print(f"   📊 {name}: {dataset_size} 个样本")
        print(f"   🔧 批处理策略: 每批 {batch_size} 样本")

        processed_parts = []
        total_batches = (dataset_size + batch_size - 1) // batch_size


        from tqdm import tqdm
        import logging


        datasets_logger = logging.getLogger("datasets")
        original_level = datasets_logger.level
        datasets_logger.setLevel(logging.ERROR)

        with tqdm(total=dataset_size, desc=f"   处理{name}", unit="样本", ncols=100,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                current_batch_size = end_idx - start_idx


                batch_dataset = dataset.select(range(start_idx, end_idx))


                processed_batch = batch_dataset.map(
                    self.tokenize_row,
                    batched=False,
                    num_proc=None,
                    writer_batch_size=writer_batch_size,
                    remove_columns=batch_dataset.column_names,
                    load_from_cache_file=False,
                    keep_in_memory=False,
                    desc=None,
                )

                processed_parts.append(processed_batch)


                pbar.update(current_batch_size)


                del batch_dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


        datasets_logger.setLevel(original_level)


        print(f"   🔗 合并批次...")
        result_dataset = concatenate_datasets(processed_parts)


        del processed_parts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        original_len = len(result_dataset)
        result_dataset = result_dataset.filter(
            lambda x: x is not None and "prompt_input_ids" in x and len(x.get("prompt_input_ids", [])) > 0,
            desc=None,
            keep_in_memory=False,
        )
        filtered_len = len(result_dataset)

        if filtered_len < original_len:
            print(f"   ⚠️  过滤掉 {original_len - filtered_len} 个无效样本，剩余 {filtered_len}")
        else:
            print(f"   ✅ {name}处理完成: {filtered_len} 个样本")


        if filtered_len > 0:
            sample = result_dataset[0]
            vocab_size = self.tokenizer.vocab_size
            print(f"   🔍 样本检查: prompt长度={len(sample.get('prompt_input_ids', []))}, "
                  f"chosen长度={len(sample.get('chosen_input_ids', []))}, "
                  f"rejected长度={len(sample.get('rejected_input_ids', []))}")
            print(f"   📖 Tokenizer词表大小: {vocab_size}")


            for key in ["prompt_input_ids", "chosen_input_ids", "rejected_input_ids"]:
                if key in sample:
                    ids = sample[key]
                    if len(ids) == 0:
                        print(f"   ⚠️  {key} 为空列表!")
                        continue
                    min_id = min(ids)
                    max_id = max(ids)
                    print(f"   🔢 {key}: min={min_id}, max={max_id}")

                    if max_id >= vocab_size:
                        print(f"   ℹ️  {key} 包含特殊token (ID={max_id})，这在音频模型中是正常的")
                    if min_id < 0:
                        print(f"   ❌ {key} 包含负数token ID={min_id}!")

        gc.collect()
        return result_dataset

    def _wrap_model(self, model, training=True, dataloader=None):



        import logging
        logger = logging.getLogger(__name__)


        if torch.cuda.device_count() > 1 and not isinstance(model, torch.nn.DataParallel):
            logger.warning(f"🚫 检测到 {torch.cuda.device_count()} 个 GPU，但强制禁用 DataParallel")
            logger.warning(f"   主模型在: {next(model.parameters()).device}")
            if hasattr(self, 'ref_model') and self.ref_model is not None:
                logger.warning(f"   参考模型在: {next(self.ref_model.parameters()).device}")

        return model

    def tokenize_row(self, feature, model=None) -> Dict:

        modal_hint = (feature.get("modal_type") or self.input_modality).lower()
        has_audio = bool(feature.get("audio_path"))
        if self.is_audio_model and (modal_hint == "audio" or has_audio):
            return self._tokenize_audio_row(feature)
        return self._tokenize_text_row(feature)

    def _tokenize_text_row(self, feature: Dict) -> Optional[Dict]:
        max_length = self.max_length
        max_prompt_length = self.max_prompt_length

        prompt = feature.get("prompt", "")
        chosen = feature.get("matching", "")
        rejected = feature.get("not_matching", "")

        if not prompt or not chosen or not rejected:
            return None

        try:
            prompt_tokens = self.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors=None,
            )

            chosen_full_text = prompt + " " + chosen
            chosen_tokens = self.tokenizer(
                chosen_full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            rejected_full_text = prompt + " " + rejected
            rejected_tokens = self.tokenizer(
                rejected_full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
                return_tensors=None,
            )

            prompt_len = len(prompt_tokens["input_ids"])

            chosen_labels = chosen_tokens["input_ids"].copy()
            rejected_labels = rejected_tokens["input_ids"].copy()

            mask_len = min(prompt_len, len(chosen_labels))
            chosen_labels[:mask_len] = [-100] * mask_len

            mask_len = min(prompt_len, len(rejected_labels))
            rejected_labels[:mask_len] = [-100] * mask_len

            result_dict = {
                "prompt_input_ids": prompt_tokens["input_ids"],
                "prompt_attention_mask": prompt_tokens["attention_mask"],
                "chosen_input_ids": chosen_tokens["input_ids"],
                "chosen_attention_mask": chosen_tokens["attention_mask"],
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_tokens["input_ids"],
                "rejected_attention_mask": rejected_tokens["attention_mask"],
                "rejected_labels": rejected_labels,
            }

            vocab_size = self.tokenizer.vocab_size
            for key in ["chosen_input_ids", "rejected_input_ids", "prompt_input_ids"]:
                if key in result_dict:
                    ids = result_dict[key]
                    max_id = max(ids) if ids else 0
                    if max_id >= vocab_size:
                        print(f"⚠️  警告: {key} 包含超出词表的 token ID {max_id} (vocab_size={vocab_size})")
                        result_dict[key] = [min(i, vocab_size - 1) for i in ids]

            return result_dict

        except Exception as e:
            prompt_preview = feature.get("prompt", "")[:50]
            print(f"❌ 处理文本样本时出错 (prompt: {prompt_preview}...): {str(e)[:100]}")
            return None

    def _tokenize_audio_row(self, feature: Dict) -> Optional[Dict]:
        if self.processor is None:
            print("❌ Audio processor not available,无法处理音频样本")
            return None

        max_length = self.max_length
        max_prompt_length = self.max_prompt_length

        audio_path = feature.get("audio_path", "")
        chosen = feature.get("matching", "")
        rejected = feature.get("not_matching", "")
        prompt_text = feature.get("prompt", "").strip()
        if self.input_modality == "audio":

            prompt_text = ""

        if not audio_path or not chosen or not rejected:
            return None

        if not os.path.exists(audio_path):
            print(f"❌ 音频文件不存在: {audio_path}")
            return None

        try:
            waveform, sampling_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)


            expected_sr = None
            try:
                if hasattr(self.processor, "feature_extractor") and hasattr(self.processor.feature_extractor, "sampling_rate"):
                    expected_sr = int(self.processor.feature_extractor.sampling_rate)
                elif hasattr(self.processor, "sampling_rate"):
                    expected_sr = int(self.processor.sampling_rate)
            except Exception:
                expected_sr = None

            if expected_sr is not None and int(sampling_rate) != expected_sr:
                try:
                    waveform = torchaudio.functional.resample(waveform, int(sampling_rate), expected_sr)
                    sampling_rate = expected_sr
                except Exception as e:
                    print(
                        f"❌ 音频重采样失败 (路径: {audio_path}): {str(e)[:120]} "
                        f"(sr={sampling_rate} -> {expected_sr})"
                    )
                    return None

            waveform_np = waveform.squeeze(0).numpy()

            base_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>"
            if prompt_text:
                prompt_text = prompt_text + "\n" + base_prompt
            else:
                prompt_text = base_prompt

            audio_processor = self.processor

            def _call_audio_processor(text: str, max_len: int):







                common_kwargs = dict(
                    text=text,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )


                try:
                    out = audio_processor(audio=waveform_np, **common_kwargs)
                    if out is not None and out.get("input_features") is not None:
                        return out
                except TypeError:
                    pass


                try:
                    out = audio_processor(audios=waveform_np, **common_kwargs)
                    if out is not None and out.get("input_features") is not None:
                        return out
                except TypeError:
                    pass


                try:
                    out = audio_processor(audio=[waveform_np], **common_kwargs)
                    if out is not None and out.get("input_features") is not None:
                        return out
                except TypeError:
                    pass
                try:
                    out = audio_processor(audios=[waveform_np], **common_kwargs)
                    if out is not None and out.get("input_features") is not None:
                        return out
                except TypeError:
                    pass


                return out


            chosen_text = prompt_text + " " + chosen
            chosen_inputs = _call_audio_processor(chosen_text, max_length)

            rejected_text = prompt_text + " " + rejected
            rejected_inputs = _call_audio_processor(rejected_text, max_length)

            prompt_only_inputs = _call_audio_processor(prompt_text, max_prompt_length)
            prompt_input_ids = prompt_only_inputs["input_ids"][0].tolist()
            prompt_attention_mask = prompt_only_inputs["attention_mask"][0].tolist()
            prompt_len = len(prompt_input_ids)


            chosen_input_ids = chosen_inputs["input_ids"][0].tolist()
            chosen_attention_mask = chosen_inputs["attention_mask"][0].tolist()
            chosen_labels = chosen_input_ids.copy()
            chosen_labels[:prompt_len] = [-100] * min(prompt_len, len(chosen_labels))

            rejected_input_ids = rejected_inputs["input_ids"][0].tolist()
            rejected_attention_mask = rejected_inputs["attention_mask"][0].tolist()
            rejected_labels = rejected_input_ids.copy()
            rejected_labels[:prompt_len] = [-100] * min(prompt_len, len(rejected_labels))


            input_features = chosen_inputs.get("input_features")
            feature_attention_mask = chosen_inputs.get("feature_attention_mask")

            result_dict = {
                "prompt_input_ids": prompt_input_ids,
                "prompt_attention_mask": prompt_attention_mask,
                "chosen_input_ids": chosen_input_ids,
                "chosen_attention_mask": chosen_attention_mask,
                "chosen_labels": chosen_labels,
                "rejected_input_ids": rejected_input_ids,
                "rejected_attention_mask": rejected_attention_mask,
                "rejected_labels": rejected_labels,
            }

            if input_features is not None:
                result_dict["input_features"] = input_features[0].tolist()
                if feature_attention_mask is not None:
                    result_dict["feature_attention_mask"] = feature_attention_mask[0].tolist()

            import gc
            del waveform, waveform_np, chosen_inputs, rejected_inputs, prompt_only_inputs
            gc.collect()

            return result_dict
        except Exception as e:
            print(f"❌ 音频tokenization失败 (路径: {audio_path}): {str(e)[:100]}")
            return None

    def _pad_tokens_to_max_length(self, tokens: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:



        if max_length is None:
            return tokens

        current_length = tokens.shape[1]

        if current_length >= max_length:
            return tokens[:, :max_length]

        pad_length = max_length - current_length

        pad_shape = list(tokens.shape)
        pad_shape[1] = pad_length


        if (tokens == self.label_pad_token_id).any():
            pad_value = self.label_pad_token_id
        else:
            pad_value = self.padding_value

        padding = torch.full(
            pad_shape,
            pad_value,
            dtype=tokens.dtype,
            device=tokens.device
        )

        return torch.cat([tokens, padding], dim=1)

    def concatenated_inputs(
        self,
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_peft_model: bool = False,
        is_encoder_decoder: bool = False,
        is_vision_model: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[str] = None,
    ) -> Dict[str, torch.LongTensor]:



        concatenated_batch = {}


        processed_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 1:
                    processed_batch[k] = v.unsqueeze(0)
                else:
                    processed_batch[k] = v
            else:
                processed_batch[k] = v


        chosen_len = processed_batch["chosen_input_ids"].shape[1]
        rejected_len = processed_batch["rejected_input_ids"].shape[1]
        max_length = max(chosen_len, rejected_len)


        chosen_input_ids = self._pad_tokens_to_max_length(
            processed_batch["chosen_input_ids"],
            max_length
        )
        chosen_attention_mask = self._pad_tokens_to_max_length(
            processed_batch["chosen_attention_mask"],
            max_length
        )
        chosen_labels = self._pad_tokens_to_max_length(
            processed_batch["chosen_labels"],
            max_length
        )


        rejected_input_ids = self._pad_tokens_to_max_length(
            processed_batch["rejected_input_ids"],
            max_length
        )
        rejected_attention_mask = self._pad_tokens_to_max_length(
            processed_batch["rejected_attention_mask"],
            max_length
        )
        rejected_labels = self._pad_tokens_to_max_length(
            processed_batch["rejected_labels"],
            max_length
        )


        concatenated_batch["concatenated_input_ids"] = torch.cat(
            [chosen_input_ids, rejected_input_ids],
            dim=0,
        ).to(device=device)

        concatenated_batch["concatenated_attention_mask"] = torch.cat(
            [chosen_attention_mask, rejected_attention_mask],
            dim=0,
        ).to(device=device)

        concatenated_batch["concatenated_labels"] = torch.cat(
            [chosen_labels, rejected_labels],
            dim=0,
        ).to(device=device)



        if "input_features" in processed_batch and processed_batch["input_features"] is not None:
            input_features = processed_batch["input_features"]
            if not isinstance(input_features, torch.Tensor):
                input_features = torch.tensor(input_features, dtype=torch.float32)
            if input_features.ndim == 2:
                input_features = input_features.unsqueeze(0)
            if device is not None:
                input_features = input_features.to(device=device)



            expected_mel_len = getattr(self, "expected_mel_len", 3000)
            if input_features.ndim == 3:
                cur_len = int(input_features.shape[-1])
                if cur_len < expected_mel_len:
                    pad_len = expected_mel_len - cur_len
                    input_features = F.pad(input_features, (0, pad_len), mode="constant", value=0.0)
                elif cur_len > expected_mel_len:
                    input_features = input_features[..., :expected_mel_len]

            concatenated_batch["input_features"] = input_features.repeat_interleave(2, dim=0)

            feature_attention_mask = processed_batch.get("feature_attention_mask")
            if feature_attention_mask is not None:
                if not isinstance(feature_attention_mask, torch.Tensor):
                    feature_attention_mask = torch.tensor(feature_attention_mask, dtype=torch.long)
                if feature_attention_mask.ndim == 1:
                    feature_attention_mask = feature_attention_mask.unsqueeze(0)
                if device is not None:
                    feature_attention_mask = feature_attention_mask.to(device=device)


                cur_len = int(feature_attention_mask.shape[-1])
                if cur_len < expected_mel_len:
                    pad_len = expected_mel_len - cur_len
                    feature_attention_mask = F.pad(feature_attention_mask, (0, pad_len), mode="constant", value=0)
                elif cur_len > expected_mel_len:
                    feature_attention_mask = feature_attention_mask[..., :expected_mel_len]
                concatenated_batch["feature_attention_mask"] = feature_attention_mask.repeat_interleave(2, dim=0)

        if is_encoder_decoder:
            chosen_decoder_len = processed_batch["chosen_decoder_input_ids"].shape[1]
            rejected_decoder_len = processed_batch["rejected_decoder_input_ids"].shape[1]
            max_decoder_length = max(chosen_decoder_len, rejected_decoder_len)

            concatenated_batch["concatenated_decoder_input_ids"] = torch.cat(
                [
                    self._pad_tokens_to_max_length(processed_batch["chosen_decoder_input_ids"], max_decoder_length),
                    self._pad_tokens_to_max_length(processed_batch["rejected_decoder_input_ids"], max_decoder_length)
                ],
                dim=0
            ).to(device=device)

            concatenated_batch["concatenated_decoder_attention_mask"] = torch.cat(
                [
                    self._pad_tokens_to_max_length(processed_batch["chosen_decoder_attention_mask"], max_decoder_length),
                    self._pad_tokens_to_max_length(processed_batch["rejected_decoder_attention_mask"], max_decoder_length)
                ],
                dim=0,
            ).to(device=device)

        return concatenated_batch

    @contextmanager
    def null_ref_context(self):

        with self.accelerator.unwrap_model(
            self.model
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:

        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        multiplier: float = 1.0,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        chosen_logratios = policy_chosen_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_chosen_logps.to(self.accelerator.device)
        rejected_logratios = policy_rejected_logps.to(self.accelerator.device) - (
            not self.reference_free
        ) * reference_rejected_logps.to(self.accelerator.device)

        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            pi_logratios = policy_chosen_logps - policy_rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
            else:
                ref_logratios = reference_chosen_logps - reference_rejected_logps

            pi_logratios = pi_logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = pi_logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)

        if multiplier < 0:
            logits = -logits

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "robust":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                + F.logsigmoid(-self.beta * logits) * self.label_smoothing
            ) / (1 - 2 * self.label_smoothing)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "bco_pair":
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
            self.running.update(rewards)
            delta = self.running.mean

            losses = -F.logsigmoid((self.beta * chosen_logratios) - delta) - F.logsigmoid(
                -(self.beta * rejected_logratios - delta)
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'bco_pair', 'robust']"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:

        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id


        labels[labels == label_pad_token_id] = 0


        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.error(f"❌ NaN/Inf detected in logits!")
            logger.error(f"   logits shape: {logits.shape}")
            logger.error(f"   logits min: {logits.min().item()}, max: {logits.max().item()}")
            logger.error(f"   NaN count: {torch.isnan(logits).sum().item()}")
            logger.error(f"   Inf count: {torch.isinf(logits).sum().item()}")

            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:



        if model is self.ref_model:
            target_device = next(model.parameters()).device
        else:
            target_device = self.accelerator.device

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            is_vision_model=False,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=target_device,
        )




        if self.input_modality == "audio":
            if "input_features" not in concatenated_batch:
                raise RuntimeError(
                    "[CrossSteer][ASSERT] input_modality='audio' but 'input_features' is missing in concatenated_batch. "
                    "This means audio features were dropped before model forward (training is not conditioning on audio)."
                )
            if concatenated_batch["input_features"] is None:
                raise RuntimeError(
                    "[CrossSteer][ASSERT] input_modality='audio' but 'input_features' is None."
                )
            if concatenated_batch["input_features"].shape[0] != concatenated_batch["concatenated_input_ids"].shape[0]:
                raise RuntimeError(
                    "[CrossSteer][ASSERT] Batch size mismatch: input_features batch != input_ids batch. "
                    f"input_features.shape={tuple(concatenated_batch['input_features'].shape)}, "
                    f"input_ids.shape={tuple(concatenated_batch['concatenated_input_ids'].shape)}"
                )


            if not hasattr(self, "_printed_audio_feature_debug"):
                self._printed_audio_feature_debug = True
                try:
                    print(
                        "[CrossSteer][AUDIO_DEBUG] input_features:",
                        tuple(concatenated_batch["input_features"].shape),
                        concatenated_batch["input_features"].dtype,
                        concatenated_batch["input_features"].device,
                    )
                    if "feature_attention_mask" in concatenated_batch:
                        fam = concatenated_batch["feature_attention_mask"]
                        print(
                            "[CrossSteer][AUDIO_DEBUG] feature_attention_mask:",
                            tuple(fam.shape),
                            fam.dtype,
                            fam.device,
                        )
                except Exception:

                    pass

        len_chosen = concatenated_batch["concatenated_input_ids"].shape[0] // 2

        model_kwargs = {}

        if self.is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = concatenated_batch.pop("concatenated_decoder_input_ids", None)

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True


        if "input_features" in concatenated_batch:
            model_kwargs["input_features"] = concatenated_batch["input_features"]
        if "feature_attention_mask" in concatenated_batch:
            model_kwargs["feature_attention_mask"] = concatenated_batch["feature_attention_mask"]

        outputs = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        )

        all_logits = outputs.logits

        if all_logits.shape[:2] != concatenated_batch["concatenated_labels"].shape[:2]:
            seq_len = concatenated_batch["concatenated_labels"].shape[1]
            all_logits = all_logits[:, -seq_len:]

        all_logps, size_completion = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        def cross_entropy_loss(logits, labels):
            if not self.is_encoder_decoder:
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            return loss

        labels = concatenated_batch["concatenated_labels"].clone()
        nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        if self.loss_type == "ipo":
            all_logps = all_logps / size_completion

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)

    def get_batch_loss_metrics(self, model, inputs, train_eval="train"):







        metrics = {}


        unwrapped_model = model.module if isinstance(model, torch.nn.DataParallel) else model



        with torch.no_grad():
            if self.ref_model is None:
                if self.is_peft_model:

                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_output = self.concatenated_forward(self.model, inputs)
                else:

                    wrapped_layer = unwrapped_model.language_model.model.layers[self.layer]


                    wrapped_layer.use_reference_mode(True)
                    wrapped_layer.set_multiplier(0.0)

                    ref_output = self.concatenated_forward(self.model, inputs)


                    wrapped_layer.use_reference_mode(False)
            else:
                ref_output = self.concatenated_forward(self.ref_model, inputs)

            if len(ref_output) >= 5:
                reference_chosen_logps = ref_output[0]
                reference_rejected_logps = ref_output[1]
            else:
                reference_chosen_logps, reference_rejected_logps = ref_output[:2]

        del ref_output
        if train_eval == "train":
            torch.cuda.empty_cache()


        unwrapped_model.language_model.model.layers[self.layer].set_multiplier(1.0)

        forward_output_add = self.concatenated_forward(model, inputs)
        if len(forward_output_add) >= 4:
            policy_chosen_logps_add = forward_output_add[0]
            policy_rejected_logps_add = forward_output_add[1]
            policy_chosen_logits = forward_output_add[2]
            policy_rejected_logits = forward_output_add[3]

        losses_add, chosen_rewards_add, rejected_rewards_add = self.dpo_loss(
            policy_chosen_logps_add,
            policy_rejected_logps_add,
            reference_chosen_logps,
            reference_rejected_logps,
            multiplier=1.0,
        )

        del forward_output_add
        if train_eval == "train":
            torch.cuda.empty_cache()


        unwrapped_model.language_model.model.layers[self.layer].set_multiplier(-1.0)

        forward_output_sub = self.concatenated_forward(model, inputs)
        if len(forward_output_sub) >= 4:
            policy_chosen_logps_sub = forward_output_sub[0]
            policy_rejected_logps_sub = forward_output_sub[1]

        losses_sub, chosen_rewards_sub, rejected_rewards_sub = self.dpo_loss(
            policy_chosen_logps_sub,
            policy_rejected_logps_sub,
            reference_chosen_logps,
            reference_rejected_logps,
            multiplier=-1.0,
        )

        del forward_output_sub
        if train_eval == "train":
            torch.cuda.empty_cache()


        unwrapped_model.language_model.model.layers[self.layer].set_multiplier(0.0)


        loss = losses_add.mean() + losses_sub.mean()


        reward_accuracies = (chosen_rewards_add > rejected_rewards_add).float()


        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards_add.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards_add.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards_add - rejected_rewards_add).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps_add.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps_add.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()


        metrics[f"{prefix}rewards/chosen_add"] = chosen_rewards_add.mean().cpu()
        metrics[f"{prefix}rewards/rejected_add"] = rejected_rewards_add.mean().cpu()
        metrics[f"{prefix}rewards/chosen_sub"] = chosen_rewards_sub.mean().cpu()
        metrics[f"{prefix}rewards/rejected_sub"] = rejected_rewards_sub.mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        loss = loss.to(self.args.device)
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        print('Enter customized evaluation_loop (text mode)...')
        print('multiplier: ', self.model.language_model.model.layers[self.layer].multiplier)
        print('multiplier_counts: ', self.multiplier_counts)

        if self.model.language_model.model.layers[self.layer].get_multiplier() >= 0:
            self.epoch_for_saving_vec += 1

            steer_vec = self.model.language_model.model.layers[self.layer].vec.detach().cpu().to(torch.float32)

            save_path = f"{self.vec_dir}/layer{self.layer}/vec_ep{self.epoch_for_saving_vec}_layer{self.layer}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(steer_vec, save_path)

            print(f'✅ Saved steering vector at epoch {self.epoch_for_saving_vec}')
            print(f'   Path: {save_path}')
            print(f'   Vec preview: {steer_vec[:10]}')
            print(f'   Vec dtype: {steer_vec.dtype}')
            print(f'   Vec norm: {steer_vec.norm().item():.4f}')


            vec_norm = steer_vec.norm().item()
            vec_l1 = steer_vec.abs().sum().item()
            vec_mean = steer_vec.mean().item()
            vec_std = steer_vec.std(unbiased=False).item()
            vec_min = steer_vec.min().item()
            vec_max = steer_vec.max().item()
            vec_max_abs = steer_vec.abs().max().item()

            delta_norm = None
            cosine_to_prev = None
            prev_norm = None
            if self._last_saved_vec is not None:
                prev = self._last_saved_vec
                delta = steer_vec - prev
                delta_norm = delta.norm().item()
                prev_norm = prev.norm().item()
                if vec_norm > 0.0 and prev_norm > 0.0:
                    cosine_to_prev = float((steer_vec * prev).sum().item() / (vec_norm * prev_norm))

            stats_path = f"{self.vec_dir}/layer{self.layer}/vec_stats.jsonl"
            stats = {
                "epoch": int(self.epoch_for_saving_vec),
                "global_step": int(getattr(self.state, "global_step", 0)),
                "layer": int(self.layer),
                "vec_norm": float(vec_norm),
                "vec_l1": float(vec_l1),
                "vec_mean": float(vec_mean),
                "vec_std": float(vec_std),
                "vec_min": float(vec_min),
                "vec_max": float(vec_max),
                "vec_max_abs": float(vec_max_abs),
                "delta_norm": float(delta_norm) if delta_norm is not None else None,
                "prev_norm": float(prev_norm) if prev_norm is not None else None,
                "cosine_to_prev": float(cosine_to_prev) if cosine_to_prev is not None else None,
            }
            with open(stats_path, "a", encoding="utf-8") as stats_file:
                stats_file.write(json.dumps(stats, ensure_ascii=True) + "\n")
            print(f"   Vec stats: {stats_path}")
            self._last_saved_vec = steer_vec

        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset

        if isinstance(eval_dataset, dict):
            metrics = {}
            unwrapped_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                print('Eval_dataset_name: ', eval_dataset_name)
                if 'add' in eval_dataset_name:
                    unwrapped_model.language_model.model.layers[self.layer].set_multiplier(1.0)
                    print(f'set_multiplier at layer {self.layer} 1.0')
                elif 'sub' in eval_dataset_name:
                    unwrapped_model.language_model.model.layers[self.layer].set_multiplier(-1.0)
                    print(f'set_multiplier at layer {self.layer} -1.0')
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        self.log(output.metrics)
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None, **kwargs) -> None:

        train_eval = "train" if "loss" in logs else "eval"

        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        return super().log(logs, start_time=start_time, **kwargs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:

        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
