# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments

import time
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    XLA_FSDPV2_MIN_VERSION,
    PushInProgress,
    PushToHubMixin,
    can_return_loss,
    check_torch_load_is_safe,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_in_notebook,
    is_ipex_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_schedulefree_available,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    logging,
    strtobool,
)
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from accelerate import Accelerator, skip_first_batches
from accelerate import __version__ as accelerate_version
from accelerate.state import AcceleratorState
from accelerate.utils import (
    AutocastKwargs,
    DistributedDataParallelKwargs,
    DistributedType,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)

logger = logging.get_logger(__name__)



class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


def compute_kd_loss_with_debug(student_outputs, teacher_outputs, temperature, inputs):
    student_logits = student_outputs["logits"]
    teacher_logits = teacher_outputs["logits"]
    
    if len(student_logits.shape) == 3:
        batch_size, seq_len, vocab_size = student_logits.shape
        
        if "attention_mask" in inputs and inputs["attention_mask"] is not None:
            mask = inputs["attention_mask"]

            student_logits_flat = student_logits.view(-1, vocab_size)
            teacher_logits_flat = teacher_logits.view(-1, vocab_size)
            mask_flat = mask.view(-1).bool()

            valid_student_logits = student_logits_flat[mask_flat]
            valid_teacher_logits = teacher_logits_flat[mask_flat]
        else:
            valid_student_logits = student_logits.view(-1, student_logits.size(-1))
            valid_teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
    else:
        valid_student_logits = student_logits
        valid_teacher_logits = teacher_logits
    
    student_log_probs = F.log_softmax(valid_student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(valid_teacher_logits / temperature, dim=-1)
    
    with torch.no_grad():
        student_entropy = -(teacher_probs * student_log_probs).sum(-1).mean()
        teacher_entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(-1).mean()
        # print(f"Student entropy: {student_entropy:.3f}, Teacher entropy: {teacher_entropy:.3f}")
    
    loss_kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    
    # print(f"KL Loss: {loss_kd:.4f}")
    # print("-" * 50)
    
    return loss_kd


class CustomDistillSeq2SeqTrainer(CustomSeq2SeqTrainer):
    def __init__(
        self,
        teacher,
        model_args: "ModelArguments",
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            finetuning_args=finetuning_args,
            processor=processor,
            gen_kwargs=gen_kwargs,
            **kwargs,
        )
        self.teacher = teacher
        self.model_args = model_args
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        
        self.beta = model_args.beta
        self.temperature = model_args.logits_distill_temperature

        self.minilmv2_loss_coeff = model_args.minilmv2_loss_coeff
        self.split_head_number = model_args.split_head_number
        self.layers_used_for_distill = model_args.layers_used_for_distill
        self.distill_to_student_final_layer = model_args.distill_to_student_final_layer

        self.final_loss = 0.
        self.student_ce_loss = 0.
        self.loss_logits_kd = 0.
        self.vv_distill_loss = 0.
        self.logging_steps = 5

    def compute_distill_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        student_outputs = model(**inputs)

        # device = next(model.parameters()).device
        # self.teacher = self.teacher.to(device)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
        # self.teacher = self.teacher.to("cpu")
        torch.cuda.empty_cache()

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(student_outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(student_outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(student_outputs, labels)
        else:
            if isinstance(student_outputs, dict) and "loss" not in student_outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(student_outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = student_outputs["loss"] if isinstance(student_outputs, dict) else student_outputs[0]

        loss_hard = loss
        # logits (KD) loss
        student_logits = student_outputs["logits"]
        teacher_logits = teacher_outputs["logits"]

        loss_logits_kd = compute_kd_loss_with_debug(student_outputs, teacher_outputs, self.temperature, inputs)

        vv_distill_loss = torch.tensor(0.0, device=student_outputs.logits.device)

        if self.minilmv2_loss_coeff != 0.: # uisng minilm distill

            if self.layers_used_for_distill == [-1]:
                num_layers = len(student_outputs["all_key_states"])
                self.layers_used_for_distill = list(range(num_layers))

            assert len(self.layers_used_for_distill) != 0 
            
        if self.is_world_process_zero():
            print("self.layers_used_for_distill", self.layers_used_for_distill, "self.distill_to_student_final_layer", self.distill_to_student_final_layer)

        if self.minilmv2_loss_coeff != 0.:
            compute_keys = ['all_value_states', 'all_key_states', 'all_query_states']

            for compute_key in compute_keys:
                student_value_states = student_outputs[compute_key]
                teacher_value_states = teacher_outputs[compute_key]
                num_layers = len(student_value_states)

                for i in range(num_layers):
                    if i not in self.layers_used_for_distill:
                        continue

                    # with torch.cuda.amp.autocast(enabled=True):
                    if self.distill_to_student_final_layer:
                        assert len(self.layers_used_for_distill) == 1, print("only support distill one teacher's layer to student's last layer now.")
                        s_values = student_value_states[-1]
                    else:
                        s_values = student_value_states[i]  # [B, num_heads, seq_len, head_dim]
                    t_values = teacher_value_states[i]
                    
                    batch_size, num_heads, seq_len, head_dim = s_values.shape

                    # trans to our specific head number
                    # if self.is_world_process_zero():
                    #     print(compute_key, s_values.shape)
                    total_dim = num_heads * head_dim
                    if total_dim % self.split_head_number != 0:
                        raise ValueError(f"{compute_key}: Total dimension ({total_dim}) must be divisible by target number of heads ({self.split_head_number})")

                    new_head_dim = total_dim // self.split_head_number

                    # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
                    s_values = s_values.transpose(1, 2).reshape(batch_size, seq_len, total_dim)
                    # -> [batch_size, seq_len, self.split_head_number, new_head_dim]
                    s_values = s_values.reshape(batch_size, seq_len, self.split_head_number, new_head_dim)
                    # -> [batch_size, self.split_head_number, seq_len, new_head_dim]
                    s_values = s_values.transpose(1, 2)

                    # [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads * head_dim]
                    t_values = t_values.transpose(1, 2).reshape(batch_size, seq_len, total_dim)
                    # -> [batch_size, seq_len, self.split_head_number, new_head_dim]
                    t_values = t_values.reshape(batch_size, seq_len, self.split_head_number, new_head_dim)
                    # -> [batch_size, self.split_head_number, seq_len, new_head_dim]
                    t_values = t_values.transpose(1, 2)

                    s_values_norm = F.normalize(s_values, p=2, dim=-1)
                    t_values_norm = F.normalize(t_values, p=2, dim=-1)

                    # del s_values, t_values
                    # torch.cuda.empty_cache()
                    
                    # [B, num_heads, seq_len, seq_len]
                    s_vv_relation = torch.matmul(s_values_norm, s_values_norm.transpose(-2, -1))
                    t_vv_relation = torch.matmul(t_values_norm, t_values_norm.transpose(-2, -1))

                    # del s_values_norm, t_values_norm
                    
                    temperature = 1.0  # 或者 (head_dim ** 0.5)
                    s_vv_relation = s_vv_relation / temperature
                    t_vv_relation = t_vv_relation / temperature

                    # Reshape: [B, num_heads, seq_len, seq_len] -> [B*num_heads*seq_len, seq_len]
                    s_vv_relation = s_vv_relation.reshape(-1, seq_len)
                    t_vv_relation = t_vv_relation.reshape(-1, seq_len)
                    
                    s_vv_prob = F.softmax(s_vv_relation, dim=-1)
                    t_vv_prob = F.softmax(t_vv_relation, dim=-1)
                    
                    eps = 1e-8
                    s_vv_prob = s_vv_prob.clamp(min=eps)
                    t_vv_prob = t_vv_prob.clamp(min=eps)
                    
                    vv_distill_loss += F.kl_div(
                        torch.log(s_vv_prob),  # student: log(p)
                        t_vv_prob,             # teacher: q
                        reduction="batchmean",
                        log_target=False
                    )

                    # del s_vv_relation, t_vv_relation
                
            vv_distill_loss = vv_distill_loss / (len(self.layers_used_for_distill) * self.split_head_number)

        final_loss = loss_hard + self.beta * loss_logits_kd + self.minilmv2_loss_coeff * vv_distill_loss


        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            final_loss *= self.accelerator.num_processes
            loss_hard *= self.accelerator.num_processes
            loss_logits_kd *=self.accelerator.num_processes
            vv_distill_loss *= self.accelerator.num_processes

        if self.is_world_process_zero():
            self.final_loss += final_loss.detach().cpu().item()
            self.student_ce_loss += loss_hard.detach().cpu().item()
            self.loss_logits_kd += loss_logits_kd.detach().cpu().item()
            self.vv_distill_loss += vv_distill_loss.detach().cpu().item()

        step = getattr(self.state, 'global_step', None)
        if step is not None and step % (self.logging_steps * 10) == 0 and step != 0 and self.is_world_process_zero():
            logs = {
                "distill/all_loss": self.final_loss / self.logging_steps,
                "distill/student_ce_loss": self.student_ce_loss / self.logging_steps,
                "distill/loss_kd": self.loss_logits_kd / self.logging_steps,
                "distill/minilmv2_loss_coeff": self.minilmv2_loss_coeff,
                "distill/vv_distill_loss": self.vv_distill_loss / self.logging_steps,
                "distill/beta": self.beta,
                "distill/temperature": self.temperature,
            }
            self.log(logs)

            self.final_loss = 0.
            self.student_ce_loss = 0.
            self.loss_logits_kd = 0.
            self.vv_distill_loss = 0.

        if return_outputs:
            return (final_loss, student_outputs)
        else:
            return final_loss

    @override
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_distill_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()