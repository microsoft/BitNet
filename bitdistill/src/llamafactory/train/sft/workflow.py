# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
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

from typing import TYPE_CHECKING, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_teacher_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer, CustomDistillSeq2SeqTrainer
import os

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def replace_linear_with_fusedbit(model, standard_bitnet = False, inquant_layers_keys = [], weight_quant_method = 'minmax'):
    from bitnetsrc import core

    for name, module in model.named_modules():
        if 'lm_head' in name: # we always keep the precision of last lm_head  
            continue

        # skip params in inquant_layers_keys
        quant = True
        for key in inquant_layers_keys:
            if key in name:
                quant = False
        if not quant:
            continue

        if isinstance(module, nn.Linear):
            
            if is_main_process(): print("quating", name, "into bitnet ...")

            if standard_bitnet:
                if 'o_proj' in name or 'down_proj' in name:
                    fusedbit_layer = core.BitLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        should_rms = True,
                        bias=(module.bias is not None),
                        weight_quant_method=weight_quant_method
                    )
                else:
                    fusedbit_layer = core.BitLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        should_rms = False,
                        bias=(module.bias is not None),
                        weight_quant_method=weight_quant_method
                    )
            else:
                fusedbit_layer = core.BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    should_rms = True,
                    bias=(module.bias is not None),
                    weight_quant_method=weight_quant_method
                )

            # Copy existing weights/bias
            with torch.no_grad():
                fusedbit_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    fusedbit_layer.bias.copy_(module.bias)

            # Replace the original module in its parent
            parent_path = name.rsplit('.', 1)
            if len(parent_path) == 1:
                setattr(model, parent_path[0], fusedbit_layer)
            else:
                parent_module_name, child_name = parent_path
                parent_module = dict(model.named_modules())[parent_module_name]
                setattr(parent_module, child_name, fusedbit_layer)
    return model

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    if model_args.use_bitnet:
        model = replace_linear_with_fusedbit(model, standard_bitnet=model_args.standard_bitnet, inquant_layers_keys=model_args.inquant_layers_keys, weight_quant_method=model_args.weight_quant_method)
        if is_main_process(): 
            print("===> Student:", model)

    if model_args.distill:
        teacher = load_teacher_model(model_args.teacher_model_path)
        if is_main_process(): print("===> Teacher:", teacher)

    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # Initialize our Trainer
    if model_args.distill:
        trainer = CustomDistillSeq2SeqTrainer(
            model=model,
            teacher=teacher,
            args=training_args,
            model_args=model_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )
    else:
        trainer = CustomSeq2SeqTrainer(
            model=model,
            args=training_args,
            finetuning_args=finetuning_args,
            data_collator=data_collator,
            callbacks=callbacks,
            gen_kwargs=gen_kwargs,
            **dataset_module,
            **tokenizer_module,
            **metric_module,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]

            plot_loss(training_args.output_dir, keys=keys)

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
