# -*- coding: utf-8 -*-
# @Time    : 2024/11/23
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : ChatTTSPlus
# @FileName: train_lora.py

import logging
import math
import os.path
import pdb
import numpy as np
from datetime import timedelta
from tqdm import tqdm
import peft
from accelerate import InitProcessGroupKwargs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from omegaconf import OmegaConf
import warnings
from peft import LoraConfig, get_peft_model
import time
import pybase16384 as b14
from huggingface_hub import hf_hub_download
from transformers.trainer_pt_utils import LabelSmoother
from vector_quantize_pytorch.residual_fsq import GroupedResidualFSQ
from einops import rearrange

from chattts_plus.commons.logger import get_logger
from chattts_plus.commons import constants
from chattts_plus.models.tokenizer import Tokenizer
from chattts_plus.models.dvae import DVAE
from chattts_plus.models.gpt import GPT
from chattts_plus.datasets.base_dataset import BaseDataset
from chattts_plus.datasets.collator import BaseCollator
from chattts_plus.commons import norm

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
AUDIO_EOS_TOKEN_ID: int = 0
AUDIO_PAD_TOKEN_ID: int = AUDIO_EOS_TOKEN_ID

warnings.filterwarnings("ignore")


def get_mel_attention_mask(
        waveform_attention_mask: torch.Tensor,  # (batch_size, time)
        mel_len: int,
):
    batch_size = waveform_attention_mask.size(0)
    mel_attention_mask = torch.ones(
        (batch_size, mel_len),
        device=waveform_attention_mask.device,
    )
    indices = waveform_attention_mask.int().sum(dim=1)  # (batch_size,)
    indices = torch.ceil(indices * mel_len / waveform_attention_mask.size(1)).int()
    for i in range(batch_size):
        mel_attention_mask[i, indices[i]:] = 0
    return mel_attention_mask  # (batch_size, mel_len)


def dvae_quantize(
        quantizer: GroupedResidualFSQ,
        audio_latents: torch.Tensor,  # (batch_size, audio_dim=1024, mel_len / 2)
) -> tuple[torch.Tensor, torch.Tensor]:
    # feat shape (batch_size, mel_len / 2, audio_dim)
    # ind shape (GFSQ.G, batch_size, mel_len / 2, GFSQ.R)
    # num_vq=GFSQ.G*GFSQ.R
    feat, ind = quantizer(audio_latents.transpose(1, 2))
    audio_quantized_latents = feat.transpose(
        1, 2
    )  # (batch_size, audio_dim, mel_len / 2)
    audio_input_ids = rearrange(
        ind, "g b t r ->b t (g r)"
    )  # (batch_size, mel_len / 2, num_vq)
    return audio_quantized_latents, audio_input_ids


def main(cfg):
    output_dir = os.path.join(cfg.output_dir, f"{cfg.exp_name}-{time.time()}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = get_logger("Lora Training", log_file=os.path.join(log_dir, "train.log"))
    logger.info(cfg)
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        kwargs_handlers=kwargs_handlers,
    )
    if accelerator.is_local_main_process:
        from torch.utils.tensorboard import SummaryWriter
        tf_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tf_logs"))

    # load model
    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )
    logger.info(f"weight_dtype: {str(weight_dtype)}")
    logger.info("loading tokenizer >>>")
    tokenizer_kwargs = cfg.MODELS["tokenizer"]["kwargs"]
    model_path_org = tokenizer_kwargs["model_path"]
    model_path_new = os.path.join(constants.CHECKPOINT_DIR, model_path_org.replace("checkpoints/", ""))
    if not os.path.exists(model_path_new):
        logger.info(f"{model_path_new} not exists! Need to download from HuggingFace")
        hf_hub_download(repo_id="2Noise/ChatTTS", subfolder="asset",
                        filename=os.path.basename(model_path_new),
                        local_dir=constants.CHECKPOINT_DIR)
        logger.info(f"download {model_path_new} from 2Noise/ChatTTS")
    tokenizer_kwargs["model_path"] = model_path_new
    tokenizer = Tokenizer(**tokenizer_kwargs)

    # load DVAE encoder
    logger.info("loading DVAE encode >>>")
    dvae_kwargs = cfg.MODELS["dvae_encode"]["kwargs"]
    if not dvae_kwargs["coef"]:
        coef_ = torch.rand(100)
        dvae_kwargs["coef"] = b14.encode_to_string(coef_.numpy().astype(np.float32).tobytes())
        logger.info(f"Set DAVE Coef: {dvae_kwargs['coef']}")
    model_path_org = dvae_kwargs["model_path"]
    model_path_new = os.path.join(constants.CHECKPOINT_DIR, model_path_org.replace("checkpoints/", ""))
    if not os.path.exists(model_path_new):
        logger.info(f"{model_path_new} not exists! Need to download from HuggingFace")
        hf_hub_download(repo_id="2Noise/ChatTTS", subfolder="asset",
                        filename=os.path.basename(model_path_new),
                        local_dir=constants.CHECKPOINT_DIR)
        logger.info(f"download {model_path_new} from 2Noise/ChatTTS")
    dvae_kwargs["model_path"] = model_path_new
    dvae_encoder = DVAE(**dvae_kwargs)
    dvae_encoder.eval().to(accelerator.device, dtype=weight_dtype)
    dvae_encoder.requires_grad_(False)

    # Load GPT
    logger.info("loading GPT model >>>")
    gpt_kwargs = cfg.MODELS["gpt"]["kwargs"]
    model_path_org = gpt_kwargs["model_path"]
    model_path_new = os.path.join(constants.CHECKPOINT_DIR, model_path_org.replace("checkpoints/", ""))
    if not os.path.exists(model_path_new):
        logger.info(f"{model_path_new} not exists! Need to download from HuggingFace")
        hf_hub_download(repo_id="2Noise/ChatTTS", subfolder="asset",
                        filename=os.path.basename(model_path_new),
                        local_dir=constants.CHECKPOINT_DIR)
        logger.info(f"download {model_path_new} from 2Noise/ChatTTS")
    gpt_kwargs["model_path"] = model_path_new
    gpt_model = GPT(**gpt_kwargs)
    gpt_model.to(accelerator.device, dtype=weight_dtype)
    gpt_model.requires_grad_(False)

    # Lora
    logger.info("Setting Lora model >>>")
    lora_cfg = OmegaConf.to_container(cfg.LORA, resolve=True)
    lora_config = LoraConfig(
        r=lora_cfg['lora_r'],
        lora_alpha=lora_cfg['lora_alpha'],
        target_modules=lora_cfg['lora_target_modules'],
        lora_dropout=lora_cfg['lora_dropout']
    )
    peft_model = get_peft_model(gpt_model.gpt, lora_config)
    peft_model.config.use_cache = False
    if cfg.lora_model_path and os.path.exists(cfg.lora_model_path):
        logger.info(f"loading lora weight: {cfg.lora_model_path} >>>")
        peft.set_peft_model_state_dict(peft_model, peft.load_peft_weights(cfg.lora_model_path))
    gpt_model.gpt = peft_model

    if cfg.solver.scale_lr:
        learning_rate = (
                cfg.solver.learning_rate
                * cfg.solver.gradient_accumulation_steps
                * cfg.data.train_bs
                * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate

    # Initialize the optimizer
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, gpt_model.gpt.parameters()))
    logger.info(f"Total trainable params {len(trainable_params)}")
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )

    # dataset
    normalizer_json = os.path.join(constants.CHECKPOINT_DIR, "homophones_map.json")
    if not os.path.exists(normalizer_json):
        logger.warning(f"{normalizer_json} not exists! Need to download from HuggingFace")
        hf_hub_download(repo_id="warmshao/ChatTTSPlus",
                        filename=os.path.basename(normalizer_json),
                        local_dir=constants.CHECKPOINT_DIR)
        logger.info(f"download {normalizer_json} from warmshao/ChatTTSPlus")
    logger.info(f"loading normalizer: {normalizer_json}")
    normalizer = norm.Normalizer(normalizer_json)

    train_dataset = BaseDataset(
        meta_infos=cfg.DATA.meta_infos,
        sample_rate=cfg.DATA.sample_rate,
        num_vq=cfg.DATA.num_vq,
        tokenizer=tokenizer,
        normalizer=normalizer
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.DATA.train_bs, shuffle=True,
        num_workers=min(cfg.DATA.train_bs, 4), drop_last=True, collate_fn=BaseCollator()
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_train_epochs, cfg.solver.min_learning_rate
    )

    # Prepare everything with our `accelerator`.
    (
        gpt_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        gpt_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # Train!
    total_batch_size = (
            cfg.DATA.train_bs
            * accelerator.num_processes
            * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.DATA.train_bs}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(gpt_model):
                text_input_ids: torch.Tensor = batch[
                    "text_input_ids"
                ]  # (batch_size, text_len, num_vq)
                text_attention_mask: torch.Tensor = batch[
                    "text_mask"
                ]  # (batch_size, text_len)
                audio_wavs: torch.Tensor = batch["audio_wavs"]  # (batch_size, time)
                audio_wavs_mask: torch.Tensor = batch["audio_mask"]  # (batch_size, time)

                batch_size = text_input_ids.size(0)
                text_input_ids = text_input_ids.to(accelerator.device, non_blocking=True)
                text_attention_mask = text_attention_mask.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                audio_wavs = audio_wavs.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                audio_wavs_mask = audio_wavs_mask.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                with torch.no_grad():
                    dvae_audio_input_ids = dvae_encoder(audio_wavs, mode="encode").permute(0, 2, 1).clone()
                    mel_attention_mask = get_mel_attention_mask(audio_wavs_mask,
                                                                mel_len=dvae_audio_input_ids.size(1) * 2)
                    audio_attention_mask = mel_attention_mask[:, ::2]
                    dvae_audio_input_ids[~audio_attention_mask.bool()] = AUDIO_PAD_TOKEN_ID

                    # add audio eos token
                    extended_audio_attention_mask = torch.cat(
                        [
                            audio_attention_mask,
                            torch.zeros(
                                (batch_size, 1),
                                dtype=audio_attention_mask.dtype,
                                device=audio_attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )  # (batch_size, mel_len+1)
                    extended_audio_input_ids = torch.cat(
                        [
                            dvae_audio_input_ids,
                            AUDIO_PAD_TOKEN_ID
                            * torch.ones(
                                (batch_size, 1, gpt_model.num_vq),
                                dtype=dvae_audio_input_ids.dtype,
                                device=dvae_audio_input_ids.device,
                            ),
                        ],
                        dim=1,
                    )  # (batch_size, mel_len+1, num_vq)
                    indices = audio_attention_mask.int().sum(dim=1)  # (batch_size,)
                    for i in range(batch_size):
                        extended_audio_attention_mask[i, indices[i]] = 1
                        extended_audio_input_ids[i, indices[i]] = AUDIO_EOS_TOKEN_ID

                    # combine text and audio
                    input_ids = torch.cat(  # (batch_size, text_len + mel_len + 1, num_vq)
                        [
                            text_input_ids,
                            extended_audio_input_ids,  # (batch_size, mel_len, num_vq)
                        ],
                        dim=1,
                    )
                    attention_mask = torch.cat(  # (batch_size, text_len + mel_len + 1)
                        [text_attention_mask, extended_audio_attention_mask],
                        dim=1,
                    )
                    text_mask = torch.cat(  # (batch_size, text_len + mel_len + 1)
                        [
                            torch.ones_like(text_attention_mask, dtype=bool),
                            torch.zeros_like(extended_audio_attention_mask, dtype=bool),
                        ],
                        dim=1,
                    )

                    # set labels
                    labels = input_ids.clone()  # (batch_size, text_len + mel_len + 1, num_vq)
                    labels[~attention_mask.bool()] = IGNORE_TOKEN_ID
                # (batch_size, text_len + mel_len, 768)
                inputs_embeds = gpt_model.forward(input_ids=input_ids, text_mask=text_mask)
                outputs = gpt_model.gpt.forward(inputs_embeds=inputs_embeds.to(dtype=weight_dtype),
                                                attention_mask=attention_mask.to(dtype=weight_dtype))
                hidden_states = outputs.last_hidden_state
                text_len = text_input_ids.size(1)
                audio_hidden_states = hidden_states[
                                      :, text_len - 1: -1
                                      ]  # (batch_size, mel_len+1, 768)
                audio_labels = labels[:, text_len:]
                audio_logits = torch.stack(
                    [
                        gpt_model.head_code[i](audio_hidden_states)
                        for i in range(gpt_model.num_vq)
                    ],
                    dim=2,
                )  # (batch_size, mel_len+1, num_vq, num_class_audio)
                audio_loss: torch.Tensor = torch.nn.functional.cross_entropy(
                    audio_logits.flatten(0, 2), audio_labels.flatten(0, 2)
                )
                loss = audio_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(cfg.DATA.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(
                #         trainable_params,
                #         cfg.solver.max_grad_norm,
                #     )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                if accelerator.is_main_process:
                    tf_writer.add_scalar('train_loss', train_loss, global_step)
                train_loss = 0.0

                if global_step == 1 or global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        unwrap_net = accelerator.unwrap_model(gpt_model)
                        unwrap_net.gpt.save_pretrained(os.path.join(checkpoints_dir, f"step-{global_step}"))
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0]
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/train_voice_clone_lora.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
