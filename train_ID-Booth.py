#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import gc
import logging
import math
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
try:
    from huggingface_hub.utils import insecure_hashlib
except ImportError:
    import hashlib as insecure_hashlib
from packaging import version
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
try:
    from diffusers import FluxPriorReduxPipeline
except ImportError:
    FluxPriorReduxPipeline = None
try:
    from diffusers import BitsAndBytesConfig
except ImportError:
    try:
        from diffusers.quantizers.quantization_config import BitsAndBytesConfig
    except Exception:
        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            BitsAndBytesConfig = None
from diffusers.optimization import get_scheduler
try:
    from diffusers.training_utils import cast_training_params
except ImportError:
    def cast_training_params(models, dtype=torch.float32):
        if not isinstance(models, list):
            models = [models]

        for model in models:
            if model is None:
                continue

            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(dtype=dtype)

from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
)
from diffusers.utils.torch_utils import is_compiled_module

from ArcFace_files.ArcFace_functions import preprocess_image_for_ArcFace, prepare_locked_ArcFace_model
#from ArcFace_dataset import ArcFaceDataset, collate_fn_arcface
import re 
from facenet_pytorch import MTCNN

import configs.config_train_SD21 as cfg
from utils.sorting_utils import natural_keys
import json 


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)

def dict_from_module(module):
    context = {}
    for setting in dir(module):
        # you can write your filter here
        if setting.islower() and setting.isalpha():
            context[setting] = getattr(module, setting)

    return context


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {cfg.num_validation_images} images with prompt:"
        f" {cfg.validation_prompt}."
    )
    use_inference_offload = bool(
        getattr(cfg, "inference_model_cpu_offload", False)
        or getattr(cfg, "inference_sequential_cpu_offload", False)
    )
    if not use_inference_offload:
        pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator_device = "cpu" if use_inference_offload else accelerator.device
    generator = torch.Generator(device=generator_device).manual_seed(cfg.seed) if cfg.seed else None

    phase_name = "test" if is_final_validation else "validation"
    images = []
    is_flux = isinstance(pipeline, FluxPipeline)
    for i in range(cfg.num_validation_images):
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            if is_flux:
                image = pipeline(
                    **pipeline_args,
                    generator=generator,
                    height=cfg.resolution,
                    width=cfg.resolution,
                    num_inference_steps=getattr(cfg, "validation_num_inference_steps", 28),
                    guidance_scale=getattr(cfg, "guidance_scale", 3.5),
                    max_sequence_length=getattr(cfg, "max_sequence_length", 512),
                ).images[0]
            else:
                image = pipeline(**pipeline_args, generator=generator).images[0]
            images.append(image)
            folder_path = os.path.join(args.output_dir, phase_name)
            os.makedirs(folder_path,exist_ok=True)
            image_filename = f"{folder_path}/{epoch}_validation_img_{i}.jpg"
            image.save(image_filename)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")

    del pipeline
    torch.cuda.empty_cache()

    return images


def build_flux_inference_pipeline(accelerator, weight_dtype, lora_dir=None, lora_weight_name=None):
    inference_quantize_transformer_nf4 = getattr(cfg, "inference_quantize_transformer_nf4", True)
    inference_model_cpu_offload = getattr(cfg, "inference_model_cpu_offload", True)
    inference_sequential_cpu_offload = getattr(cfg, "inference_sequential_cpu_offload", False)
    inference_vae_tiling = getattr(cfg, "inference_vae_tiling", True)
    inference_vae_slicing = getattr(cfg, "inference_vae_slicing", True)

    transformer = None
    if inference_quantize_transformer_nf4:
        if BitsAndBytesConfig is None:
            raise ImportError("BitsAndBytesConfig is required for inference_quantize_transformer_nf4=True")
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=weight_dtype,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            cfg.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=cfg.revision,
            variant=cfg.variant,
            quantization_config=nf4_config,
            torch_dtype=weight_dtype,
        )

    pipeline = FluxPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        transformer=transformer,
        text_encoder=None,
        text_encoder_2=None,
        revision=cfg.revision,
        variant=cfg.variant,
        torch_dtype=weight_dtype,
    )

    if lora_dir is not None:
        load_kwargs = {}
        if lora_weight_name is not None:
            load_kwargs["weight_name"] = lora_weight_name
        pipeline.load_lora_weights(lora_dir, **load_kwargs)

    if inference_vae_slicing and hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.enable_slicing()
    if inference_vae_tiling and hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.enable_tiling()

    if inference_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif inference_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to(accelerator.device)

    return pipeline


def configure_flux_inference_memory(pipeline, accelerator):
    inference_model_cpu_offload = getattr(cfg, "inference_model_cpu_offload", True)
    inference_sequential_cpu_offload = getattr(cfg, "inference_sequential_cpu_offload", False)
    inference_vae_tiling = getattr(cfg, "inference_vae_tiling", True)
    inference_vae_slicing = getattr(cfg, "inference_vae_slicing", True)

    if inference_vae_slicing and hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.enable_slicing()
    if inference_vae_tiling and hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.enable_tiling()

    if inference_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif inference_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    else:
        pipeline.to(accelerator.device)


def prepare_flux_validation_kwargs(validation_flux_prompt, accelerator, weight_dtype):
    use_inference_offload = bool(
        getattr(cfg, "inference_model_cpu_offload", False)
        or getattr(cfg, "inference_sequential_cpu_offload", False)
    )
    inference_device = torch.device("cpu") if use_inference_offload else accelerator.device
    pipeline_args = {
        "prompt_embeds": validation_flux_prompt["prompt_embeds"].to(device=inference_device, dtype=weight_dtype),
        "pooled_prompt_embeds": validation_flux_prompt["pooled_prompt_embeds"].to(
            device=inference_device, dtype=weight_dtype
        ),
    }

    if getattr(cfg, "inference_use_redux", False):
        validation_reference_image = getattr(cfg, "validation_reference_image", None)
        if FluxPriorReduxPipeline is None:
            logger.warning("inference_use_redux=True but FluxPriorReduxPipeline is unavailable. Falling back to text-only validation.")
            return pipeline_args
        if not validation_reference_image:
            logger.warning("inference_use_redux=True but validation_reference_image is not set. Falling back to text-only validation.")
            return pipeline_args

        redux_prior = FluxPriorReduxPipeline.from_pretrained(
            getattr(cfg, "redux_model_name_or_path", "black-forest-labs/FLUX.1-Redux-dev"),
            torch_dtype=weight_dtype,
        )
        if getattr(cfg, "inference_model_cpu_offload", True) or getattr(cfg, "inference_sequential_cpu_offload", False):
            redux_prior.enable_model_cpu_offload()
        else:
            redux_prior.to(accelerator.device)

        reference = Image.open(validation_reference_image).convert("RGB")
        prior_output = redux_prior(reference, prompt=cfg.validation_prompt)
        pipeline_args.update(prior_output)
        del redux_prior

    return pipeline_args


def convert_peft_state_dict_for_flux(state_dict):
    converted = {}
    peft_prefix = "base_model.model."
    for key, value in state_dict.items():
        stripped_key = key[len(peft_prefix):] if key.startswith(peft_prefix) else key
        converted[f"transformer.{stripped_key}"] = value
    return converted


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--test_run",
        action="store_true",
        help="Run a short smoke test: 1 epoch on 1 identity subfolder.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != cfg.local_rank:
        cfg.local_rank = env_local_rank
    
    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        instance_prompt,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        flux_prompt_embeds=None,
        flux_pooled_prompt_embeds=None,
        flux_text_ids=None,
        class_flux_prompt_embeds=None,
        class_flux_pooled_prompt_embeds=None,
        class_flux_text_ids=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.class_prompt_encoder_hidden_states = class_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.flux_prompt_embeds = flux_prompt_embeds
        self.flux_pooled_prompt_embeds = flux_pooled_prompt_embeds
        self.flux_text_ids = flux_text_ids
        self.class_flux_prompt_embeds = class_flux_prompt_embeds
        self.class_flux_pooled_prompt_embeds = class_flux_pooled_prompt_embeds
        self.class_flux_text_ids = class_flux_text_ids

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        def collect_files(root_dir, allowed_extensions=None):
            files = []
            for path in Path(root_dir).rglob("*"):
                if not path.is_file():
                    continue
                if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
                    continue
                files.append(path)
            return sorted(files, key=lambda p: str(p))

        self.instance_images_path = collect_files(self.instance_data_root, image_extensions)
        #self.instance_images_path.sort(key=natural_keys)
        #print("IMGS", self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        self.instance_identity_embeds_path = collect_files(Path(instance_data_root.replace("images", "ArcFace_embeds")))
        if len(self.instance_images_path) == 0:
            raise ValueError(f"No images found in instance_data_root: {self.instance_data_root}")
        if len(self.instance_identity_embeds_path) < len(self.instance_images_path):
            raise ValueError(
                "Not enough instance ArcFace embeds. "
                f"Found {len(self.instance_identity_embeds_path)} embeds for {len(self.instance_images_path)} images."
            )
        
        
        #self.instance_identity_embeds_path.sort(key=natural_keys)
        #print("IDS", self.instance_identity_embeds_path)
        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = collect_files(self.class_data_root, image_extensions)
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
                #print(".")
            else:
                self.num_class_images = len(self.class_images_path)
                
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
            self.class_identity_embeds_path = collect_files(Path(class_data_root.replace("images", "ArcFace_embeds")))
            if len(self.class_images_path) == 0:
                raise ValueError(f"No class images found in class_data_root: {self.class_data_root}")
            if len(self.class_identity_embeds_path) < self.num_class_images:
                raise ValueError(
                    "Not enough class ArcFace embeds. "
                    f"Found {len(self.class_identity_embeds_path)} embeds for {self.num_class_images} class images."
                )
        else:
            self.class_data_root = None


        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.flux_prompt_embeds is not None:
            example["instance_prompt_embeds"] = self.flux_prompt_embeds
            example["instance_pooled_prompt_embeds"] = self.flux_pooled_prompt_embeds
            example["instance_text_ids"] = self.flux_text_ids
        elif self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = tokenize_prompt(
                self.tokenizer, self.instance_prompt, tokenizer_max_length=self.tokenizer_max_length
            )
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask

        #print("Index:", index)
        #print("Self ID embeds", len(self.instance_identity_embeds_path))
        #print(index % self.num_instance_images)
        example["instance_identity_embeds"] = torch.load(self.instance_identity_embeds_path[index % self.num_instance_images]) 
        
        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.class_flux_prompt_embeds is not None:
                example["class_prompt_embeds"] = self.class_flux_prompt_embeds
                example["class_pooled_prompt_embeds"] = self.class_flux_pooled_prompt_embeds
                example["class_text_ids"] = self.class_flux_text_ids
            elif self.class_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.class_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

            #print("Index:", index)
            #print("Self ID embeds", len(self.class_identity_embeds_path))
            #print(index % self.num_class_images)
            example["class_identity_embeds"] = torch.load(self.class_identity_embeds_path[index % self.num_class_images]) 
            
            #exit()
        return example


def collate_fn(examples, with_prior_preservation=False):
    has_flux_embeddings = "instance_prompt_embeds" in examples[0]
    has_attention_mask = "instance_attention_mask" in examples[0]

    pixel_values = [example["instance_images"] for example in examples]
    identity_embed = [example["instance_identity_embeds"] for example in examples]

    if has_flux_embeddings:
        prompt_embeds = [example["instance_prompt_embeds"] for example in examples]
        pooled_prompt_embeds = [example["instance_pooled_prompt_embeds"] for example in examples]
        text_ids = [example["instance_text_ids"] for example in examples]
    else:
        input_ids = [example["instance_prompt_ids"] for example in examples]

    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        if has_flux_embeddings:
            prompt_embeds += [example["class_prompt_embeds"] for example in examples]
            pooled_prompt_embeds += [example["class_pooled_prompt_embeds"] for example in examples]
            text_ids += [example["class_text_ids"] for example in examples]
        else:
            input_ids += [example["class_prompt_ids"] for example in examples]
        if has_attention_mask:
            attention_mask += [example["class_attention_mask"] for example in examples]

        identity_embed += [example["class_identity_embeds"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    identity_embed = torch.cat(identity_embed, dim=0)

    batch = {
        "pixel_values": pixel_values,
        "identity_embed": identity_embed,
    }

    if has_flux_embeddings:
        batch["prompt_embeds"] = torch.stack(prompt_embeds).squeeze(1)
        batch["pooled_prompt_embeds"] = torch.stack(pooled_prompt_embeds).squeeze(1)
        batch["text_ids"] = torch.stack(text_ids)[0]
    else:
        batch["input_ids"] = torch.cat(input_ids, dim=0)

    if has_attention_mask:
        batch["attention_mask"] = attention_mask

    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example

def latents_to_pil_images(latents, vae):
    # bath of latents -> list of images
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents / scaling_factor) + shift_factor
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


# convert a normalized image back to RGB format suitable for PIL
def reverse_normalized_image(img, multiply_255=True):
    mean = 0.5
    std = 0.5 

    denorm = transforms.Normalize(mean=[-mean / std], std=[1.0 / std])

    img = denorm(img).clamp(0, 1)    
    if multiply_255:
        img = (img * 255).to(dtype=torch.uint8)    
    return img 


def latents_to_image_for_mtcnn(latents, vae):
    # batch of latents -> list of images
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    shift_factor = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents / scaling_factor) + shift_factor
    # decode with the pretrained VAE
    image = vae.decode(latents).sample #[0]
    # Transform to image range ... TODO denormalize? 
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image * 255)[0]
    image = torch.permute(image, (1, 2, 0))
    return image


def cropped_image_to_arcface_input(img):
    #  transform to (1, 3, X, X)
    img = torch.permute(img, (2, 0, 1))
    #img = torch.nn.functional.adaptive_avg_pool2d(img, (112,112)) # TODO 
    img = transforms.functional.resize(img, (112, 112), antialias=None) # TODO 
    #img = torch.nn.functional.interpolate(img, [112,112]) 
    # TODO maybe interpolate?
    img = ((img / 255) - 0.5) / 0.5 
    #img = img[None, :, :, :]
    img = torch.unsqueeze(img, 0)
    return img

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs




def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


@torch.no_grad()
def encode_prompt_for_flux(prompt_encoder_pipeline, prompt, device, max_sequence_length):
    prompt_embeds, pooled_prompt_embeds, text_ids = prompt_encoder_pipeline.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    return {
        "prompt_embeds": prompt_embeds.detach().cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.detach().cpu(),
        "text_ids": text_ids.detach().cpu(),
    }


def unpack_flux_latents(latents, latent_height, latent_width):
    batch_size, _, channels = latents.shape
    latents = latents.view(batch_size, latent_height // 2, latent_width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // 4, latent_height, latent_width)
    return latents



def contrastive_loss(x1, x2, label, margin: float = 1.0):
    dist = torch.nn.functional.pairwise_distance(x1, x2)

    loss = (1 - label) * torch.pow(dist, 2) + (label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    loss = torch.mean(loss)

    return loss



def main(args):

    logging_dir = Path(args.output_dir, cfg.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=accelerator_project_config
    )
    
    print(accelerator)

    if cfg.train_text_encoder:
        logger.warning("FLUX QLoRA mode does not support text encoder training in this pipeline. Forcing train_text_encoder=False.")
        cfg.train_text_encoder = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Generate class images if prior preservation is enabled.
    if cfg.with_prior_preservation:
        class_images_dir = Path(cfg.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < cfg.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            if cfg.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif cfg.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif cfg.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = FluxPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=cfg.revision,
                variant=cfg.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = cfg.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(cfg.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=cfg.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(
                    example["prompt"],
                    height=cfg.resolution,
                    width=cfg.resolution,
                    num_inference_steps=getattr(cfg, "validation_num_inference_steps", 28),
                    guidance_scale=getattr(cfg, "guidance_scale", 3.5),
                    max_sequence_length=getattr(cfg, "max_sequence_length", 512),
                ).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    print("Model path:", cfg.pretrained_model_name_or_path)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision, variant=cfg.variant
    )

    bnb_4bit_compute_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        bnb_4bit_compute_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        bnb_4bit_compute_dtype = torch.bfloat16

    if BitsAndBytesConfig is None:
        raise ImportError(
            "BitsAndBytesConfig is not available in this environment. "
            "Install a newer diffusers (recommended >= 0.31) or a compatible transformers/bitsandbytes stack."
        )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=cfg.revision,
        variant=cfg.variant,
        quantization_config=nf4_config,
        torch_dtype=bnb_4bit_compute_dtype,
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer = get_peft_model(transformer, transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    model = unwrap_model(model)
                    transformer_lora_layers_to_save = convert_peft_state_dict_for_flux(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)
        if isinstance(lora_state_dict, tuple):
            lora_state_dict = lora_state_dict[0]

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if cfg.mixed_precision == "fp16":
            models = [transformer_]
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate * cfg.gradient_accumulation_steps * cfg.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if cfg.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    optimizer = optimizer_class(
        params_to_optimize,
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.adam_weight_decay,
        eps=cfg.adam_epsilon,
    )

    cfg.pre_compute_text_embeddings = True
    prompt_encoder_device = torch.device("cpu")
    prompt_encoder_pipeline = FluxPipeline.from_pretrained(
        cfg.pretrained_model_name_or_path,
        transformer=None,
        vae=None,
        revision=cfg.revision,
        variant=cfg.variant,
        torch_dtype=torch.float32,
    )

    precomputed_flux_instance = encode_prompt_for_flux(
        prompt_encoder_pipeline,
        cfg.instance_prompt,
        prompt_encoder_device,
        max_sequence_length=getattr(cfg, "max_sequence_length", 512),
    )
    if cfg.validation_prompt is not None:
        validation_flux_prompt = encode_prompt_for_flux(
            prompt_encoder_pipeline,
            cfg.validation_prompt,
            prompt_encoder_device,
            max_sequence_length=getattr(cfg, "max_sequence_length", 512),
        )
    else:
        validation_flux_prompt = None

    if cfg.class_prompt is not None:
        precomputed_flux_class = encode_prompt_for_flux(
            prompt_encoder_pipeline,
            cfg.class_prompt,
            prompt_encoder_device,
            max_sequence_length=getattr(cfg, "max_sequence_length", 512),
        )
    else:
        precomputed_flux_class = None

    del prompt_encoder_pipeline
    gc.collect()
    torch.cuda.empty_cache()

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=cfg.instance_data_dir,
        instance_prompt=cfg.instance_prompt,
        class_data_root=cfg.class_data_dir if cfg.with_prior_preservation else None,
        class_prompt=cfg.class_prompt,
        class_num=cfg.num_class_images,
        tokenizer=None,
        size=cfg.resolution,
        encoder_hidden_states=None,
        class_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        flux_prompt_embeds=precomputed_flux_instance["prompt_embeds"],
        flux_pooled_prompt_embeds=precomputed_flux_instance["pooled_prompt_embeds"],
        flux_text_ids=precomputed_flux_instance["text_ids"],
        class_flux_prompt_embeds=None if precomputed_flux_class is None else precomputed_flux_class["prompt_embeds"],
        class_flux_pooled_prompt_embeds=None if precomputed_flux_class is None else precomputed_flux_class["pooled_prompt_embeds"],
        class_flux_text_ids=None if precomputed_flux_class is None else precomputed_flux_class["text_ids"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, cfg.with_prior_preservation),
        num_workers=cfg.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        #tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth-lora", config=tracker_config)

    # Train!
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        print(cfg.resume_from_checkpoint)
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[2])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            #print(global_step)
            #print(num_update_steps_per_epoch)
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, cfg.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    ###########################################################
    # ArcFaceModel for Loss 
    arcface_model = prepare_locked_ArcFace_model()
    arcface_model.to(device=accelerator.device)

    # TODO 
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def cosine_distance(x, y):
        sim = F.cosine_similarity(x, y) 
        distance = 1 - sim
        return distance
    
    triplet_loss_function = torch.nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance)
    
    # MTCNN model for face detection
    mtcnn_model = MTCNN(image_size=112,device=accelerator.device, margin=0)


    final_state = False

    vae_config_shift_factor = getattr(vae.config, "shift_factor", 0.0)
    vae_config_scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    vae_config_block_out_channels = vae.config.block_out_channels

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, cfg.num_train_epochs):
        transformer.train()
        avg_combined_loss = []; avg_id_loss = []; avg_instance_loss = []; avg_prior_loss = []

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype) # shape [2,3,512,512]
                gt_arcface_embed = batch["identity_embed"].to(dtype=weight_dtype) # shape [2,512]

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)
                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2],
                    model_input.shape[3],
                    accelerator.device,
                    weight_dtype,
                )

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                indices = torch.randint(0, noise_scheduler_copy.config.num_train_timesteps, (bsz,), device=model_input.device)
                schedule_timesteps = noise_scheduler_copy.timesteps.to(device=model_input.device)
                timesteps = schedule_timesteps[indices]
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                prompt_embeds = batch["prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device=accelerator.device, dtype=weight_dtype)
                text_ids = batch["text_ids"].to(device=accelerator.device, dtype=weight_dtype)

                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor([getattr(cfg, "guidance_scale", 3.5)], device=accelerator.device)
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = unpack_flux_latents(
                    model_pred,
                    latent_height=model_input.shape[2],
                    latent_width=model_input.shape[3],
                )

                target = noise - model_input


                """
                TODO
                """
                if cfg.with_prior_preservation: 
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    noise_instance, _ = torch.chunk(noise, 2, dim=0)
                    timestep_instance = timesteps[0]

                    # Compute instance loss
                    instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    # Add the prior loss to the instance loss.
                    loss = instance_loss + cfg.prior_loss_weight * prior_loss

                    #print(cfg.with_identity_loss)
                    if cfg.which_loss == "identity": 
                        latent_x0 = (noise_instance - model_pred)[0:1]
                        
                        # Perform face detection first 
                        img = latents_to_image_for_mtcnn(latent_x0.to(weight_dtype), vae) 
                        bboxs, probs = mtcnn_model.detect(img, landmarks=False)

                        if bboxs is not None: #and bboxs_prior is not None:  
                            bbox = bboxs[0].astype(int) 
                            initial_size = img.shape[0]
                            img_cropped = img[max(0,bbox[1]): min(bbox[3], initial_size ) , max(0, bbox[0]): min(bbox[2], initial_size)] 
                            
                            img_cropped = cropped_image_to_arcface_input(img_cropped)
                            pred_arcface_features = arcface_model(img_cropped)
                            
                            # compare arcface features between predicted face and gt face
                            arcface_cos_similarity = cos(pred_arcface_features, gt_arcface_embed[0])
                            
                            identity_loss = 1 - arcface_cos_similarity #((1 - arcface_cos_similarity) + (1 - arcface_cos_similarity_prior)) / 2 # TODO check is this ok                         

                            identity_noise_level_weight = (1 - timestep_instance / noise_scheduler_copy.config.num_train_timesteps) ** 2
                            if not cfg.timestep_loss_weighting: identity_noise_level_weight = 1 
                            
                            loss = loss + identity_noise_level_weight * identity_loss
                        #else: 
                        #    print("not detected", timesteps[0])
                    
                    if cfg.which_loss == "triplet_prior": 
                        latent_x0 = (noise_instance - model_pred)[0:1]
                        
                        #print(latent_x0)
                        
                        # Perform face detection first 
                        img = latents_to_image_for_mtcnn(latent_x0.to(weight_dtype), vae) 
                        bboxs, probs = mtcnn_model.detect(img, landmarks=False)
                        
                        #latent_x0_prior = noise_scheduler.step(model_pred_prior, timesteps[1], noisy_model_input[1]).pred_original_sample
                        #img_prior = latents_to_image_for_mtcnn(latent_x0_prior.to(weight_dtype), vae) 
                        # bboxs_prior, probs_prior = mtcnn_model.detect(img_prior, landmarks=False)
                        
                        if bboxs is not None: #and bboxs_prior is not None:  
                            bbox = bboxs[0].astype(int) 
                            initial_size = img.shape[0]
                            img_cropped = img[max(0,bbox[1]): min(bbox[3], initial_size ) , max(0, bbox[0]): min(bbox[2], initial_size)] 
                            
                            img_cropped = cropped_image_to_arcface_input(img_cropped)
                            pred_arcface_features = arcface_model(img_cropped)
                            
                            identity_noise_level_weight = (1 - timestep_instance / noise_scheduler_copy.config.num_train_timesteps)  ** 2
                            if not cfg.timestep_loss_weighting: identity_noise_level_weight = 1 

                            # input: anchor, positive, negative
                            triplet_loss = triplet_loss_function(pred_arcface_features, gt_arcface_embed[0][None, :], gt_arcface_embed[1][None, :])
                            loss = loss +  identity_noise_level_weight * triplet_loss
                        
                else:
                    instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    loss = instance_loss
                
                accelerator.backward(loss)  

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            step_loss = loss.detach().item()
            step_instance_loss = instance_loss.detach().item()

            if cfg.with_prior_preservation:
                step_prior_loss = prior_loss.detach().item()
            else: step_prior_loss = 0 

            if cfg.which_loss == "identity":
                step_id_loss = identity_loss.detach().item()
            elif cfg.which_loss == "triplet_prior":
                step_id_loss = triplet_loss.detach().item()
            else: step_id_loss = 0

            avg_combined_loss.append(step_loss)
            avg_instance_loss.append(step_instance_loss)
            avg_prior_loss.append(step_prior_loss)
            avg_id_loss.append(step_id_loss)

            logs = {"Step Loss/Reconstruction": step_instance_loss, "Step Loss/ID": step_id_loss,  
                    "Step Loss/Prior": step_prior_loss, "Step Loss/Combined": step_loss, "LR": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
                final_state = True
                break
        
        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.is_main_process:
            if (epoch + 1) % cfg.checkpointing_epochs == 0 or final_state: 
            #if global_step % cfg.checkpointing_steps == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if cfg.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= cfg.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - cfg.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        if accelerator.is_main_process:
            if cfg.validation_prompt is not None and (epoch + 1) % cfg.validation_epochs == 0:
                pipeline = FluxPipeline.from_pretrained(
                    cfg.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=None,
                    text_encoder_2=None,
                    revision=cfg.revision,
                    variant=cfg.variant,
                    torch_dtype=weight_dtype,
                )
                configure_flux_inference_memory(pipeline, accelerator)

                pipeline_args = prepare_flux_validation_kwargs(validation_flux_prompt, accelerator, weight_dtype)

                images = log_validation(
                    pipeline,
                    args,
                    accelerator,
                    pipeline_args,
                    epoch,
                )
        
        epoch_logs = {"Epoch Loss/Reconstruction": np.mean(np.array(avg_instance_loss)), "Epoch Loss/ID": np.mean(np.array(avg_id_loss)),
                      "Epoch Loss/Prior": np.mean(np.array(avg_prior_loss)), "Epoch Loss/Combined": np.mean(np.array(avg_combined_loss)),}
        accelerator.log(epoch_logs, step=global_step)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.to(torch.float32)

        transformer_lora_state_dict = convert_peft_state_dict_for_flux(get_peft_model_state_dict(transformer))

        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_state_dict,
            text_encoder_lora_layers=None,
        )

        # load attention processors
        lora_loaded_for_inference = True
        try:
            pipeline = build_flux_inference_pipeline(
                accelerator,
                weight_dtype=weight_dtype,
                lora_dir=args.output_dir,
                lora_weight_name="pytorch_lora_weights.safetensors",
            )
        except Exception as error:
            lora_loaded_for_inference = False
            logger.warning(
                "Skipping final LoRA inference load due to compatibility issue in this diffusers version: %s",
                str(error),
            )

        # run inference
        images = []
        if lora_loaded_for_inference and cfg.validation_prompt and cfg.num_validation_images > 0:
            pipeline_args = prepare_flux_validation_kwargs(validation_flux_prompt, accelerator, weight_dtype)
            images = log_validation(
                pipeline,
                args,
                accelerator,
                pipeline_args,
                epoch,
                is_final_validation=True,
            )


    accelerator.end_training()



if __name__ == "__main__":
    args = parse_args()

    if args.test_run:
        cfg.num_train_epochs = 1
        cfg.max_train_steps = None

    id_folders = os.listdir(cfg.source_folder)
    
    for which_loss in cfg.losses_to_test:#["triplet_prior"]:#,# "identity", "triplet_prior"]:#["", "identity", "triplet_prior"]:
        cfg.which_loss = which_loss
        output_folder = cfg.output_folder
        
        if cfg.train_text_encoder: 
            output_folder += "_WithTextEncoder"

        loss_folder = ""
        if cfg.which_loss == "":
            loss_folder = "DreamBooth"
        elif cfg.which_loss == "identity":
            loss_folder = "PortraitBooth"
        elif cfg.which_loss == "triplet_prior":
            loss_folder = "ID-Booth"

        output_folder = os.path.join(output_folder, loss_folder)
        
        cfg.instance_data_dir = cfg.source_folder
        args.output_dir = output_folder
        
        print("Args:", vars(args))
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save args to json
        json_output_file = os.path.join(args.output_dir, "training_config.json")
        with open(json_output_file, 'w') as fp:

            config_vars={var:vars(cfg)[var] for var in dir(cfg) if not var.startswith('_')}
            args_vars = vars(args) 
            all_args = config_vars | args_vars
            json.dump(all_args, fp, indent=4)
        
        id_folders.sort(key=natural_keys)
        id_limit = 1 if args.test_run else int(os.environ.get("ID_BOOTH_MAX_IDENTITIES", "0"))
        for i, id_folder in enumerate(id_folders): 
            print(id_folder)
            if id_limit != 0 and i >= id_limit:
                print(f"Limit training to {id_limit} identities.")
                break
            
            cfg.instance_data_dir = os.path.join(cfg.source_folder, id_folder) # "./DATASETS/TUFTS_TEST_512/images_id_1"
            args.output_dir = os.path.join(output_folder, id_folder) 
            main(args)
