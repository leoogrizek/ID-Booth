import torch
import os 
from torchvision.utils import save_image
import random 
from accelerate.utils import  set_seed
from tqdm import tqdm 
import re 
import json 
#from compel import Compel
from diffusers import FluxPipeline, FluxTransformer2DModel
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
        from transformers import BitsAndBytesConfig
from itertools import product 
from utils.sorting_utils import natural_keys
from PIL import Image

backgrounds_list = ["","forest", "city street", "beach", "office", "bus", "laboratory", "factory", "construction site", "hospital", "night club"]
backgrounds_list = [f"{b} background"  if b != "" else "" for b in backgrounds_list]#

age_phases = ["", "young", "middle-aged", "old"]

num_samples_per_prompt = 1
num_prompts = 21 # 21 #21 #50 #21 #len(additions_list)

add_gender = True
add_pose = True
add_age = False # should be first in combination
add_background = True 

do_not_use_negative_prompt = False
use_non_finetuned = False 

if add_age and add_background: 
    all_prompt_combinations = list(product(age_phases, backgrounds_list))
elif add_background: 
    if num_prompts == 100:
        all_prompt_combinations = list(backgrounds_list[1:] * 10)
    else:
        all_prompt_combinations = list([""] + backgrounds_list[1:] * 2)
    
elif add_age: 
    all_prompt_combinations = list(age_phases * 6)
else: 
    all_prompt_combinations = list([""] * num_prompts)
print(all_prompt_combinations)

device = "cuda:0"
seed = 0 
guidance_scale = 3.0
num_inference_steps = 28
max_sequence_length = 512

# FLUX inference memory controls
quantize_transformer_nf4 = True
model_cpu_offload = True
sequential_cpu_offload = False
vae_tiling = True
vae_slicing = True

# Optional Redux conditioning
use_redux = False
redux_model_name_or_path = "black-forest-labs/FLUX.1-Redux-dev"
reference_image_path = None

folder_of_models = f"Trained_LoRA_Models" 
#models_to_test = ["DreamBooth", "PortraitBooth", "ID-Booth"]
models_to_test = ["ID-Booth"]
checkpoint =  "checkpoint-0-20" 

folder_output = f"Generated_Samples/FacePortrait_Photo_21"  # _NonFinetuned
if add_gender: folder_output += "_Gender"
if add_pose: folder_output+= "_Pose"
if add_age: folder_output+= "_Age"
if add_background: folder_output += "_Background"
if do_not_use_negative_prompt: folder_output += "_NoNegPrompt"

architectures = ["black-forest-labs/FLUX.1-dev"]
model_architecture = architectures[0]
arch = model_architecture.split("/")[1]

set_seed(seed)

width, height = 512, 512

ids = os.listdir(os.path.join(folder_of_models, models_to_test[0]))
ids = [i for i in ids if ".json" not in i]
ids.sort(key=natural_keys)

print(ids)
gender_dict = {}
if add_gender:
    with open("tufts_gender_dict.json", "r") as fp:
        gender_dict = json.load(fp)


negative_prompt = "cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions, landscape" 
original_prompt = f"face portrait photo of sks person"

prompt = ""


def load_lora_weights_compat(pipe, model_path, weight_name="pytorch_lora_weights.safetensors"):
    pipe.load_lora_weights(model_path, weight_name=weight_name)


def build_flux_pipe(dtype):
    transformer = None
    if quantize_transformer_nf4:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            model_architecture,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=dtype,
        )

    if use_redux:
        if FluxPriorReduxPipeline is None:
            raise ImportError("FluxPriorReduxPipeline is unavailable in this diffusers version.")
        if reference_image_path is None:
            raise ValueError("Set reference_image_path when use_redux=True")

    pipe_kwargs = {
        "transformer": transformer,
        "torch_dtype": dtype,
    }
    if use_redux:
        pipe_kwargs["text_encoder"] = None
        pipe_kwargs["text_encoder_2"] = None
    pipe = FluxPipeline.from_pretrained(model_architecture, **pipe_kwargs)

    redux_prior = None
    if use_redux:
        if FluxPriorReduxPipeline is None:
            raise ImportError("FluxPriorReduxPipeline is unavailable in this diffusers version.")
        redux_prior = FluxPriorReduxPipeline.from_pretrained(redux_model_name_or_path, torch_dtype=dtype)

    if vae_slicing and hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
    if vae_tiling and hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_tiling()

    if model_cpu_offload:
        if redux_prior is not None:
            redux_prior.enable_model_cpu_offload()
        pipe.enable_model_cpu_offload()
    elif sequential_cpu_offload:
        if redux_prior is not None:
            redux_prior.enable_model_cpu_offload()
        pipe.enable_sequential_cpu_offload()
    else:
        if redux_prior is not None:
            redux_prior.to(device)
        pipe.to(device)

    return pipe, redux_prior


def build_prompt_encoder_pipe(dtype):
    return FluxPipeline.from_pretrained(
        model_architecture,
        transformer=None,
        vae=None,
        torch_dtype=torch.float32 if dtype == torch.float16 else dtype,
    )


@torch.no_grad()
def encode_prompt_for_flux(prompt_encoder_pipe, prompt, target_dtype):
    prompt_embeds, pooled_prompt_embeds, _ = prompt_encoder_pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        device=torch.device("cpu"),
        max_sequence_length=max_sequence_length,
    )
    return {
        "prompt_embeds": prompt_embeds.to(dtype=target_dtype),
        "pooled_prompt_embeds": pooled_prompt_embeds.to(dtype=target_dtype),
    }

for id_number, which_id in enumerate(ids):
    print("\n", which_id) 
    gender = ""

    if add_gender: 
        gender = gender_dict[which_id]
        if gender == "M": gender = "male"
        elif gender == "F": gender = "female"
    
    all_prompts_for_id = random.sample(all_prompt_combinations, num_prompts)
    
    comparison_image_list = [] 
    for model_name in models_to_test:
        full_model_path = os.path.join(folder_of_models, model_name, which_id, checkpoint)

        output_dir = os.path.join(folder_output, model_name)#"GENERATED_SAMPLES/FINAL_No_ID_loss_TEST"
        print("Load:", full_model_path)
        
        dtype = torch.float16
        pipe, redux_prior = build_flux_pipe(dtype)
        prompt_encoder_pipe = build_prompt_encoder_pipe(dtype) if not use_redux else None

        if not use_non_finetuned:
            load_lora_weights_compat(pipe, full_model_path)
        pipe.set_progress_bar_config(disable=True)
        
        os.makedirs(output_dir, exist_ok=True)
        generator_device = "cpu" if (model_cpu_offload or sequential_cpu_offload) else device
        generator = torch.Generator(device=generator_device).manual_seed(id_number)
        prompt_embed_cache = {}

        reference = Image.open(reference_image_path).convert("RGB") if use_redux else None

        for i, num_prompt in enumerate(tqdm(range(num_prompts))): 
            prompt_additions = all_prompts_for_id[i]
            prompt = original_prompt
            if add_age:
                age_insert = ""
                if isinstance(prompt_additions, str): age_insert = prompt_additions
                else: 
                    age_insert = prompt_additions[0] 
                    prompt_additions = prompt_additions[1:]
                if age_insert != "": prompt = prompt.replace(" sks person", f" {age_insert} sks person")
                

            if add_gender: prompt = prompt.replace(" sks person", f" {gender} sks person")
            if add_pose and random.choice([True, False]): prompt = prompt.replace("portrait", "side-portrait")

            if add_background:
                if isinstance(prompt_additions, str): 
                    prompt += f", {prompt_additions}"  
                else:
                    for addition in prompt_additions:
                        if addition != "":
                            prompt += f", {addition}"

            # generate samples
            for j in range(num_samples_per_prompt):  
                if use_redux:
                    if redux_prior is None:
                        raise RuntimeError("Redux prior is not initialized while use_redux=True")
                    prior_output = redux_prior(reference, prompt=prompt)
                    output = pipe(
                        output_type="np",
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        max_sequence_length=max_sequence_length,
                        **prior_output,
                    )
                else:
                    prompt_kwargs = prompt_embed_cache.get(prompt)
                    if prompt_kwargs is None:
                        prompt_kwargs = encode_prompt_for_flux(prompt_encoder_pipe, prompt, dtype)
                        prompt_embed_cache[prompt] = prompt_kwargs
                    output = pipe(
                        output_type="np",
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        max_sequence_length=max_sequence_length,
                        **prompt_kwargs,
                    )
                output = torch.Tensor(output.images)
                comparison_image_list.append(output)
                output = torch.permute(output, (0, 3, 1, 2))
                path_to_output = f"{output_dir}/{which_id}_{checkpoint}_{arch}"
                os.makedirs(path_to_output, exist_ok=True)
                save_image(output, fp=f"{path_to_output}/{i}_{j}_{prompt}.png")

        if prompt_encoder_pipe is not None:
            del prompt_encoder_pipe
        del pipe
        if redux_prior is not None:
            del redux_prior
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    images = torch.cat(comparison_image_list)
    images = torch.permute(images, (0, 3, 1, 2)) # permute dimensions to be (x, 3, 512, 512)
    
    print("Saving comparison image") 
    comparison_folder = "Comparison"

    comparison_folder = os.path.join(folder_output, comparison_folder)
    os.makedirs(comparison_folder, exist_ok=True)
    save_path = f"{comparison_folder}/{which_id}_{checkpoint}_{arch}_{guidance_scale}.jpg"
    print(save_path)
    save_image(images, fp=save_path, nrow=num_prompts*num_samples_per_prompt, padding=0)

