from diffusers import StableDiffusionPipeline
import torch
import os 
from torchvision.utils import save_image
from diffusers import DPMSolverMultistepScheduler
from diffusers import DDPMScheduler
import random 
from accelerate.utils import  set_seed
from tqdm import tqdm 
import re 
import json 
#from compel import Compel
from diffusers import AutoPipelineForText2Image
from itertools import product 
from utils.sorting_utils import natural_keys
from diffusers.loaders import LoraLoaderMixin

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
guidance_scale = 5.0
num_inference_steps = 30 

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

architectures = ["sd2-community/stable-diffusion-2-1-base"]
model_architecture = architectures[0]
arch = model_architecture.split("/")[1]

set_seed(seed)

width, height = 512, 512

ids = os.listdir(os.path.join(folder_of_models, models_to_test[0]))
ids = [i for i in ids if ".json" not in i]
ids.sort(key=natural_keys)

print(ids)
if add_gender:
    with open("tufts_gender_dict.json", "r") as fp:
        gender_dict = json.load(fp)


negative_prompt = "cartoon, cgi, render, illustration, painting, drawing, black and white, bad body proportions, landscape" 
original_prompt = f"face portrait photo of sks person"

prompt = ""


def convert_peft_lora_state_dict_for_legacy_diffusers(state_dict):
    converted = {}

    for key, value in state_dict.items():
        key = key.replace("unet.base_model.model.", "unet.")
        key = key.replace("text_encoder.base_model.model.", "text_encoder.")

        if ".lora_A.weight" in key or ".lora_B.weight" in key:
            up_or_down = "down" if ".lora_A.weight" in key else "up"
            key = key.replace(".lora_A.weight", "")
            key = key.replace(".lora_B.weight", "")
            key = key.replace(".to_out.0", ".to_out")

            if ".attn1." in key:
                key = key.replace(".attn1.", ".attn1.processor.")
            if ".attn2." in key:
                key = key.replace(".attn2.", ".attn2.processor.")

            key = f"{key}_lora.{up_or_down}.weight"

        converted[key] = value

    return converted


def load_lora_weights_compat(pipe, model_path, weight_name="pytorch_lora_weights.safetensors"):
    try:
        pipe.load_lora_weights(model_path, weight_name=weight_name)
        return
    except Exception as error:
        if "base_model" not in str(error):
            raise

    state_dict, _ = LoraLoaderMixin.lora_state_dict(model_path, weight_name=weight_name)
    state_dict = convert_peft_lora_state_dict_for_legacy_diffusers(state_dict)
    pipe.load_lora_weights(state_dict)

for id_number, which_id in enumerate(ids):
    print("\n", which_id) 

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
        
        pipe = StableDiffusionPipeline.from_pretrained(model_architecture, torch_dtype=torch.float16).to(device)
        pipe.scheduler = DDPMScheduler.from_pretrained(model_architecture, subfolder="scheduler")

        if not use_non_finetuned:
            load_lora_weights_compat(pipe, full_model_path)
        pipe.set_progress_bar_config(disable=True)
        
        os.makedirs(output_dir, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(id_number) 

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
                output = pipe(prompt=prompt, negative_prompt=negative_prompt,  output_type="np", generator=generator, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height)
                output = torch.Tensor(output.images)
                comparison_image_list.append(output)
                output = torch.permute(output, (0, 3, 1, 2))
                path_to_output = f"{output_dir}/{which_id}_{checkpoint}_{arch}"
                os.makedirs(path_to_output, exist_ok=True)
                save_image(output, fp=f"{path_to_output}/{i}_{j}_{prompt}.png")
        
    images = torch.cat(comparison_image_list)
    images = torch.permute(images, (0, 3, 1, 2)) # permute dimensions to be (x, 3, 512, 512)
    
    print("Saving comparison image") 
    comparison_folder = "Comparison"

    comparison_folder = os.path.join(folder_output, comparison_folder)
    os.makedirs(comparison_folder, exist_ok=True)
    save_path = f"{comparison_folder}/{which_id}_{checkpoint}_{arch}_{guidance_scale}.jpg"
    print(save_path)
    save_image(images, fp=save_path, nrow=num_prompts*num_samples_per_prompt, padding=0)

