pretrained_model_name_or_path = "sd2-community/stable-diffusion-2-1-base"

mixed_precision = "fp16"
logging_dir = "Logs"
report_to = "tensorboard"

revision = None
variant = None

resume_from_checkpoint = None

# Source images
source_folder = "FACE_DATASET/images"
resolution = 512
instance_prompt = "photo of sks person"

# Prior preservation images
with_prior_preservation = True
class_prompt = "photo of a person"
class_data_dir = "CLASS_IMAGES/SD21_Class_imgs_20/images"
num_class_images = 20
prior_loss_weight = 1.0

validation_prompt = "photo of sks person with blue hair"
validation_negative_prompt = ""
validation_prompt_path = None

num_validation_images = 4

dataloader_num_workers = 0
use_8bit_adam = False
enable_xformers_memory_efficient_attention = False
allow_tf32 = True
prior_generation_precision = "fp16"

local_rank = -1
tokenizer_max_length = None
tokenizer_name = None
text_encoder_use_attention_mask = False
class_labels_conditioning = None
max_train_steps = None

seed = 0

# Training parameters
lora_rank = 4
train_batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = False

num_train_epochs = 1
validation_epochs = 1
checkpointing_epochs = 1
checkpoints_total_limit = None

learning_rate = 1e-4
lr_scheduler = "cosine"
lr_warmup_steps = 0

adam_beta1 = 0.9
adam_beta2 = 0.999
adam_weight_decay = 1e-2
adam_epsilon = 1e-08
max_grad_norm = 1.0

scale_lr = False
lr_num_cycles = 1
lr_power = 1.0

train_text_encoder = False
pre_compute_text_embeddings = False

losses_to_test = ["triplet_prior"]
timestep_loss_weighting = True
alpha_id_loss_weighting = 0.1

sample_batch_size = 1

output_folder = "Trained_LoRA_Models/"
show_tqdm = True