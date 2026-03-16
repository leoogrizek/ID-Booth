import numpy as np
import torch
import json
import os

from ArcFace_files.ArcFace_functions import prepare_locked_ArcFace_model
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def prepare_for_arcface_model_torch(img):
    img = img.permute(2, 0, 1)
    img = transforms.functional.resize(img, (112, 112), antialias=True)
    img = img.float()
    img = ((img / 255) - 0.5) / 0.5
    img = img.unsqueeze(0)
    return img


origin_path = "FACE_DATASET"
device = "cuda:0"

arcface_model = prepare_locked_ArcFace_model()
arcface_model.to(device=device)
arcface_model.eval()

mtcnn_model = MTCNN(image_size=112, device=device, margin=0)

files_without_faces = dict()
files_without_faces["files_without_faces"] = []

folders = os.listdir(os.path.join(origin_path, "images"))

for folder in tqdm(folders):
    folder_path = os.path.join(origin_path, "images", folder)

    output_path = folder_path.replace("images", "ArcFace_embeds")
    os.makedirs(output_path, exist_ok=True)
    img_files = os.listdir(folder_path)

    for img_name in img_files:
        img_path = os.path.join(folder_path, img_name)

        image = Image.open(img_path).convert("RGB")
        image = torch.from_numpy(np.array(image)).to(device)

        # MTCNN expects a batch here in your current usage pattern
        image_batch = image.unsqueeze(0)

        with torch.no_grad():
            bboxs, probs = mtcnn_model.detect(image_batch, landmarks=False)

        bbox = bboxs[0]
        if bbox is None:
            files_without_faces["files_without_faces"].append(img_path)
            continue

        bbox = bbox[0].astype(int)

        initial_h = image.shape[0]
        initial_w = image.shape[1]

        img_cropped = image[
            max(0, bbox[1]): min(bbox[3], initial_h),
            max(0, bbox[0]): min(bbox[2], initial_w)
        ]

        img_cropped = prepare_for_arcface_model_torch(img_cropped)

        with torch.no_grad():
            face_embed = arcface_model(img_cropped)

        embed_file_name = os.path.splitext(img_name)[0] + ".pt"
        torch.save(face_embed.cpu(), os.path.join(output_path, embed_file_name))

json_pth = f"{origin_path}/files_without_faces.json"

print(json_pth)

with open(json_pth, "w") as fp:
    json.dump(files_without_faces, fp)