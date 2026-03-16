import os
import torch

root = "FACE_DATASET"
images_root = os.path.join(root, "images")
embeds_root = os.path.join(root, "ArcFace_embeds")

missing = 0
broken = 0
ok = 0

for folder in sorted(os.listdir(images_root)):
    img_dir = os.path.join(images_root, folder)
    emb_dir = os.path.join(embeds_root, folder)

    if not os.path.isdir(img_dir):
        continue

    if not os.path.isdir(emb_dir):
        print(f"[MISSING FOLDER] {emb_dir}")
        missing += 1
        continue

    for filename in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, filename)

        if not os.path.isfile(img_path):
            continue

        stem, _ = os.path.splitext(filename)
        emb_path = os.path.join(emb_dir, stem + ".pt")

        if not os.path.exists(emb_path):
            print(f"[MISSING EMBED] {img_path} -> {emb_path}")
            missing += 1
            continue

        try:
            emb = torch.load(emb_path, map_location="cpu")
            shape = tuple(emb.shape) if hasattr(emb, "shape") else type(emb)
            print(f"[OK] {emb_path} | shape={shape}")
            ok += 1
        except Exception as e:
            print(f"[BROKEN EMBED] {emb_path} | error={e}")
            broken += 1

print("\nSummary:")
print(f"  OK:      {ok}")
print(f"  Missing: {missing}")
print(f"  Broken:  {broken}")