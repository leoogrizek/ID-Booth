import os
import shutil
import random

# ====== CONFIG ======
source_root = "FACE_DATASET"
target_root = "CLASS_IMAGES"
num_identities_to_move = 20
random_seed = 42
dry_run = False
# ====================

source_images_root = os.path.join(source_root, "images")
source_embeds_root = os.path.join(source_root, "ArcFace_embeds")

target_dataset_name = f"SD21_Class_imgs_{num_identities_to_move}"
target_images_root = os.path.join(target_root, target_dataset_name, "images")
target_embeds_root = os.path.join(target_root, target_dataset_name, "ArcFace_embeds")


def get_identity_folders(root_path):
    if not os.path.isdir(root_path):
        raise FileNotFoundError(f"Missing directory: {root_path}")

    return sorted([
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ])


def main():
    random.seed(random_seed)

    image_ids = set(get_identity_folders(source_images_root))
    embed_ids = set(get_identity_folders(source_embeds_root))

    common_ids = sorted(image_ids & embed_ids)
    only_images = sorted(image_ids - embed_ids)
    only_embeds = sorted(embed_ids - image_ids)

    if only_images:
        print("These identities exist in images but not ArcFace_embeds:")
        for name in only_images:
            print(f"  {name}")

    if only_embeds:
        print("These identities exist in ArcFace_embeds but not images:")
        for name in only_embeds:
            print(f"  {name}")

    if len(common_ids) < num_identities_to_move:
        raise ValueError(
            f"Requested {num_identities_to_move} identities, but only "
            f"{len(common_ids)} matching identity folders were found."
        )

    selected_ids = random.sample(common_ids, num_identities_to_move)
    selected_ids.sort()

    print(f"Selected {len(selected_ids)} identities to move:")
    for name in selected_ids:
        print(f"  {name}")

    os.makedirs(target_images_root, exist_ok=True)
    os.makedirs(target_embeds_root, exist_ok=True)

    for identity in selected_ids:
        src_img = os.path.join(source_images_root, identity)
        src_emb = os.path.join(source_embeds_root, identity)

        dst_img = os.path.join(target_images_root, identity)
        dst_emb = os.path.join(target_embeds_root, identity)

        if os.path.exists(dst_img):
            raise FileExistsError(f"Destination already exists: {dst_img}")
        if os.path.exists(dst_emb):
            raise FileExistsError(f"Destination already exists: {dst_emb}")

        print(f"\nMoving identity: {identity}")
        print(f"  images: {src_img} -> {dst_img}")
        print(f"  embeds: {src_emb} -> {dst_emb}")

        if not dry_run:
            shutil.move(src_img, dst_img)
            shutil.move(src_emb, dst_emb)

    print("\nDone.")
    print(f"Class dataset created at: {os.path.join(target_root, target_dataset_name)}")


if __name__ == "__main__":
    main()