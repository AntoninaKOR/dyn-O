# Copyright (c) Meta Platforms, Inc. and affiliates.
# Lightly adapted from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
import os

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


np.random.seed(3)

model = "large"
points_per_side = 32
pred_iou_thresh = 0.7
stability_score_thresh = 0.9
stability_score_offset = 0.5


def process_images(gpu_id, img_paths):
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(f"cuda:{gpu_id}")
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id=f"facebook/sam2-hiera-large",
        device=device,
        points_per_side=points_per_side,
        points_per_batch=1024,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
    )

    for img_path in tqdm(img_paths):
        image = Image.open(img_path)
        image = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image)
        masks = np.array([mask['segmentation'] for mask in masks])
        # save the masks
        new_filename = f"{img_path.stem}_mask"
        mask_path = img_path.with_name(new_filename)
        np.savez_compressed(mask_path, masks)


def main():
    # Define your output directory
    data_dir = Path("/scratch/cluster/zzwang_new/oc_ssm/data/plunder/train_no_background")
    available_gpus_ids = [0, 2, 3]

    # Collect all image paths
    image_paths = []
    for epi_path in data_dir.iterdir():
        if not epi_path.is_dir():
            continue
        for img_path in epi_path.iterdir():
            if img_path.suffix != ".png":
                continue
            image_paths.append(img_path)

    # Number of GPUs
    num_gpus = len(available_gpus_ids)

    # Split the dataset
    split_paths = np.array_split(image_paths, num_gpus)

    # Process images in parallel using multiple GPUs
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_images, available_gpus_ids[gpu_id], split_paths[gpu_id])
                   for gpu_id in range(num_gpus)]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
