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


# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.0]])
        img[m] = color_mask
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 1), thickness=1)

    ax.imshow(img)


from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

model = "large"
points_per_side = 32
pred_iou_thresh = 0.5
stability_score_thresh = 0.8
stability_score_offset = 0.5

mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
    model_id=f"facebook/sam2-hiera-{model}",
    device=torch.device("cuda:0"),
    points_per_side=points_per_side,
    points_per_batch=1024,
    pred_iou_thresh=pred_iou_thresh,
    stability_score_thresh=stability_score_thresh,
    stability_score_offset=stability_score_offset,
)

out_dir = f"sam_test_{model}_{points_per_side}_{pred_iou_thresh}_{stability_score_thresh}_{stability_score_offset}"
out_dir = Path(out_dir.replace(".", ""))
for env_name in ["bossfight"]:
    env_path = Path(f'/scratch/cluster/zzwang_new/procgen/data/{env_name}/train/')
    for epi_path in env_path.iterdir():
        for img_path in epi_path.iterdir():
            print(img_path)
            image = Image.open(img_path)
            image = np.array(image.convert("RGB"))
            masks = mask_generator.generate(image)
            segmentations = np.empty((len(masks) + 1, *image.shape[:2]), dtype=bool)
            plt.figure(figsize=(5, 5))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            output_dir_path = out_dir / env_name
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_path = output_dir_path / f"{epi_path.stem}_{img_path.stem}.png"
            plt.savefig(output_path)
            plt.close("all")
