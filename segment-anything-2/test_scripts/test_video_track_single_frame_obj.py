import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path

import torch

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


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


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


# Setup
device = "cuda:1"

model = "large"
points_per_side = 32
pred_iou_thresh = 0.8
stability_score_thresh = 0.9
stability_score_offset = 0.8

mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
    model_id=f"facebook/sam2-hiera-{model}",
    device=device,
    points_per_side=points_per_side,
    points_per_batch=1024,
    pred_iou_thresh=pred_iou_thresh,
    stability_score_thresh=stability_score_thresh,
    stability_score_offset=stability_score_offset,
)

predictor = SAM2VideoPredictor.from_pretrained(
    model_id="facebook/sam2-hiera-large",
    hydra_overrides_extra=[
        "++model.non_overlap_masks=True",
    ],
    device=device,
)

out_dir = f"sam_test_{model}_{points_per_side}_{pred_iou_thresh}_{stability_score_thresh}_{stability_score_offset}"
out_dir = Path(out_dir.replace(".", ""))

# Auto-masking of first frame (from automatic mask generation notebook)
video_path = Path("/scratch/cluster/zzwang_new/procgen/data")
for env_name in ["bossfight", "bigfish", "plunder"]:
    env_path = video_path / env_name / "train"
    output_dir_path = out_dir / env_name
    output_dir_path.mkdir(parents=True, exist_ok=True)
    for epi_path in env_path.iterdir():
        print(epi_path)
        epi_idx = epi_path.name
        frame_names = [
            p for p in os.listdir(epi_path)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))

        frame_idx = 0
        first_frame_path = os.path.join(epi_path, frame_names[frame_idx])
        first_frame = Image.open(first_frame_path)
        first_frame = np.array(first_frame.convert("RGB"))
        auto_masks = mask_generator.generate(first_frame)
        auto_masks = sorted(auto_masks, key=(lambda x: x['area']), reverse=True)

        plt.figure(figsize=(5, 5))
        plt.title(f"frame {frame_idx}")
        plt.imshow(first_frame)
        show_anns(auto_masks)
        plt.axis('off')
        output_path = output_dir_path / f"first_frame_{epi_idx}.png"
        plt.savefig(output_path)
        plt.close("all")

        print("Number of auto-masks:", len(auto_masks))

        inference_state = predictor.init_state(video_path=str(epi_path))
        dtype = next(predictor.parameters()).dtype
        lowres_side_length = predictor.image_size // 4
        prompts = {}

        for mask_idx, mask_result in enumerate(auto_masks):
            mask = mask_result["segmentation"]
            if mask.mean() > 0.5:
                assert mask_idx == 0

            num_points = 1

            true_coords = np.where(mask)
            flat_true_indices = np.ravel_multi_index(true_coords, mask.shape)
            sampled_flat_indices = np.random.choice(flat_true_indices, size=num_points, replace=False)
            sampled_coords = np.array(true_coords).T[np.isin(flat_true_indices, sampled_flat_indices)]

            points = sampled_coords[:, [1, 0]].astype(np.float32)
            labels = np.ones(num_points, np.int32)
            prompts[mask_idx] = points, labels

            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=mask_idx,
                points=points,
                labels=labels,
            )

        plt.figure(figsize=(5, 5))
        plt.title(f"frame {frame_idx}")
        plt.imshow(first_frame)
        for i, out_obj_id in enumerate(out_obj_ids):
            show_points(*prompts[out_obj_id], plt.gca())
            show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
        plt.axis('off')
        output_path = output_dir_path / f"first_frame_{epi_idx}_video.png"
        plt.savefig(output_path)
        plt.close("all")

        # Do video segmentation (same as video segmentation notebook)
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=frame_idx,
            reverse=False,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy() for i, out_obj_id in enumerate(out_obj_ids)
            }

        for out_frame_idx in sorted(video_segments.keys()):
            plt.figure(figsize=(5, 5))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(epi_path, frame_names[out_frame_idx])))
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
            plt.axis('off')
            output_path = output_dir_path / f"out_{epi_idx}_frame_{out_frame_idx}.jpg"
            plt.savefig(output_path)
            plt.close("all")
