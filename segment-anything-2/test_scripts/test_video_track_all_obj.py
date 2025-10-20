import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from copy import deepcopy
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt, label

import tyro
import torch
import torchvision.transforms as T

# use bfloat16 for the entire notebook
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import build_point_grid

# compress UserWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


CMAP = plt.get_cmap("tab20")


def show_anns(obj_ids, anns, borders=True, random_color=False):
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)

    if isinstance(anns, np.ndarray):
        sorted_anns = anns
        img = np.ones((sorted_anns.shape[1], sorted_anns.shape[2], 4))
    elif isinstance(anns, list):
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        img = np.ones((sorted_anns[0]["segmentation"].shape[0], sorted_anns[0]["segmentation"].shape[1], 4))
    else:
        raise ValueError("anns should be a numpy array or a list of dictionaries")

    img[:, :, 3] = 0
    for id, ann in zip(obj_ids, sorted_anns):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
        else:
            color = np.array([*CMAP(id % CMAP.N)[:3], 1.0])

        m = ann["segmentation"] if isinstance(ann, dict) else ann
        img[m] = color
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 1), thickness=1)

    ax.imshow(img)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1.0])], axis=0)
    else:
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*CMAP(cmap_idx % CMAP.N)[:3], 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)


def viz_auto_mask(frame, obj_ids, auto_masks, frame_idx=0, fig_path=None, points=None):
    fig = plt.figure(figsize=(5, 5))
    plt.title(f"auto mask frame {frame_idx}: {len(auto_masks)} masks")
    plt.imshow(frame)
    show_anns(obj_ids, auto_masks)
    if points is not None:
        plt.gca().scatter(points[:, 0], points[:, 1], color="green", marker="*", s=10, edgecolor="white", linewidth=1)
    plt.axis("off")
    plt.savefig(fig_path)
    plt.close(fig)


def viz_prompt_mask(frame, prompts, prompt_mask, frame_idx, fig_path):
    fig = plt.figure(figsize=(5, 5))
    plt.title(f"prompt mask frame {frame_idx}: {len(prompts)} masks")
    plt.imshow(frame)
    for i, out_obj_id in enumerate(prompts):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask(prompt_mask[i], plt.gca(), obj_id=out_obj_id)
    plt.axis("off")
    plt.savefig(fig_path)
    plt.close(fig)


def viz_video_mask(frame, video_segments, frame_idx, fig_path):
    fig = plt.figure(figsize=(5, 5))
    num_masks = len(video_segments[frame_idx]["obj_ids"])
    plt.title(f"frame {frame_idx} v1: {num_masks} masks")
    plt.imshow(frame)
    for out_obj_id, out_mask in zip(
        video_segments[frame_idx]["obj_ids"],
        video_segments[frame_idx]["masks"],
    ):
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.axis("off")
    if not fig_path.exists():
        plt.savefig(fig_path)

    plt.title(f"frame {frame_idx} v2: {num_masks} masks")
    fig_path = str(fig_path).replace("v1_", "v2_")
    plt.savefig(fig_path)
    plt.close(fig)


def sample_point_prompt_from_segmentation(segmentation, num_points=1):
    assert segmentation.ndim == 2, f"segmentation should be a 2D numpy array, but got shape: {segmentation.shape}"

    labels = np.ones(num_points, np.int32)
    points = np.zeros((num_points, 2), np.float32)

    # to avoid selecting a point on the boundary, overwrite boundary to False
    segmentation[0, :] = False
    segmentation[-1, :] = False
    segmentation[:, 0] = False
    segmentation[:, -1] = False

    for i in range(num_points):
        distance_map = distance_transform_edt(segmentation)
        y, x = np.unravel_index(distance_map.argmax(), distance_map.shape)
        points[i] = np.array([x, y], dtype=np.float32)

        if i == num_points - 1:
            break

        distance = distance_map[y, x]
        window_size = int(distance // 3)
        segmentation[y - window_size: y + window_size + 1, x - window_size: x + window_size + 1] = False

    return points, labels


def compute_iou_batch(masks1, masks2):
    """
    masks1: (num_masks1, H, W)
    masks2: (num_masks2, H, W)
    return: (num_masks1, num_masks2)
    """
    masks1 = masks1[:, None]
    masks2 = masks2[None]
    intersection = (masks1 & masks2).sum(dim=(2, 3))
    union = (masks1 | masks2).sum(dim=(2, 3))
    iou = intersection / union
    return iou


def postprocess_auto_masks(
    auto_masks,
    frame=None,
    video_masks=None,
    min_mask_region_area=5,
    new_obj_mask_iou_thresh=0.3,
    new_obj_mask_overlap_thresh=0.95,
):
    """
    auto_masks: dict{segmentation, segmentation_tensor, area, predicted_iou, stability_score, bbox}
    video_masks: (num_masks2, H, W)
    return: (num_masks, H, W)
    """

    # remove overlap with the background and remove the mask if it has high overlap with the background
    if frame is not None:
        is_background = (frame == 0).all(dim=-1)
        filtered_masks = []
        for mask_result in auto_masks:
            mask = mask_result["segmentation_tensor"]
            area = mask_result["area"]
            overlap_with_background = (mask & is_background).sum().item()
            if overlap_with_background / area > new_obj_mask_overlap_thresh:
                continue

            refined_mask = mask & ~is_background
            mask_result["segmentation_tensor"] = refined_mask
            mask_result["area"] = refined_mask.sum().item()
            filtered_masks.append(mask_result)

        auto_masks = filtered_masks

    auto_masks = [mask_result for mask_result in auto_masks if mask_result["area"] > min_mask_region_area]
    if len(auto_masks) == 0:
        return []

    if video_masks is not None:
        # remove masks that have high iou with existing masks
        auto_masks_stacked = torch.stack([mask_result["segmentation_tensor"] for mask_result in auto_masks], dim=0)

        iou = compute_iou_batch(auto_masks_stacked, video_masks)            # (num_auto_masks, num_existing_objects)
        iou_valid = iou.max(dim=1).values < new_obj_mask_iou_thresh

        auto_masks = [mask_result for mask_result, valid in zip(auto_masks, iou_valid) if valid]
        if len(auto_masks) == 0:
            return []

        is_background = video_masks.sum(dim=(-2, -1)) > video_masks[0].numel() * 0.5
        occupied = video_masks[~is_background].any(dim=0)
    else:
        occupied = torch.zeros_like(auto_masks[0]["segmentation_tensor"])

    # apply non-overlapping constraint and filter out large masks that are completely covered small masks
    filtered_masks = []
    for mask_result in sorted(auto_masks, key=(lambda x: x["area"])):       # sort by area, small to large
        mask = mask_result["segmentation_tensor"]
        area = mask_result["area"]
        overlap_with_occupied = (mask & occupied).sum().item()
        if overlap_with_occupied / area > new_obj_mask_overlap_thresh:
            continue

        # mask should not overlap with occupied
        mask, occupied = mask & ~occupied, occupied | mask
        mask_result["segmentation_tensor"] = mask
        filtered_masks.append(mask_result)

    if len(filtered_masks) == 0:
        return []

    segmentation_tensor = torch.stack([mask_result["segmentation_tensor"] for mask_result in filtered_masks], dim=0)
    segmentation = segmentation_tensor.cpu().numpy()
    for mask_result, mask in zip(filtered_masks, segmentation):
        mask_result["segmentation"] = mask

    # sort by area, large to small
    filtered_masks = sorted(filtered_masks, key=(lambda x: x["area"]), reverse=True)

    return filtered_masks


def postprocess_video_masks(
    obj_ids,
    video_masks,
    frame,
):
    background = (frame == 0).all(dim=-1)
    video_masks = video_masks & ~background
    video_masks = torch.cat([background[None], video_masks], dim=0)
    
    # add background mask
    assert 0 not in obj_ids, f"The background mask should not be in obj_ids, but got: {obj_ids}"
    video_masks = {
        "obj_ids": [0] + obj_ids,
        "masks": video_masks.cpu().numpy(),             # (num_objects, H, W)
    }
    return video_masks


def load_video(
    episode_path,
    transform,
):
    image_paths = [
        path for path in episode_path.iterdir()
        if path.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    ]
    img_list = sorted(image_paths, key=lambda x: int(x.stem.split("_")[-1]))
    frame_num = len(img_list)

    original_frames = None
    for i in range(frame_num):
        frame = np.array(Image.open(img_list[i]).convert('RGB'))
        frame = torch.from_numpy(frame)

        if original_frames is None:
            original_frames = torch.zeros(frame_num, *frame.shape, dtype=torch.uint8)

        original_frames[i] = frame

    video_input_frames = (original_frames / 255.0).permute(0, 3, 1, 2)
    video_input_frames = transform(video_input_frames)

    return video_input_frames, original_frames


@dataclass
class Config:
    env_name: str = "plunder"                                             # bossfight, bigfish, plunder
    video_path: Path = Path("/scratch/cluster/zzwang_new/oc_ssm/data")
    num_episodes: int = 10

    cuda_id: dict = field(default_factory=lambda: {
        "bossfight": 1,
        "bigfish": 2,
        "plunder": 3,
    })
    split: dict = field(default_factory=lambda: {
        "bossfight": "ppg",
        "bigfish": "ppg",
        "plunder": "train_no_background",
    })

    # =============== Automatic Mask Generator ===============
    points_per_side_first_frame: dict = field(default_factory=lambda: {
        "bossfight": 32,
        "bigfish": 32,
        "plunder": 32,
    })
    points_per_side_other_frames: dict = field(default_factory=lambda: {
        "bossfight": 32,    # 2 pixels between each point
        "bigfish": 32,
        "plunder": 32,
    })
    pred_iou_thresh: dict = field(default_factory=lambda: {
        "bossfight": 0.6,
        "bigfish": 0.6,
        "plunder": 0.7,
    })
    stability_score_thresh: dict = field(default_factory=lambda: {
        "bossfight": 0.7,
        "bigfish": 0.7,
        "plunder": 0.7,
    })
    stability_score_offset: dict = field(default_factory=lambda: {
        "bossfight": 0.5,
        "bigfish": 0.5,
        "plunder": 0.5,
    })
    box_nms_thresh: dict = field(default_factory=lambda: {
        "bossfight": 0.5,
        "bigfish": 0.5,
        "plunder": 0.5,
    })
    # minimum num of pixels for a mask region to be considered valid
    min_mask_region_area: dict = field(default_factory=lambda: {
        "bossfight": 5,
        "bigfish": 5,
        "plunder": 5,
    })

    # =============== Video Predictor ===============
    num_point_prompt_per_segmentation: dict = field(default_factory=lambda: {
        "bossfight": 1,
        "bigfish": 1,
        "plunder": 1,
    })
    new_obj_detection_interval: dict = field(default_factory=lambda: {
        "bossfight": 5,
        "bigfish": 10,
        "plunder": 5,
    })
    # if the iou between the new mask and existing masks is larger than this threshold, we ignore the new mask
    new_obj_mask_iou_thresh: dict = field(default_factory=lambda: {
        "bossfight": 0.3,
        "bigfish": 0.3,
        "plunder": 0.3,
    })
    # if a new mask overlaps with existing masks more than this threshold, we ignore the new mask
    new_obj_mask_overlap_thresh: dict = field(default_factory=lambda: {
        "bossfight": 0.95,
        "bigfish": 0.95,
        "plunder": 0.95,
    })

    def __post_init__(self):
        self.cuda_id = self.cuda_id[self.env_name]
        self.split = self.split[self.env_name]

        self.points_per_side_first_frame = self.points_per_side_first_frame[self.env_name]
        self.points_per_side_other_frames = self.points_per_side_other_frames[self.env_name]
        self.pred_iou_thresh = self.pred_iou_thresh[self.env_name]
        self.stability_score_thresh = self.stability_score_thresh[self.env_name]
        self.stability_score_offset = self.stability_score_offset[self.env_name]
        self.box_nms_thresh = self.box_nms_thresh[self.env_name]
        self.min_mask_region_area = self.min_mask_region_area[self.env_name]

        self.num_point_prompt_per_segmentation = self.num_point_prompt_per_segmentation[self.env_name]
        self.new_obj_detection_interval = self.new_obj_detection_interval[self.env_name]
        self.new_obj_mask_iou_thresh = self.new_obj_mask_iou_thresh[self.env_name]
        self.new_obj_mask_overlap_thresh = self.new_obj_mask_overlap_thresh[self.env_name]


def main(cfg: Config):
    device = f"cuda:{cfg.cuda_id}"
    env_name = cfg.env_name
    min_mask_region_area = cfg.min_mask_region_area
    new_obj_detection_interval = cfg.new_obj_detection_interval

    out_dir = (f"out_{env_name}_"
               f"{cfg.pred_iou_thresh}_"
               f"{cfg.stability_score_thresh}_"
               f"{cfg.stability_score_offset}_"
               f"{cfg.box_nms_thresh}")
    out_dir = Path(out_dir.replace(".", ""))
    env_path = cfg.video_path / env_name / cfg.split
    epi_paths = list(env_path.iterdir())
    
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id=f"facebook/sam2-hiera-large",
        device=device,
        points_per_side=cfg.points_per_side_first_frame,
        points_per_batch=1024,
        pred_iou_thresh=cfg.pred_iou_thresh,
        stability_score_thresh=cfg.stability_score_thresh,
        stability_score_offset=cfg.stability_score_offset,
        box_nms_thresh=cfg.box_nms_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    if mask_generator.crop_n_layers > 0:
        raise NotImplementedError(
            "Crop n layers is not supported in this script due to the overwriting of mask_generator.points_grid",
        )

    predictor = SAM2VideoPredictor.from_pretrained(
        model_id="facebook/sam2-hiera-large",
        hydra_overrides_extra=[
            "++model.non_overlap_masks=True",
        ],
        device=device,
    )

    points_grid = None
    transform = T.Compose([
        T.Resize((predictor.image_size, predictor.image_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for epi_path in epi_paths[:cfg.num_episodes]:
        print(epi_path)

        # create output directory
        epi_idx = epi_path.name
        # if epi_idx != "eps_3":
        #     continue
        output_dir_path = out_dir / epi_idx
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # load frames
        video_frames_tensor, original_frames_tensor = load_video(epi_path, transform)
        original_frames = original_frames_tensor.numpy()
        original_frames_tensor = original_frames_tensor.to(device)
        original_shape = (original_frames.shape[2], original_frames.shape[1])   # (W, H)
        num_frames = len(original_frames)

        # ============ Init mask in the start frame as objects to track ============
        frame_idx = 0
        frame = original_frames[frame_idx]                                      # (H, W, 3), uint8
        frame_tensor = original_frames_tensor[frame_idx]                        # (H, W, 3), uint8

        # Init points grid for new object detection
        if points_grid is None:
            unit_points_grid_first_frame = build_point_grid(cfg.points_per_side_first_frame)
            unit_points_grid = build_point_grid(cfg.points_per_side_other_frames)
            points_grid = unit_points_grid * np.array(frame.shape[:2])
            points_grid = np.round(points_grid).astype(np.int32)

        mask_generator.point_grids = [unit_points_grid_first_frame]             # restore the point grid
        auto_masks = mask_generator.generate(frame)
        auto_masks = postprocess_auto_masks(
            auto_masks,
            frame=frame_tensor,
            min_mask_region_area=min_mask_region_area,
            new_obj_mask_iou_thresh=cfg.new_obj_mask_iou_thresh,
            new_obj_mask_overlap_thresh=cfg.new_obj_mask_overlap_thresh,
        )
        assert len(auto_masks) > 0, "The first frame should have at least one mask"
        print("Number of auto-masks:", len(auto_masks))

        # viz auto mask
        viz_auto_mask(
            frame, auto_masks,
            frame_idx=frame_idx,
            fig_path=output_dir_path / f"auto_mask_step_{frame_idx}.png",
        )

        # ============ Init video predictor with auto mask from the first frame ============
        inference_state = predictor.init_state(video_frames=video_frames_tensor, video_hw=original_shape)

        prompts, prompts_frame_idx = {}, frame_idx
        object_id_offset = 1
        for mask_idx, mask_result in enumerate(auto_masks):
            mask = mask_result["segmentation"]
            assert mask.mean() < 0.5, "The mask should not include the background"

            obj_id = object_id_offset + mask_idx

            # sample points from the auto mask and add them to the predictor as prompts
            points, labels = sample_point_prompt_from_segmentation(
                mask,
                num_points=cfg.num_point_prompt_per_segmentation,
            )
            prompts[obj_id] = points, labels

            _, obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=prompts_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
        object_id_offset = object_id_offset + len(auto_masks)

        # viz video predictor's post-processing of prompts
        prompt_mask = (out_mask_logits > 0.0).cpu().numpy()
        viz_prompt_mask(
            frame, prompts, prompt_mask,
            frame_idx, output_dir_path / f"prompt_mask_step_{frame_idx}.png",
        )

        # ===================== Do video segmentation =====================
        video_segments = {}
        for out_frame_idx, obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=new_obj_detection_interval,
        ):
            video_masks_tensor = out_mask_logits[:, 0] > 0.0
            video_segments[out_frame_idx] = postprocess_video_masks(
                deepcopy(obj_ids),              # to avoid obj_ids is reset by predictor.reset_state()
                video_masks_tensor,
                original_frames_tensor[out_frame_idx],
            )
        video_masks_cuda_frame_idx = out_frame_idx

        # visualize the results
        for out_frame_idx in range(0, new_obj_detection_interval + 1):
            frame = original_frames[out_frame_idx]
            fig_path = output_dir_path / f"results_v1_frame_{out_frame_idx}.jpg"
            viz_video_mask(frame, video_segments, out_frame_idx, fig_path)

        # For the rest of the frames, detect new objects and track them for every {new_obj_detection_interval} frames
        for frame_idx in range(new_obj_detection_interval, len(original_frames), new_obj_detection_interval):

            assert video_masks_cuda_frame_idx == frame_idx, \
                f"video_masks_cuda_frame_idx: {video_masks_cuda_frame_idx}, frame_idx: {frame_idx}"

            # ===================== Identify new objects =====================
            print(f"Frame {frame_idx}")
            frame = original_frames[frame_idx]                                  # (H, W, 3), uint8
            frame_tensor = original_frames_tensor[frame_idx]                    # (H, W, 3), uint8

            # Filter out points that belong to existing objects
            # get the masks from the previous frame
            video_masks = video_segments[frame_idx]["masks"]                    # (num_objects, H, W)
            video_obj_ids = np.array(video_segments[frame_idx]["obj_ids"]).astype(np.int64)

            video_masks_valid = video_masks.sum(axis=(1, 2)) > min_mask_region_area

            video_masks = video_masks[video_masks_valid]                        # (num_existing_objects, H, W)
            video_obj_ids = video_obj_ids[video_masks_valid]                    # (num_existing_objects)

            is_novel = ~video_masks.any(axis=0)                                 # (H, W)

            # video_masks[0] is the ground_truth background mask
            first_obj_id = video_segments[frame_idx]["obj_ids"][0]
            assert first_obj_id == 0, f"The first object id should be 0 (i.e., the background), but is {first_obj_id}"

            # ==== post-process the novel mask ====
            # filter out single pixel noise by edt
            is_novel = distance_transform_edt(is_novel) > 1.0                   # (H, W)

            ys, xs = points_grid[:, 1], points_grid[:, 0]                       # (num_points,), (num_points,)

            is_point_novel = is_novel[ys, xs]                                   # (num_points)
            novel_points = unit_points_grid[is_point_novel]                     # (num_novel_points, 2)

            # add small region that are not detected as novel points
            region_label, num_regions = label(is_novel)
            selected_regions = region_label[ys, xs]                             # (num_points,)
            selected_regions = np.unique(selected_regions[selected_regions > 0])

            missing_region_ids = [i for i in range(1, num_regions + 1) if i not in selected_regions]
            small_region_points = np.zeros((len(missing_region_ids), 2))
            for i, region_id in enumerate(missing_region_ids):
                region_mask = region_label == region_id
                region_point, _ = sample_point_prompt_from_segmentation(region_mask)
                small_region_points[i] = region_point[0]
            small_region_points = small_region_points / np.array([frame.shape[1], frame.shape[0]])

            novel_points = np.concatenate([novel_points, small_region_points], axis=0)

            print(f"Number of novel points: {len(novel_points)}")

            #  ===================== Get masks for novel points =====================
            auto_masks = []
            if len(novel_points):
                mask_generator.point_grids = [novel_points]                     # overwrite the point grid
                auto_masks = mask_generator.generate(frame)

                auto_masks = postprocess_auto_masks(
                    auto_masks,
                    frame=frame_tensor,
                    video_masks=video_masks_tensor,
                    min_mask_region_area=min_mask_region_area,
                    new_obj_mask_iou_thresh=cfg.new_obj_mask_iou_thresh,
                    new_obj_mask_overlap_thresh=cfg.new_obj_mask_overlap_thresh,
                )

            has_new_objects = len(auto_masks) > 0

            print(f"Number of new objects: {len(auto_masks)}")
            if has_new_objects:
                viz_auto_mask(
                    frame, auto_masks,
                    frame_idx=frame_idx,
                    fig_path=output_dir_path / f"auto_mask_step_{frame_idx}.png",
                )

            # ========================= video predictor initialize with auto mask =========================
            predictor.reset_state(inference_state)

            # re-add the prompts at t = {frame_idx - new_obj_detection_interval}
            for obj_id, (points, labels) in prompts.items():
                if obj_id == 0:
                    # skip adding prompt for the background, we will use the ground truth background mask
                    continue

                _, obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=prompts_frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )
            print(f"Frame {prompts_frame_idx}: re-add prompt for objects {list(prompts.keys())}")

            # add prompts for the existing and new objects at t = {frame_idx}
            if has_new_objects:
                new_obj_ids = np.arange(object_id_offset, object_id_offset + len(auto_masks))

                # Update prompts and object_id_offset only if there are new objects
                prompts, prompts_frame_idx = {}, frame_idx
                object_id_offset += len(auto_masks)

                all_obj_ids = np.concatenate([video_obj_ids, new_obj_ids], axis=0)
                all_mask = np.concatenate(
                    [
                        video_masks,                                                    # (num_existing_objects, H, W)
                        [mask_result["segmentation"] for mask_result in auto_masks],    # (num_new_objects, H, W)
                    ],
                    axis=0)

                for obj_id, segmentation in zip(all_obj_ids, all_mask):
                    if obj_id == 0:
                        # skip adding prompt for the background, we will use the ground truth background mask
                        continue

                    points, labels = sample_point_prompt_from_segmentation(
                        segmentation,
                        num_points=cfg.num_point_prompt_per_segmentation,
                    )
                    prompts[obj_id] = points, labels

                    _, obj_ids, out_mask_logits = predictor.add_new_points(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        points=points,
                        labels=labels,
                    )
                print(
                    f"Frame {prompts_frame_idx}: re-add prompt for objects, "
                    f"existing {[key for key in prompts.keys() if key in video_obj_ids]}, "
                    f"new {[key for key in prompts.keys() if key in new_obj_ids]}",
                )

                prompt_mask = (out_mask_logits > 0.0).cpu().numpy()
                viz_prompt_mask(
                    frame, prompts, prompt_mask,
                    frame_idx, output_dir_path / f"prompt_mask_step_{frame_idx}.png",
                )

            # ============================= Do video segmentation =============================
            if has_new_objects:
                # als redo segmentation at earlier frames because new objects may appear earlier than {frame_idx}
                start_frame_idx = frame_idx - new_obj_detection_interval + 1
                max_frame_num_to_track = 2 * new_obj_detection_interval - 1
            else:
                start_frame_idx = frame_idx
                max_frame_num_to_track = new_obj_detection_interval

            print(f"Segmentating frames {start_frame_idx} to {start_frame_idx + max_frame_num_to_track}")
            for out_frame_idx, obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            ):
                video_masks_tensor = out_mask_logits[:, 0] > 0.0
                video_segments[out_frame_idx] = postprocess_video_masks(
                    deepcopy(obj_ids),              # to avoid obj_ids is reset by predictor.reset_state()
                    video_masks_tensor,
                    original_frames_tensor[out_frame_idx],
                )

            video_masks_cuda_frame_idx = out_frame_idx
            obj_ids, masks = video_segments[frame_idx]["obj_ids"], video_segments[frame_idx]["masks"]
            print(f"Frame {out_frame_idx} obj_ids: {[obj_id for obj_id, mask in zip(obj_ids, masks) if mask.any()]}")
            print()

            # visualize the results
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames)
            for out_frame_idx in range(start_frame_idx, end_frame_idx):
                frame = original_frames[out_frame_idx]
                fig_path = output_dir_path / f"results_v1_frame_{out_frame_idx}.jpg"
                viz_video_mask(frame, video_segments, out_frame_idx, fig_path)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
