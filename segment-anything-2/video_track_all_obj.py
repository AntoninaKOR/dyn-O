import os
import tyro
import h5py
import minari
import shutil
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from filelock import FileLock
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt, label, generate_binary_structure

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# use bfloat16 for the entire script
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import build_point_grid

from hdf5_utils import find_latest_backup, is_hdf5_file_readable
from test_scripts.test_video_track_all_obj import viz_video_mask, viz_auto_mask, viz_prompt_mask
from video_dataset.procgen_minari import ProcgenMinari

# compress UserWarning
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class SamConfig:
    points_per_side_first_frame: int = 32
    points_per_side_other_frames: int = 32
    pred_iou_thresh: float = 0.6
    stability_score_thresh: float = 0.7
    stability_score_offset: float = 0.5
    box_nms_thresh: float = 0.5
    min_mask_region_area: int = 10
    new_obj_detection_interval: int = 5
    new_obj_mask_iou_thresh: float = 0.3
    new_obj_mask_overlap_thresh: float = 0.95
    use_mask_merge: bool = False


@dataclass
class Config:
    data_path: str | Path
    visualize: bool = False
    cuda_ids: list = field(default_factory=lambda: [0])
    force_recompute: bool = False
    backup_freq: int = 5

    # =============== Automatic Mask Generator ===============
    skiing_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=False,
        new_obj_detection_interval=10,
    ))
    boxing_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=False,
        new_obj_detection_interval=10,
    ))
    bigfish_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=True,
    ))
    bossfight_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=True,
    ))
    caveflyer_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        stability_score_offset=0.8,
        min_mask_region_area=15,
        new_obj_detection_interval=2,
        use_mask_merge=True,
    ))
    climber_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        points_per_side_first_frame=32,
        points_per_side_other_frames=32,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.7,
        stability_score_offset=0.5,
        box_nms_thresh=0.5,
        min_mask_region_area=5,
        new_obj_detection_interval=5,
        new_obj_mask_iou_thresh=0.3,
        new_obj_mask_overlap_thresh=0.95,
    ))
    coinrun_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        min_mask_region_area=10,
    ))
    dodgeball_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=True,
    ))
    fruitbot_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        min_mask_region_area=5,
        use_mask_merge=True,
    ))
    jumper_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        points_per_side_first_frame=32,
        points_per_side_other_frames=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        min_mask_region_area=10,
        new_obj_mask_iou_thresh=0.3,
        new_obj_mask_overlap_thresh=0.95,
    ))
    leaper_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        points_per_side_first_frame=32,
        points_per_side_other_frames=32,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        box_nms_thresh=0.7,
        min_mask_region_area=5,
        new_obj_detection_interval=5,
        new_obj_mask_iou_thresh=0.3,
        new_obj_mask_overlap_thresh=0.95,
    ))
    ninja_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        stability_score_offset=0.7,
        min_mask_region_area=10,
    ))
    plunder_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        points_per_side_first_frame=32,
        points_per_side_other_frames=64,
        min_mask_region_area=5,
        new_obj_detection_interval=10,
        new_obj_mask_iou_thresh=0.3,
        new_obj_mask_overlap_thresh=0.95,
        use_mask_merge=True,
    ))
    rt1_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        stability_score_offset=0.8,
        min_mask_region_area=15,
        new_obj_detection_interval=5,
        use_mask_merge=False,
    ))
    starpilot_cfg: SamConfig = field(default_factory=lambda: SamConfig(
        pred_iou_thresh=0.7,
        stability_score_thresh=0.7,
        stability_score_offset=0.8,
        min_mask_region_area=15,
        new_obj_detection_interval=2,
        use_mask_merge=True,
    ))
    sam_cfg: SamConfig = None

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.lock_file_name = self.data_path / "sam_masks.lock"
        self.sam_mask_file_name = self.data_path / "data" / "sam_masks.h5"
        self.sam_backup_file_names = [
            self.data_path / "data" / f"sam_masks_backup_{i}.h5" for i in range(2)
        ]
        self.cur_sam_backup_file_name = self.sam_backup_file_names[0]
        self.tmp_mask_file_name = self.data_path / "data" / "tmp_masks.h5"
        self.dataset_id = self.data_path.stem

        # assume dataset_id is in the format of f"procgen-{env_id}-v{data_version}"
        self.env_name = env_name = self.dataset_id.split("-")[1]
        self.sam_cfg = getattr(self, f"{env_name}_cfg")


def sample_point_prompt_from_segmentation(segmentation):
    assert (
        segmentation.ndim == 2
    ), f"segmentation should be a 2D numpy array, but got shape: {segmentation.shape}"
    assert segmentation.any(), "The segmentation should not be empty"

    labels = np.ones(5, np.int32)
    points = np.zeros((5, 2), np.float32)

    # to avoid selecting a point on the boundary, overwrite boundary to False
    segmentation[0, :] = False
    segmentation[-1, :] = False
    segmentation[:, 0] = False
    segmentation[:, -1] = False

    distance_map = distance_transform_edt(segmentation)
    y, x = np.unravel_index(distance_map.argmax(), distance_map.shape)
    points[0] = np.array([x, y], dtype=np.float32)

    y_indices, x_indices = np.where(distance_map >= 2)
    if len(y_indices) == 0:
        return points[:1], labels[:1]

    for i, at_boundary in enumerate([x_indices == np.min(x_indices), x_indices == np.max(x_indices),
                                     y_indices == np.min(y_indices), y_indices == np.max(y_indices)]):
        idxes = np.where(at_boundary)[0]
        idx = idxes[len(idxes) // 2]
        points[i + 1] = np.array([x_indices[idx], y_indices[idx]], dtype=np.float32)

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


def merge_masks(video_masks: np.ndarray[bool], auto_masks, min_mask_region_area=5):
    """
    merge masks that are connected
    video_masks: (num_masks, H, W)
    masks2: (num_masks2, H, W)
    return: (num_masks1 + num_masks2, H, W)
    """
    device = auto_masks[0]["segmentation_tensor"].device
    structure = generate_binary_structure(2, 2)
    combined_mask = np.any([auto_mask["segmentation"] for auto_mask in auto_masks], axis=0).astype(int)

    if video_masks is not None:
        for i, video_mask in enumerate(video_masks):
            if not video_mask.any() or i == 0:
                # skip the video mask if it is the background
                continue
            combined_mask[video_mask] = 1
            labeled_mask, _ = label(combined_mask, structure=structure)
            video_id = labeled_mask[video_mask][0]          # all new masks are connected to video_mask
            new_video_mask = (labeled_mask == video_id)
            video_masks[i] = new_video_mask
            combined_mask[new_video_mask] = 0

    auto_masks = []
    labeled_mask, num_features = label(combined_mask, structure=structure)
    for component_id in range(1, num_features + 1):
        region = (labeled_mask == component_id)
        area = region.sum()
        if area <= min_mask_region_area:
            continue
        auto_masks.append({
            "segmentation_tensor": torch.tensor(region, dtype=torch.bool, device=device),
            "segmentation": region,
            "area": region.sum(),
        })
    return video_masks, auto_masks


def postprocess_auto_masks(
    auto_masks,
    frame=None,
    video_masks=None,
    video_masks_tensor=None,
    min_mask_region_area=5,
    new_obj_mask_iou_thresh=0.3,
    new_obj_mask_overlap_thresh=0.95,
    use_mask_merge=False,
):
    """
    auto_masks: dict{segmentation, segmentation_tensor, area, predicted_iou, stability_score, bbox}
    video_masks_tensor: (num_video_masks, H, W)
    return: (num_masks, H, W)
    """
    assert (video_masks is None) == (video_masks_tensor is None), \
        "video_masks and video_masks_tensor should be both None or not None"
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

            refined_mask_tensor = mask & ~is_background
            refined_mask = refined_mask_tensor.cpu().numpy()
            area = refined_mask.sum()
            mask_result["segmentation_tensor"] = refined_mask_tensor
            mask_result["segmentation"] = refined_mask
            mask_result["area"] = area

            if area <= min_mask_region_area:
                continue

            # remove masks that are too small compared to the bbox
            coords = np.argwhere(refined_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            height = y_max - y_min + 1
            width = x_max - x_min + 1
            bbox_size = height * width
            if area / bbox_size < 0.2:
                continue
            filtered_masks.append(mask_result)

        auto_masks = filtered_masks

    if len(auto_masks) == 0:
        return [], video_masks

    if video_masks_tensor is not None:
        # remove masks that have high iou with existing masks
        auto_masks_stacked = torch.stack(
            [mask_result["segmentation_tensor"] for mask_result in auto_masks], dim=0
        )

        iou = compute_iou_batch(
            auto_masks_stacked, video_masks_tensor
        )  # (num_auto_masks, num_existing_objects)
        iou_valid = iou.max(dim=1).values < new_obj_mask_iou_thresh

        auto_masks = [mask_result for mask_result, valid in zip(auto_masks, iou_valid) if valid]
        if len(auto_masks) == 0:
            return [], video_masks

        # video_masks[0] is the background mask
        occupied = video_masks_tensor[1:].any(dim=0)
    else:
        occupied = torch.zeros_like(auto_masks[0]["segmentation_tensor"])

    # merge masks that are connected
    if use_mask_merge:
        video_masks, auto_masks = merge_masks(video_masks, auto_masks, min_mask_region_area)
    else:
        # apply non-overlapping constraint and filter out small masks that are completely covered large masks
        filtered_masks = []
        for mask_result in sorted(auto_masks, key=(lambda x: x["area"]), reverse=True):  # sort by area, small to large
            mask = mask_result["segmentation_tensor"]
            area = mask_result["area"]
            overlap_with_occupied = (mask & occupied).sum().item()
            if overlap_with_occupied / area > new_obj_mask_overlap_thresh:
                continue

            # mask should not overlap with occupied
            mask, occupied = mask & ~occupied, occupied | mask
            mask_result["segmentation_tensor"] = mask
            filtered_masks.append(mask_result)

        auto_masks = filtered_masks

    # sort by area, large to small
    auto_masks = sorted(auto_masks, key=(lambda x: x["area"]), reverse=True)
    return auto_masks, video_masks


def postprocess_video_masks(
    obj_ids,
    video_masks_tensor,
    frame,
):
    background = (frame == 0).all(dim=-1)
    video_masks_tensor = video_masks_tensor & ~background
    video_masks_valid = video_masks_tensor.sum(dim=(-2, -1)) > 0
    obj_ids = [obj_id for obj_id, valid in zip(obj_ids, video_masks_valid) if valid]
    video_masks_tensor = video_masks_tensor[video_masks_valid]

    video_masks_tensor = torch.cat([background[None], video_masks_tensor], dim=0)

    # add background mask
    assert 0 not in obj_ids, f"The background mask should not be in obj_ids, but got: {obj_ids}"
    video_masks = {
        "obj_ids": [0] + obj_ids,
        "masks": video_masks_tensor.cpu().numpy(),  # (num_objects, H, W)
    }
    return video_masks, video_masks_tensor


def compute_episode_video_masks(
    cfg: Config,
    batch: dict,
    mask_generator: SAM2AutomaticMaskGenerator,
    predictor: SAM2VideoPredictor,
    unit_points_grid_first_frame: np.ndarray,
    unit_points_grid: np.ndarray,
    episode_idx: int = 0,
):
    device = predictor.device
    sam_cfg = cfg.sam_cfg
    min_mask_region_area = sam_cfg.min_mask_region_area
    new_obj_detection_interval = sam_cfg.new_obj_detection_interval

    # load frames
    processed_frames = batch["processed_frames"].to(device)  # (T, C, H, W), float32 in [0, 1]
    original_frames = batch["original_frames"].to(device)  # (T, H, W, 3), uint8

    num_frames, H, W, _ = original_frames.shape

    # ============ Init mask in the start frame as objects to track ============
    frame_idx = 0
    frame_tensor = original_frames[frame_idx]  # (H, W, 3), uint8

    # Init points grid for new object detection
    points_grid = unit_points_grid * np.array([H, W])
    points_grid = np.round(points_grid).astype(np.int32)

    mask_generator.point_grids = [unit_points_grid_first_frame]  # restore the point grid
    auto_masks = mask_generator.generate(frame_tensor)
    auto_masks, _ = postprocess_auto_masks(
        auto_masks,
        frame=frame_tensor,
        min_mask_region_area=min_mask_region_area,
        new_obj_mask_iou_thresh=sam_cfg.new_obj_mask_iou_thresh,
        new_obj_mask_overlap_thresh=sam_cfg.new_obj_mask_overlap_thresh,
        use_mask_merge=sam_cfg.use_mask_merge,
    )
    assert len(auto_masks) > 0, "The first frame should have at least one mask"

    if cfg.visualize:
        # viz auto mask
        output_dir_path = Path(f"viz/{str(cfg.data_path).split(os.sep)[-1]}/eps_{episode_idx}")
        os.makedirs(output_dir_path, exist_ok=True)
        viz_auto_mask(
            frame_tensor.cpu().numpy(),
            np.arange(len(auto_masks)) + 1,
            auto_masks,
            frame_idx=frame_idx,
            fig_path=output_dir_path / f"auto_mask_step_{frame_idx}.png",
        )

    # ============ Init video predictor with auto mask from the first frame ============
    inference_state = predictor.init_state(video_frames=processed_frames, video_hw=(H, W))

    prompts, prompts_frame_idx = {}, frame_idx
    object_id_offset = 1
    for mask_idx, mask_result in enumerate(auto_masks):
        mask = mask_result["segmentation"]
        # assert mask_result["area"] < mask_tensor.numel() * 0.5, "The mask should not include the background"

        obj_id = object_id_offset + mask_idx

        # sample points from the auto mask and add them to the predictor as prompts
        points, labels = sample_point_prompt_from_segmentation(mask)
        prompts[obj_id] = points, labels

        if np.any(points == np.zeros(2)):
            continue

        _, obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=prompts_frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
    object_id_offset = object_id_offset + len(auto_masks)

    if cfg.visualize:
        # viz video predictor's post-processing of prompts
        prompt_mask = (out_mask_logits > 0.0).cpu().numpy()
        viz_prompt_mask(
            frame_tensor.cpu().numpy(), prompts, prompt_mask,
            frame_idx, output_dir_path / f"prompt_mask_step_{frame_idx}.png",
        )

    # ===================== Do video segmentation for first several frames =====================
    video_segments = {}
    for out_frame_idx, obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=new_obj_detection_interval,
    ):
        raw_video_masks_tensor = out_mask_logits[:, 0] > 0.0
        video_segments[out_frame_idx], video_masks_tensor = postprocess_video_masks(
            # deepcopy to avoid obj_ids is overwritten during predictor.reset_state()
            deepcopy(obj_ids),
            raw_video_masks_tensor,
            original_frames[out_frame_idx],
        )
    video_masks_frame_idx = out_frame_idx

    if cfg.visualize:
        # visualize the results
        for out_frame_idx in range(0, min(new_obj_detection_interval + 1, num_frames)):
            frame = original_frames[out_frame_idx]
            fig_path = output_dir_path / f"results_v1_frame_{out_frame_idx}.jpg"
            viz_video_mask(frame.cpu().numpy(), video_segments, out_frame_idx, fig_path)

    # ================================== For the rest of the frames ===================================
    # ======== Detect new objects and track them for every {new_obj_detection_interval} frames ========
    for frame_idx in range(new_obj_detection_interval, num_frames, new_obj_detection_interval):
        assert (
            video_masks_frame_idx == frame_idx
        ), f"video_masks_frame_idx: {video_masks_frame_idx}, frame_idx: {frame_idx}"

        # ===================== Identify new objects =====================
        frame_tensor = original_frames[frame_idx]  # (H, W, 3), torch.uint8

        # Filter out points that belong to existing objects
        # get the masks from the previous frame
        video_masks = video_segments[frame_idx]["masks"]  # (num_objects, H, W), np.bool
        video_obj_ids = np.array(video_segments[frame_idx]["obj_ids"]).astype(np.int64)

        video_masks_valid = video_masks.sum(axis=(1, 2)) > min_mask_region_area

        video_masks = video_masks[video_masks_valid]  # (num_existing_objects, H, W)
        video_obj_ids = video_obj_ids[video_masks_valid]  # (num_existing_objects)

        unoccupied = ~video_masks.any(axis=0)  # (H, W)

        # video_masks[0] is the ground_truth background mask
        first_obj_id = video_segments[frame_idx]["obj_ids"][0]
        assert (
            first_obj_id == 0
        ), f"The first object id should be 0 (i.e., the background), but is {first_obj_id}"

        # ==== post-process the novel mask ====
        # filter out single pixel noise by edt
        unoccupied = distance_transform_edt(unoccupied) > 1.0                   # (H, W)

        ys, xs = points_grid[:, 0], points_grid[:, 1]                           # (num_points,), (num_points,)

        is_point_novel = unoccupied[ys, xs]                                     # (num_points)
        novel_points = unit_points_grid[is_point_novel]                         # (num_novel_points, 2)

        # add small region that are not detected as novel points
        region_label, num_regions = label(unoccupied)
        selected_regions = region_label[ys, xs]                                 # (num_points,)
        selected_regions = np.unique(selected_regions[selected_regions > 0])

        missing_region_ids = [i for i in range(1, num_regions + 1) if i not in selected_regions]
        small_region_points = np.zeros((len(missing_region_ids), 2))
        for i, region_id in enumerate(missing_region_ids):
            region_mask = region_label == region_id
            region_point, _ = sample_point_prompt_from_segmentation(region_mask)
            small_region_points[i] = region_point[0]
        small_region_points = small_region_points / np.array([W, H])

        novel_points = np.concatenate([novel_points, small_region_points], axis=0)

        #  ===================== Get masks for novel points =====================
        auto_masks = []
        if len(novel_points):
            mask_generator.point_grids = [novel_points]  # overwrite the point grid
            auto_masks = mask_generator.generate(frame_tensor)
            auto_masks, video_masks = postprocess_auto_masks(
                auto_masks,
                frame=frame_tensor,
                video_masks=video_masks,
                video_masks_tensor=video_masks_tensor,
                min_mask_region_area=min_mask_region_area,
                new_obj_mask_iou_thresh=sam_cfg.new_obj_mask_iou_thresh,
                new_obj_mask_overlap_thresh=sam_cfg.new_obj_mask_overlap_thresh,
                use_mask_merge=sam_cfg.use_mask_merge,
            )

        has_new_objects = len(auto_masks) > 0
        if has_new_objects and cfg.visualize:
            viz_auto_mask(
                frame_tensor.cpu().numpy(),
                object_id_offset + np.arange(len(auto_masks)),
                auto_masks,
                frame_idx=frame_idx,
                fig_path=output_dir_path / f"auto_mask_step_{frame_idx}.png",
            )

        # ========================= video predictor initialize with auto mask =========================
        predictor.reset_state(inference_state)

        # re-add the prompts at t = {frame_idx - new_obj_detection_interval}
        for obj_id, (points, labels) in prompts.items():
            # skip adding prompt for the background, we will use the ground truth background mask
            if obj_id == 0:
                continue

            _, obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=prompts_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )

        # add prompts for the existing and new objects at t = {frame_idx}
        if has_new_objects:
            new_obj_ids = object_id_offset + np.arange(len(auto_masks))

            # Update prompts and object_id_offset only if there are new objects
            prompts, prompts_frame_idx = {}, frame_idx
            object_id_offset += len(auto_masks)

            all_obj_ids = np.concatenate([video_obj_ids, new_obj_ids], axis=0)
            all_masks = list(video_masks) + [mask_result["segmentation"] for mask_result in auto_masks]

            assert len(all_obj_ids) == len(all_masks), "The number of masks should match the number of object ids"
            for obj_id, segmentation in zip(all_obj_ids, all_masks):
                # skip adding prompt for the background, we will use the ground truth background mask
                if obj_id == 0:
                    continue

                points, labels = sample_point_prompt_from_segmentation(segmentation)
                prompts[obj_id] = points, labels

                if np.any(points == np.zeros(2)):
                    continue

                _, obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            if cfg.visualize:
                prompt_mask = (out_mask_logits > 0.0).cpu().numpy()
                viz_prompt_mask(
                    frame_tensor.cpu().numpy(), prompts, prompt_mask,
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

        for out_frame_idx, obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
        ):
            raw_video_masks_tensor = out_mask_logits[:, 0] > 0.0
            video_segments[out_frame_idx], video_masks_tensor = postprocess_video_masks(
                # deepcopy to avoid obj_ids is overwritten during predictor.reset_state()
                deepcopy(obj_ids),
                raw_video_masks_tensor,
                original_frames[out_frame_idx],
            )

        video_masks_frame_idx = out_frame_idx

        if cfg.visualize:
            # visualize the results
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames)
            for out_frame_idx in range(start_frame_idx, end_frame_idx):
                frame = original_frames[out_frame_idx].cpu().numpy()
                fig_path = output_dir_path / f"results_v1_frame_{out_frame_idx}.jpg"
                viz_video_mask(frame, video_segments, out_frame_idx, fig_path)

    # Convert video_segments to video_masks, List[(num_objects, H, W)] -> (T, H, W)
    video_masks = -1 * np.ones((num_frames, H, W), np.int32)
    max_num_objs_in_single_frame = -1

    for frame_idx, video_segment in video_segments.items():
        obj_ids = video_segment["obj_ids"]
        masks = video_segment["masks"]
        assert len(obj_ids) == len(masks), "The number of masks should match the number of object ids"
        for obj_id, mask in zip(obj_ids, masks):
            video_masks[frame_idx, mask] = obj_id

        num_obj_in_frame = masks.any(axis=(1, 2)).sum()
        max_num_objs_in_single_frame = max(max_num_objs_in_single_frame, num_obj_in_frame)

    video_mask_results = {
        "video_masks": video_masks,
        "num_objs": object_id_offset,
        "max_num_objs_in_single_frame": max_num_objs_in_single_frame,
    }

    return video_mask_results


def worker(
    cfg: Config,
    rank: int,
    episode_idxes: np.ndarray,
):
    device = f"cuda:0"
    sam_cfg = cfg.sam_cfg

    # init sam2 model
    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id=f"facebook/sam2-hiera-large",
        device=device,
        points_per_side=sam_cfg.points_per_side_first_frame,
        points_per_batch=1024,
        pred_iou_thresh=sam_cfg.pred_iou_thresh,
        stability_score_thresh=sam_cfg.stability_score_thresh,
        stability_score_offset=sam_cfg.stability_score_offset,
        box_nms_thresh=sam_cfg.box_nms_thresh,
        min_mask_region_area=sam_cfg.min_mask_region_area,
    )

    predictor = SAM2VideoPredictor.from_pretrained(
        model_id="facebook/sam2-hiera-large",
        hydra_overrides_extra=[
            "++model.non_overlap_masks=True",
        ],
        device=device,
    )

    if mask_generator.crop_n_layers > 0:
        raise NotImplementedError(
            "Crop n layers is not supported in this script due to the overwriting of mask_generator.points_grid",
        )

    # load minari dataset
    dataset = ProcgenMinari(cfg, episode_idxes=episode_idxes, resize_to=predictor.image_size)
    data_loader = DataLoader(
        dataset, batch_size=1, num_workers=0, pin_memory=True, collate_fn=dataset.collate_fn
    )

    # to be initialized after getting image size
    unit_points_grid_first_frame = build_point_grid(sam_cfg.points_per_side_first_frame)
    unit_points_grid = build_point_grid(sam_cfg.points_per_side_other_frames)

    # Create a lock file specific to the HDF5 file
    lock = FileLock(cfg.lock_file_name)

    for epi_idx, batch in tqdm(enumerate(data_loader), total=len(episode_idxes), disable=rank != 0):
        episode_data = batch["episode_data"]  # dict[str, Any]
        video_mask_results = compute_episode_video_masks(
            cfg,
            batch,
            mask_generator,
            predictor,
            unit_points_grid_first_frame,
            unit_points_grid,
            episode_idx=epi_idx,
        )
        video_masks = video_mask_results["video_masks"]

        # update SAM mask, terminations, and truncations
        with lock:
            with h5py.File(cfg.tmp_mask_file_name, "a") as f:
                ep_idx = episode_data["id"]
                group_name = f"episode_{ep_idx}"
                episode_group = f[group_name] if group_name in f else f.create_group(group_name)

                # update terminations and truncations
                # episode_group["terminations"][:] = episode_data["terminations"]
                # episode_group["truncations"][:] = episode_data["truncations"]

                # add or update video masks
                if "sam_masks" in episode_group:
                    dset = episode_group["sam_masks"]
                    if dset.shape == video_masks.shape:
                        dset[:] = video_masks  # Overwrite the data in-place
                    else:
                        # If the shape of new_data is different, we need to resize the dataset first
                        dset.resize(video_masks.shape)
                        dset[:] = video_masks
                else:
                    dshape = video_masks.shape[1:]
                    episode_group.create_dataset(
                        "sam_masks",
                        data=video_masks,
                        dtype=None,
                        chunks=(1, *dshape),
                        maxshape=(None, *dshape),
                        compression="gzip",
                        # compression_opts=9,
                    )

                # update mask metadata
                episode_group.attrs["num_objs"] = video_mask_results["num_objs"]
                episode_group.attrs["max_num_objs_in_single_frame"] = video_mask_results[
                    "max_num_objs_in_single_frame"
                ]

        if rank == 0 and epi_idx % cfg.backup_freq == 0:
            # save the file to the backup in a rotation manner
            temp_backup = str(cfg.sam_mask_file_name) + ".tmp"
            shutil.copy(cfg.tmp_mask_file_name, temp_backup)
            os.replace(temp_backup, cfg.cur_sam_backup_file_name)
            cfg.cur_sam_backup_file_name = cfg.sam_backup_file_names[
                (cfg.sam_backup_file_names.index(cfg.cur_sam_backup_file_name) + 1) % len(cfg.sam_backup_file_names)
            ]


def main(cfg: Config):
    os.environ["MINARI_DATASETS_PATH"] = str(cfg.data_path.parent)
    print(f"Minari root: {os.environ['MINARI_DATASETS_PATH']}")

    # get all episode indexes
    dataset = minari.load_dataset(cfg.dataset_id)
    episode_indices = dataset.episode_indices

    # find the latest valid backup file if the main file is not readable
    latest_backup = find_latest_backup(cfg.sam_backup_file_names + [cfg.tmp_mask_file_name])
    if latest_backup is not None:
        shutil.copy(latest_backup, cfg.sam_mask_file_name)

    # remove existing episode indices
    if cfg.force_recompute and cfg.sam_mask_file_name.exists():
        os.remove(cfg.sam_mask_file_name)

    if not cfg.sam_mask_file_name.exists():
        existing_episodes = []
        unfinished_episodes = episode_indices
    else:
        existing_episodes = []
        unfinished_episodes = []
        with h5py.File(cfg.sam_mask_file_name, "r") as f:
            for ep_idx in episode_indices:
                group_name = f"episode_{ep_idx}"
                if group_name in f:
                    existing_episodes.append(ep_idx)
                else:
                    unfinished_episodes.append(ep_idx)
    print("Total number of episodes:", len(episode_indices))
    print("Number of existing episodes:", len(existing_episodes))
    print("Number of unfinished episodes:", len(unfinished_episodes))
    if len(unfinished_episodes) == 0:
        return

    if cfg.sam_mask_file_name.exists():
        shutil.copy(cfg.sam_mask_file_name, cfg.tmp_mask_file_name)

    # split SAM preprocessing into multiple processes
    ranks = list(range(len(cfg.cuda_ids)))
    mp.set_start_method("spawn")
    episode_indices = np.array_split(unfinished_episodes, len(cfg.cuda_ids))

    processes = []
    for rank in ranks:
        # worker(cfg, rank, episode_indices[rank])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.cuda_ids[rank])
        p = mp.Process(target=worker, args=(cfg, rank, episode_indices[rank]))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # remove the lock file and rename the tmp file to the final file
    if cfg.lock_file_name.exists():
        os.remove(cfg.lock_file_name)
    if cfg.sam_mask_file_name.exists():
        os.remove(cfg.sam_mask_file_name)
    os.rename(cfg.tmp_mask_file_name, cfg.sam_mask_file_name)


if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
