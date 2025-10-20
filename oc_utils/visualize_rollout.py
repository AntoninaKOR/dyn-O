import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from encoder.solv_sam.utils import visualize_mask
from dynamics.mamba.generation import Prediction


@torch.inference_mode()
def visualize_rollout(config, encoder, input_batch, predictions: Prediction):
    num_viz_rollouts = config.logging.num_viz_rollouts
    rollout_viz_steps = np.array(config.logging.rollout_viz_steps, dtype=int)
    assert np.all(rollout_viz_steps > 0)

    # ================= Prepare original image and mask =================
    seq_len = input_batch.observations.shape[1]
    viz_idxes = np.array([idx for idx in rollout_viz_steps if idx < seq_len])
    (
        next_slots_pred,
        next_slots_visible_pred,
    ) = map(
        lambda x: None if x is None else x[:num_viz_rollouts, viz_idxes - 1],
        [
            predictions.next_slots_pred,
            predictions.next_slots_visible_pred,
        ]
    )

    # also visualize the current step where rollout starts
    viz_idxes = np.insert(viz_idxes, 0, 0, axis=0)
    (
        observations,
        slots,
        enc_feat,
        slots_visible,
        padding_mask,
    ) = map(
        lambda x: None if x is None else x[:num_viz_rollouts, viz_idxes],
        [
            input_batch.observations,
            input_batch.slots,
            input_batch.enc_feat,
            input_batch.slots_visible,
            input_batch.padding_mask,
        ]
    )

    encoder = encoder.module if isinstance(encoder, DDP) else encoder

    if config.dynamics.patch_as_slot:
        rgb_rec = encoder.decode_enc_feat_to_rgb(enc_feat)
        rollout_rgb_rec = encoder.decode_enc_feat_to_rgb(next_slots_pred)

        rollout_rgb_rec = torch.cat([rgb_rec[:, :1], rollout_rgb_rec], dim=1)
    else:
        if encoder.decode_segmentation:
            patch_masks, segmentations, rgb_rec = encoder.decode(
                slots, slots_visible,
                modes=["patch_mask", "segmentation", "rgb_rec"],
            )
            rollout_patch_masks, rollout_segmentations, rollout_rgb_rec = encoder.decode(
                next_slots_pred, next_slots_visible_pred,
                modes=["patch_mask", "segmentation", "rgb_rec"],
            )
        else:
            patch_masks, rgb_rec = encoder.decode(
                slots, slots_visible,
                modes=["patch_mask", "rgb_rec"],
            )
            rollout_patch_masks, rollout_rgb_rec = encoder.decode(
                next_slots_pred, next_slots_visible_pred,
                modes=["patch_mask", "rgb_rec"],
            )
            segmentations = rollout_segmentations = None

        rollout_patch_masks = torch.cat([patch_masks[:, :1], rollout_patch_masks], dim=1)
        rollout_rgb_rec = torch.cat([rgb_rec[:, :1], rollout_rgb_rec], dim=1)

        if segmentations is not None:
            rollout_segmentations = torch.cat([segmentations[:, :1], rollout_segmentations], dim=1)

    # ================= Visualization =================
    images = []
    for i in range(min(num_viz_rollouts, len(observations))):
        padding_mask_i = padding_mask[i]                            # (1 + num_viz_rollout_steps,)
        if torch.all(padding_mask_i):
            continue

        observations_i = observations[i][~padding_mask_i]

        rgb_rec_i = rgb_rec[i][~padding_mask_i]
        rollout_rgb_rec_i = rollout_rgb_rec[i][~padding_mask_i]

        if config.dynamics.patch_as_slot:
            img_masks = [
                (observations_i, None),
                (rgb_rec_i, None),
                (rollout_rgb_rec_i, None),
            ]
        else:
            patch_masks_i = patch_masks[i][~padding_mask_i]
            rollout_patch_masks_i = rollout_patch_masks[i][~padding_mask_i]

            img_masks = [
                (observations_i, None),
                (rgb_rec_i, None),
                (rollout_rgb_rec_i, None),
                (observations_i, patch_masks_i),
                (observations_i, rollout_patch_masks_i),
            ]

            if segmentations is not None:
                segmentations_i = segmentations[i][~padding_mask_i]
                rollout_segmentations_i = rollout_segmentations[i][~padding_mask_i]

                img_masks.extend([
                    (observations_i, segmentations_i),
                    (observations_i, rollout_segmentations_i),
                ])

        seq_len = observations_i.shape[0]
        image_w_mask = []
        for img, mask in img_masks:
            image_w_mask.append(visualize_mask(img, mask, num_images=seq_len))

        image = torch.cat(image_w_mask, dim=1)
        image = image.permute(1, 2, 0).cpu().numpy()

        images.append(image)
    return images
