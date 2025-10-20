import h5py
import numpy as np
from numba import njit

import torch
from torch.utils.data import Dataset


class EpisodeRgbDataset(Dataset):
    def __init__(self, dataset, dataset_path, sam_mask_path, config):
        self.minari_dataset = dataset
        self.minari_dataset_path = dataset_path
        self.sam_mask_path = sam_mask_path
        self.encoder_num_slots = config.encoder.num_slots

    def __len__(self):
        return len(self.minari_dataset)

    def __getitem__(self, idx):
        ep_idx = self.minari_dataset.episode_indices[idx]
        with h5py.File(self.minari_dataset_path, "r") as file:
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)

            id = ep_group.attrs["id"]

            observations = ep_group["observations"][:]                  # (T, H, W, C)
            if ep_group.attrs["total_steps"] == 1:
                observations = observations[None]                       # (1, H, W, 3)

        with h5py.File(self.sam_mask_path, "r") as file:
            ep_idx = self.minari_dataset.episode_indices[idx]
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)

            mask = ep_group["sam_masks"][:]                             # (T, H, W)
            num_objs = ep_group.attrs["num_objs"]

        # (T, num_slots, H, W), (T, num_objs)
        T, H, W = mask.shape
        binary_mask = np.zeros((T, self.encoder_num_slots, H, W), dtype=bool)
        obj_mask = np.zeros((T, num_objs), dtype=bool)
        binary_mask, obj_mask = get_binary_mask(mask, binary_mask, obj_mask)

        return {
            "id": id,
            "observations": torch.from_numpy(observations),
            "mask": torch.from_numpy(binary_mask),
            "obj_mask": obj_mask,
        }

    def collate_fn(self, batch):
        assert len(batch) == 1
        return batch[0]


class OfflineEpisodeWrapper(Dataset):
    def __init__(self, rb, split, **kwargs):
        self.rb = rb
        self.split = split
        self.kwargs = kwargs

    def __len__(self):
        return self.rb.num_episodes(self.split)

    def __getitem__(self, idx):
        return self.rb.get_episode(self.split, idx, **self.kwargs)

    def collate_fn(self, batch_list):
        return self.rb.collate_episodes(batch_list)


@njit
def get_binary_mask(mask, binary_mask, obj_mask):
    """
    mask: (T, H, W)
    binary_mask: (T, num_slots, H, W)
    obj_mask: (T, num_objs)
    """
    for t in range(mask.shape[0]):
        # convert object id mask to binary mask
        obj_ids_t = np.unique(mask[t])

        # remove -1 from obj_ids, which means no object on that pixel
        obj_ids_t = np.sort(obj_ids_t[obj_ids_t != -1])

        for i, obj_id in enumerate(obj_ids_t):
            if i >= binary_mask.shape[1]:
                break
            binary_mask[t, i] = mask[t] == obj_id
            obj_mask[t, obj_id] = True

    return binary_mask, obj_mask
