import h5py
from minari import MinariDataset

import torch
from torch.utils.data import Dataset
from torchvision import transforms


procgen_timeout = {
    "default": 1000,
    "bossfight": 4000,
    "bigfish": 6000,
    "jumper": 2000,
    "leaper": 500,
    "maze": 500,
    "plunder": 4000,
}


class ProcgenMinari(Dataset):
    def __init__(
        self,
        cfg,
        episode_idxes,
        resize_to: int,
        interpolation=transforms.InterpolationMode.BILINEAR,
    ):
        self.minari_dataset = MinariDataset(cfg.data_path / "data", episode_idxes)
        self.episode_idxes = episode_idxes

        self.minari_dataset_path = self.minari_dataset.storage._file_path
        self.env_name = cfg.env_name

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize((resize_to, resize_to), interpolation=interpolation),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.minari_dataset)

    def __getitem__(self, idx):
        with h5py.File(self.minari_dataset_path, "r") as file:
            ep_idx = self.episode_idxes[idx]
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)

            id = ep_group.attrs["id"]

            observations = ep_group["observations"][:]                  # (T, H, W, C)
            terminations = ep_group["terminations"][:]                  # (T,)
            truncations = ep_group["truncations"][:]                    # (T,)

        original_frames = torch.from_numpy(observations)
        if original_frames.ndim == 3:
            original_frames = original_frames.unsqueeze(0)

        processed_frames = (original_frames.permute(0, 3, 1, 2) / 255).float()
        processed_frames = self.transform(processed_frames)

        timeout = procgen_timeout.get(self.env_name, procgen_timeout["default"])
        horizon = len(terminations)

        if horizon < timeout:
            terminations[-1] = True
            truncations[-1] = False
        else:
            terminations[-1] = False
            truncations[-1] = True

        episode_data = {
            "id": id,
            "terminations": terminations,
            "truncations": truncations,
        }

        return {
            "episode_data": episode_data,
            "original_frames": original_frames,
            "processed_frames": processed_frames,
        }

    def collate_fn(self, batch):
        assert len(batch) == 1
        return batch[0]
