import os
import h5py
import minari
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils import data


# Returned in place of masks when training without SAM supervision. The dataloader needs a
# tensor to collate, and the training loop skips it based on args.load_sam_masks.
NO_MASK = torch.zeros(0, dtype=torch.bool)


# === Dataset Classes ===
class ProcgenMinariTrain(data.Dataset):
    def __init__(self, args):
        os.environ["MINARI_DATASETS_PATH"] = args.root
        self.dataset_ids = args.train_dataset_ids
        self.resize_to = args.resize_to
        self.num_slots = args.num_slots
        self.load_sam_masks = args.load_sam_masks

        self.minari_datasets = [
            minari.load_dataset(ids)
            for ids in self.dataset_ids
        ]
        self.minari_dataset_paths = [
            dataset.storage._file_path
            for dataset in self.minari_datasets
        ]
        self.sam_mask_paths = [
            dataset.storage._file_path.parent / "sam_masks.h5" if self.load_sam_masks else None
            for dataset in self.minari_datasets
        ]
        # === Get Video Names and Lengths ===
        self.episode_paths = []
        self.video_lengths = []

        # === Train Set ===
        max_num_objs_in_single_frame = -1
        for dataset, dataset_path, sam_mask_path in zip(self.minari_datasets, self.minari_dataset_paths, self.sam_mask_paths):
            for ep_idx in dataset.episode_indices:
                with h5py.File(dataset_path, "r") as file:
                    ep_group = file[f"episode_{ep_idx}"]
                    assert isinstance(ep_group, h5py.Group)
                    self.episode_paths.append((dataset_path, sam_mask_path, ep_idx))
                    self.video_lengths.append(ep_group.attrs.get("total_steps"))

                if not self.load_sam_masks:
                    continue

                with h5py.File(sam_mask_path, "r") as file:
                    ep_group = file[f"episode_{ep_idx}"]
                    assert isinstance(ep_group, h5py.Group)
                    max_num_objs_in_single_frame = max(
                        ep_group.attrs["max_num_objs_in_single_frame"],
                        max_num_objs_in_single_frame,
                    )

        if not self.load_sam_masks:
            print(f"Train dataset without SAM masks, num_slots {self.num_slots} is a free hyperparameter")
        else:
            print("Train dataset max_num_objs_in_single_frame", max_num_objs_in_single_frame)
            if self.num_slots < max_num_objs_in_single_frame:
                print(f"num_slots {self.num_slots} is less than max_num_objs_in_single_frame {max_num_objs_in_single_frame}. Will discard some objects.")

        self.mapping = self.create_idx_frame_mapping()

        # === Transformations ===
        self.resize = T.Resize(self.resize_to)
        self.resize_mask = T.Resize(self.resize_to, T.InterpolationMode.NEAREST_EXACT)
        self.to_tensor = T.ToTensor()

        if args.encoder.startswith("Cosmos"):
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.normalize = T.Normalize(mean=mean, std=std)
            self.denormalize = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

    def __len__(self):
        return len(self.mapping)

    def transform(self, image, mask):

        image = self.to_tensor(image)
        image = self.resize(image)
        image = self.normalize(image)

        mask = torch.tensor(mask, dtype=torch.bool)
        mask = self.resize_mask(mask)

        return image, mask

    def create_idx_frame_mapping(self):
        mapping = []

        for (dataset_path, sam_mask_path, ep_idx), video_length in zip(self.episode_paths, self.video_lengths):
            for video_frame_idx in range(video_length):
                mapping.append((dataset_path, sam_mask_path, ep_idx, video_frame_idx))

        return mapping

    def get_rgb(self, idx):
        dataset_path, sam_mask_path, ep_idx, frame_idx = self.mapping[idx]
        with h5py.File(dataset_path, "r") as file:
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)

            if ep_group.attrs["total_steps"] == 1:
                frame = ep_group["observations"][:]                     # (H, W, 3)
            else:
                frame = ep_group["observations"][frame_idx]             # (H, W, 3)

        if not self.load_sam_masks:
            frame = self.normalize(self.resize(self.to_tensor(frame)))
            return frame, NO_MASK

        with h5py.File(sam_mask_path, "r") as file:
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)
            mask = ep_group["sam_masks"][frame_idx]                     # (H, W)

        # convert object id mask to binary mask
        obj_ids, counts = np.unique(mask, return_counts=True)

        # remove -1 from obj_ids, which means no object on that pixel
        counts = counts[obj_ids != -1]
        obj_ids = obj_ids[obj_ids != -1]

        if len(obj_ids) > self.num_slots:
            # discard the least frequent objects
            sort_idx = counts.argsort()[-self.num_slots:]
            obj_ids = obj_ids[sort_idx]

        binary_mask = np.zeros((self.num_slots, *mask.shape), dtype=bool)

        # shuffle the object ids to randomize the order
        np.random.shuffle(obj_ids)
        for i, obj_id in enumerate(obj_ids):
            binary_mask[i] = mask == obj_id

        frame, binary_mask = self.transform(frame, binary_mask)

        return frame, binary_mask

    def __getitem__(self, idx):
        """
        :return:
            input_features: RGB frames t
                            in shape (3, H, W)

            frame_masks: SAM mask for frame, padded to num_slots
                            in shape (self.num_slots, H, W)
        """

        return self.get_rgb(idx)    # (3, H, W), (self.num_slots, H, W)


class ProcgenMinariVal(data.Dataset):
    def __init__(self, args):
        os.environ["MINARI_DATASETS_PATH"] = args.root
        self.dataset_ids = args.valid_dataset_ids
        self.resize_to = args.resize_to
        self.num_slots = args.num_slots
        self.load_sam_masks = args.load_sam_masks

        self.minari_datasets = [
            minari.load_dataset(ids)
            for ids in self.dataset_ids
        ]
        self.minari_dataset_paths = [
            dataset.storage._file_path
            for dataset in self.minari_datasets
        ]
        self.sam_mask_paths = [
            dataset.storage._file_path.parent / "sam_masks.h5" if self.load_sam_masks else None
            for dataset in self.minari_datasets
        ]
        # === Get Video Names and Lengths ===
        self.episode_paths = []

        # === Train Set ===
        max_num_objs_in_single_frame = -1
        for dataset, dataset_path, sam_mask_path in zip(self.minari_datasets, self.minari_dataset_paths, self.sam_mask_paths):
            for ep_idx in dataset.episode_indices:
                with h5py.File(dataset_path, "r") as file:
                    ep_group = file[f"episode_{ep_idx}"]
                    assert isinstance(ep_group, h5py.Group)
                    self.episode_paths.append((dataset_path, sam_mask_path, ep_idx))

                if not self.load_sam_masks:
                    continue

                with h5py.File(sam_mask_path, "r") as file:
                    ep_group = file[f"episode_{ep_idx}"]
                    assert isinstance(ep_group, h5py.Group)
                    max_num_objs_in_single_frame = max(
                        ep_group.attrs["max_num_objs_in_single_frame"],
                        max_num_objs_in_single_frame,
                    )

        if not self.load_sam_masks:
            print("Val dataset without SAM masks, segmentation metrics are disabled")
        else:
            print("Val dataset max_num_objs_in_single_frame", max_num_objs_in_single_frame)
            if self.num_slots < max_num_objs_in_single_frame:
                print(f"num_slots {self.num_slots} is less than max_num_objs_in_single_frame {max_num_objs_in_single_frame}. Will discard some objects.")

        # === Transformations ===
        self.resize = T.Resize(self.resize_to)
        self.resize_mask = T.Resize(self.resize_to, T.InterpolationMode.NEAREST_EXACT)
        self.to_tensor = T.ToTensor()

        if args.encoder.startswith("Cosmos"):
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.normalize = T.Normalize(mean=mean, std=std)
            self.denormalize = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

    def __len__(self):
        return len(self.episode_paths)

    def transform(self, image, mask):
        image = torch.from_numpy(image)
        image = image.permute(0, 3, 1, 2) / 255.0
        image = self.resize(image)
        image = self.normalize(image)

        mask = torch.tensor(mask, dtype=torch.bool)
        mask = self.resize_mask(mask)

        return image, mask

    def get_rgb(self, idx):
        dataset_path, sam_mask_path, ep_idx = self.episode_paths[idx]
        with h5py.File(dataset_path, "r") as file:
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)

            frame = ep_group["observations"][:150]                  # (T, H, W, 3)
            if ep_group.attrs["total_steps"] == 1:
                frame = frame[None]                                 # (1, H, W, 3)

        if not self.load_sam_masks:
            frame = torch.from_numpy(frame).permute(0, 3, 1, 2) / 255.0
            return self.normalize(self.resize(frame)), NO_MASK

        with h5py.File(sam_mask_path, "r") as file:
            ep_group = file[f"episode_{ep_idx}"]
            assert isinstance(ep_group, h5py.Group)
            mask = ep_group["sam_masks"][:150]                      # (T, H, W)

        T, H, W = mask.shape
        binary_mask = np.zeros((T, self.num_slots, H, W), dtype=bool)
        for t in range(T):
            # convert object id mask to binary mask
            obj_ids, counts = np.unique(mask[t], return_counts=True)

            # remove -1 from obj_ids, which means no object on that pixel
            counts = counts[obj_ids != -1]
            obj_ids = obj_ids[obj_ids != -1]

            if len(obj_ids) > self.num_slots:
                # discard the least frequent objects
                sort_idx = counts.argsort()[-self.num_slots:]
                obj_ids = obj_ids[sort_idx]

            for i, obj_id in enumerate(sorted(obj_ids)):
                binary_mask[t, i] = mask[t] == obj_id

        frame, mask = self.transform(frame, binary_mask)

        return frame, mask

    def __getitem__(self, idx):
        """
        :return:
            input_frames: (T, 3, H, W)
            frame_masks: (T, self.num_slots, H, W)
        """

        return self.get_rgb(idx)
