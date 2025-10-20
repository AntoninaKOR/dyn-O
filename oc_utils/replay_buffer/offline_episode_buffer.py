from typing import List, Dict, Literal

import os
import h5py
import json
import minari
import dataclasses
import numpy as np

from tqdm import tqdm
from collections import defaultdict
from contextlib import ExitStack
from einops import rearrange
from filelock import FileLock
from pathlib import Path
from gym.spaces import Discrete, MultiDiscrete, MultiBinary

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T

from oc_utils.replay_buffer.offline_episode_buffer_wrapper import (
    EpisodeRgbDataset, OfflineEpisodeWrapper,
)
from oc_utils.replay_buffer.utils import obj_to_slot_format, get_temporal_idxes
from oc_utils.replay_buffer.data_format import Episode
from oc_utils.utils import are_dicts_equal


class OfflineEpisodeBuffer:
    """
    Sample are encodings
        Otherwise, precompute encodings during initialization and load them during sampling
    Assume the following directory structure:
        config.data.data_path
        ├── dataset_id1
        │   ├── data
        │   │   ├── main_data.hdf5                  # minari data
        │   │   ├── metadata.json                   # minari metadata
        │   │   ├── config.data.encoding_fname      # precomputed encodings
    """

    def __init__(
        self,
        config,
        encoder,
        **kwargs,
    ):
        self.config = config
        self.encoder = encoder.module if isinstance(encoder, DDP) else encoder

        self.dtype = config.dtype
        self.np_dtype = config.np_dtype

        # === Load Minari Datasets ===
        self.data_path = Path(config.data.data_path)
        assert self.data_path.exists(), f"Data path {self.data_path} does not exist"
        os.environ["MINARI_DATASETS_PATH"] = config.data.data_path

        self.minari_datasets = [minari.load_dataset(ids) for ids in config.data.dataset_ids]
        self.minari_dataset_paths = [dataset.storage._file_path for dataset in self.minari_datasets]
        self.sam_mask_paths = [path.parent / "sam_masks.h5" for path in self.minari_dataset_paths]
        self.encoding_paths = [path.parent / config.data.encoding_fname for path in self.minari_dataset_paths]
        self.encoding_for_kmeans_paths = [path.parent / config.data.encoding_for_kmeans_fname for path in self.minari_dataset_paths]
        self.reward_termination_stats_paths = [path.parent / config.data.reward_termination_stats_fname for path in self.minari_dataset_paths]

        self.metadata = []
        for dataset in self.minari_datasets:
            env_id = dataset.id.split("-")[1]
            metadata = dataset.storage.metadata
            metadata["env_id"] = env_id
            self.metadata.append(metadata)

        # check if all environments have the same observation_space and action_space
        self.observation_space, self.action_space = self.verify_env_metadata(self.minari_datasets)
        self.H, self.W, C = self.observation_space.shape
        assert C == 3, f"Only support RGB images, but got {C} channels"

        if isinstance(self.action_space, (Discrete, MultiDiscrete, MultiBinary)):
            self.action_np_dtype = np.int64
            self.action_torch_dtype = torch.int64
        else:
            self.action_np_dtype = self.np_dtype
            self.action_torch_dtype = self.dtype

        # === Load Obs and Encodings ===
        # data index mapping
        self.episode_paths = []
        self.train_episode_paths = []
        self.valid_episode_paths = []
        self.reward_termination_stats = []

        self.resize = T.Resize(self.config.encoder.resize_to)

        max_num_objs_in_single_frame = -1
        max_num_slots_in_single_frame = -1

        for (
            dataset,
            metadata,
            dataset_path,
            sam_mask_path,
            encoding_path,
            encoding_for_kmeans_path,
            reward_termination_stats_path,
        ) in zip(
            self.minari_datasets,
            self.metadata,
            self.minari_dataset_paths,
            self.sam_mask_paths,
            self.encoding_paths,
            self.encoding_for_kmeans_paths,
            self.reward_termination_stats_paths,
        ):
            num_episodes = len(dataset.episode_indices)

            # --- Load Precomputed Encodings; If Fail, Precompute and Save Them ---
            load_success = self.load_encodings(dataset, dataset_path, encoding_path)
            if not load_success:
                print(
                    f"Precomputed encodings for {dataset.id} does not exist. Precomputing {num_episodes} episodes."
                )
                if self.config.distributed:
                    # wait all processes to finish the check
                    dist.barrier()
                self.precompute_encodings(dataset, dataset_path, sam_mask_path, encoding_path)

            load_success = self.load_encodings_for_kmeans(encoding_for_kmeans_path)
            if not load_success:
                print(
                    f"Precomputed encodings for kmeans for {dataset.id} does not exist. "
                    f"Precomputing {self.config.data.num_episodes_for_kmeans} episodes."
                )
                if self.config.distributed:
                    # wait all processes to finish the check
                    dist.barrier()
                self.precompute_encodings_for_kmeans(encoding_path, encoding_for_kmeans_path)

            # load or precompute reward termination stats, used by dynamics model to compute label weights
            reward_termination_stats = self.load_reward_termination_stats(reward_termination_stats_path)
            if reward_termination_stats is None:
                print(
                    f"Precomputed reward termination stats for {dataset.id} does not exist. "
                    f"Precomputing {num_episodes} episodes."
                )
                if self.config.distributed:
                    # wait all processes to finish the check
                    dist.barrier()
                reward_termination_stats = self.precompute_reward_termination_stats(dataset, dataset_path, reward_termination_stats_path)

            self.reward_termination_stats.append(reward_termination_stats)

            # --- Load Minari Mapping ---
            num_valid_episodes = max(1, int(num_episodes * (1 - config.data.train_ratio)))
            is_valid = np.zeros(num_episodes, dtype=bool)
            is_valid[:num_valid_episodes] = True
            is_valid = is_valid[np.random.permutation(num_episodes)]

            for i, epi_idx in enumerate(dataset.episode_indices):
                if not config.debug:
                    with ExitStack() as stack:
                        dataset_f = stack.enter_context(h5py.File(dataset_path, "r"))
                        sam_mask_f = stack.enter_context(h5py.File(sam_mask_path, 'r'))
                        encoding_f = stack.enter_context(h5py.File(encoding_path, 'r'))
                        dataset_ep_group = dataset_f[f"episode_{epi_idx}"]
                        assert isinstance(dataset_ep_group, h5py.Group)
                        sam_mask_ep_group = sam_mask_f[f"episode_{epi_idx}"]
                        assert isinstance(sam_mask_ep_group, h5py.Group)
                        encoding_ep_group = encoding_f[f"episode_{epi_idx}"]
                        assert isinstance(encoding_ep_group, h5py.Group)

                        assert dataset_ep_group.attrs["total_steps"] == encoding_ep_group.attrs["total_steps"]

                        if config.encoder.use_sam_mask:
                            max_num_objs_in_single_frame = sam_mask_ep_group.attrs["max_num_objs_in_single_frame"]
                            max_num_slots_in_single_frame = encoding_ep_group.attrs["num_slots"]

                            if config.encoder.num_slots < max_num_objs_in_single_frame:
                                continue
                            if config.dynamics.num_slots < max_num_slots_in_single_frame:
                                continue

                self.episode_paths.append((dataset_path, sam_mask_path, encoding_path, epi_idx))
                if is_valid[i]:
                    self.valid_episode_paths.append((dataset_path, sam_mask_path, encoding_path, epi_idx))
                else:
                    self.train_episode_paths.append((dataset_path, sam_mask_path, encoding_path, epi_idx))

        print(f"Number of episodes: {len(self.episode_paths)}")
        print(f"Number of train episodes: {len(self.train_episode_paths)}")
        print(f"Number of valid episodes: {len(self.valid_episode_paths)}")
        print("Dataset max_num_objs_in_single_frame", max_num_objs_in_single_frame)
        print("Dataset max_num_slots_in_single_frame", max_num_slots_in_single_frame)

        self.reward_termination_weights = self.compute_reward_termination_weights()
        print("Reward weights:")
        print(json.dumps(self.reward_termination_weights["rewards"], indent=4))
        print("Termination weights:")
        print(json.dumps(self.reward_termination_weights["terminations"], indent=4))

    def compute_reward_termination_weights(self):
        aggregated_reward_termination_stats = {
            "rewards": defaultdict(int),
            "terminations": defaultdict(int),
        }
        for reward_termination_stats in self.reward_termination_stats:
            for var_name in ["rewards", "terminations"]:
                var_stats = reward_termination_stats[var_name]                      # Dict[var_value, count]
                agg_var_stats = aggregated_reward_termination_stats[var_name]       # Dict[var_value, count_across_datasets]
                for k, v in var_stats.items():
                    # k: str -> float
                    agg_var_stats[float(k)] += v

        total_reward_count = sum(aggregated_reward_termination_stats["rewards"].values())
        total_termination_count = sum(aggregated_reward_termination_stats["terminations"].values())
        assert total_reward_count > 0, "Total reward count is 0"
        assert total_termination_count > 0, "Total termination count is 0"
        assert total_reward_count == total_termination_count, \
            f"Total reward count and termination count are not equal: {total_reward_count} vs {total_termination_count}"

        reward_termination_weights = {
            "rewards": {},
            "terminations": {},
        }
        for var_name in ["rewards", "terminations"]:
            agg_var_stats = aggregated_reward_termination_stats[var_name]
            num_bins = len(agg_var_stats)
            for k, v in agg_var_stats.items():
                reward_termination_weights[var_name][k] = total_reward_count / (num_bins * v)

        return reward_termination_weights

    @staticmethod
    def verify_env_metadata(minari_datasets):
        observation_space, action_space = None, None
        for dataset in minari_datasets:
            obs_space, act_space = dataset.observation_space, dataset.action_space
            if observation_space is None:
                observation_space = obs_space
            else:
                assert (
                    observation_space == obs_space
                ), f"Obs space mismatch for dataset {dataset}:\n{observation_space}\nvs\n{obs_space}"

            if action_space is None:
                action_space = act_space
            else:
                assert (
                    action_space == act_space
                ), f"Act space mismatch for dataset {dataset}:\n{action_space}\nvs\n{act_space}"

        return observation_space, action_space

    def load_encodings(self, dataset, dataset_path, encoding_path):
        if not encoding_path.exists():
            print(
                f"Precomputed encodings {encoding_path} does not exist. Re-computing."
            )
            return False

        with ExitStack() as stack:
            dataset_f = stack.enter_context(h5py.File(dataset_path, "r"))
            encoding_f = stack.enter_context(h5py.File(encoding_path, 'r'))

            # check # of episodes
            num_encoding_episodes = encoding_f.attrs.get("num_episodes", None)
            if len(dataset.episode_indices) != num_encoding_episodes:
                print(
                    f"Precomputed encodings do not match the number of episodes for {dataset.id}. Re-computing."
                )
                return False

            # load and check if encoder config matches
            encoder_config_group = encoding_f.get("encoder_config")
            assert isinstance(encoder_config_group, h5py.Group)

            encoder_config = {}
            for key, value in encoder_config_group.attrs.items():   # Load bool, int, float
                encoder_config[key] = value
            for key, value in encoder_config_group.items():       # load str, arrays, lists
                value = value[()]
                if isinstance(value, bytes):
                    # Convert bytes to string using decode
                    encoder_config[key] = value.decode('utf-8')
                else:
                    encoder_config[key] = value

            # a rough check on whether the encoder is correct
            if not are_dicts_equal(
                dataclasses.asdict(self.config.encoder),
                encoder_config,
                exclude_keys=["checkpoint_path"],
            ):
                print("Precomputed encoding config do not match the current encoder config. Re-computing.")
                return False

            # check each episode length
            if not self.config.debug:
                for epi_idx in dataset.episode_indices:
                    ep_key = f"episode_{epi_idx}"
                    ep_group = dataset_f[ep_key]
                    assert isinstance(ep_group, h5py.Group)
                    episode_length = ep_group.attrs.get("total_steps")

                    if ep_key not in encoding_f:
                        print(f"Precomputed encodings for {dataset.id} {ep_key} does not exist. Re-computing.")
                        return False

                    ep_group = encoding_f[ep_key]
                    assert isinstance(ep_group, h5py.Group)
                    encoding_episode_length = ep_group.attrs.get("total_steps")
                    if episode_length != encoding_episode_length:
                        print(
                            f"Precomputed encodings do not match the length of episodes for {dataset.id} {ep_key}. "
                            f"Re-computing."
                        )
                        return False

        return True

    def precompute_encodings(self, dataset, dataset_path, sam_mask_path, encoding_path):
        if encoding_path.exists():
            raise ValueError(f"Precomputed encodings {encoding_path} already exists.")

        # prepare dataset and dataloader
        episode_rgb_dataset = EpisodeRgbDataset(dataset, dataset_path, sam_mask_path, self.config)

        if self.config.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=False,
            )
            sampler.set_epoch(0)
        else:
            sampler = None

        dataloader = DataLoader(
            episode_rgb_dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=episode_rgb_dataset.collate_fn,
        )

        # create encoding file
        if self.config.rank == 0:
            encoding_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(encoding_path, 'a') as f:
                f.attrs["num_episodes"] = len(dataset.episode_indices)

                # add encoder config to the h5 file
                encoder_config_group = f.create_group("encoder_config")
                for key, value in dataclasses.asdict(self.config.encoder).items():
                    if isinstance(value, (int, float, bool)):
                        encoder_config_group.attrs[key] = value  # Store scalars as attributes
                    elif isinstance(value, str):
                        dtype = h5py.string_dtype(encoding="utf-8")
                        encoder_config_group.create_dataset(key, data=value, dtype=dtype)  # Store strings as datasets
                    else:
                        encoder_config_group.create_dataset(key, data=value)  # Store arrays as datasets

        # let all processes wait until the file is created
        if self.config.distributed:
            dist.barrier()

        # create a lock file to avoid concurrent writes
        lock_fname = encoding_path.with_suffix(".lock")
        lock = FileLock(lock_fname)

        # compute slots and write to the encoding file
        num_ep = 0
        for ep_data in tqdm(dataloader, desc=f"Precomputing encodings for {dataset.id}", disable=self.config.rank != 0):

            id = ep_data["id"]
            is_obj_visible = ep_data["obj_mask"]                                # (T, num_objs)
            observations = ep_data["observations"]                              # (T, H, W, C)
            mask = ep_data["mask"]                                              # (T, num_encoder_slots, H, W)

            with torch.inference_mode():
                self.encoder.eval()
                enc_feat, slots = self.encoder.forward_episode(
                    observations,
                    mask if self.config.encoder.use_sam_mask else None,
                    ["enc_feat", "slots"],
                )
                enc_feat = enc_feat.cpu().numpy()                               # (T, num_tokens, token_dim)
                slots = slots.cpu().numpy()                                     # (T, num_encoder_slots, slot_dim)

            if self.config.encoder.use_sam_mask:
                # convert obj encodings, obj visible maks, and obj existence mask to slot format
                # (T, num_slots, slot_dim), (T, num_slots), (T, num_slots)
                slots, slots_visible, slots_exist = obj_to_slot_format(slots, is_obj_visible)
            else:
                T, num_slots, _ = slots.shape
                slots_visible = np.ones((T, num_slots), dtype=bool)
                slots_exist = np.ones((T, num_slots), dtype=bool)

            with lock:
                with h5py.File(encoding_path, 'a') as f:
                    if f"episode_{id}" in f:
                        continue

                    ep_group = f.create_group(f"episode_{id}")
                    ep_group.attrs["total_steps"] = len(observations)
                    ep_group.attrs["num_slots"] = slots.shape[1]
                    for name, var in zip(
                        ["enc_feat", "slots", "slots_visible", "slots_exist"],
                        [enc_feat, slots, slots_visible, slots_exist],
                    ):
                        ep_group.create_dataset(
                            name,
                            data=var,
                            dtype=None,
                            chunks=(1, *var.shape[1:]) if name in ["slots", "enc_feat"] else True,
                            maxshape=(None, *var.shape[1:]),
                            compression="gzip",
                            shuffle=True,
                        )

            num_ep += 1
            if self.config.distributed and num_ep % 5 == 0:
                # sync up precomputation
                dist.barrier()

        if self.config.distributed:
            # let all processes wait until the precomputation is done
            dist.barrier()

        if self.config.rank == 0:
            os.remove(lock_fname)

    def load_encodings_for_kmeans(self, encoding_path):
        if not encoding_path.exists():
            print(
                f"Precomputed encodings {encoding_path} does not exist. Re-computing."
            )
            return False

        with h5py.File(encoding_path, 'r') as f:
            # load and check if encoder config matches
            encoder_config_group = f.get("encoder_config")
            assert isinstance(encoder_config_group, h5py.Group)

            encoder_config = {}
            for key, value in encoder_config_group.attrs.items():       # load bool, int, float
                encoder_config[key] = value

            for key, value in encoder_config_group.items():             # load str, arrays, lists
                value = value[()]
                if isinstance(value, bytes):
                    # Convert bytes to string using decode
                    encoder_config[key] = value.decode('utf-8')
                else:
                    encoder_config[key] = value

            # a rough check on whether the encoder is correct
            if not are_dicts_equal(
                dataclasses.asdict(self.config.encoder),
                encoder_config,
                exclude_keys=["checkpoint_path"],
            ):
                print("Precomputed encoding config do not match the current encoder config. Re-computing.")
                return False

            for k in ["num_episodes_for_kmeans", "num_steps_per_episode_for_kmeans"]:
                if k not in f.attrs:
                    print(f"Precomputed encodings for kmeans do not have {k}. Re-computing.")
                    return False
                if getattr(self.config.data, k) != f.attrs[k]:
                    print(f"Precomputed encodings for kmeans have different {k}: "
                          f"{f.attrs[k]} vs {getattr(self.config.data, k)}. Re-computing.")
                    return False

        return True

    def precompute_encodings_for_kmeans(self, encoding_path, encoding_for_kmeans_path):
        """
        prepare encodings for kmeans clustering, which will be used by dynamics model to sample new static features
        """
        if encoding_for_kmeans_path.exists():
            raise ValueError(f"Precomputed encodings {encoding_for_kmeans_path} already exists.")

        num_episodes = self.config.data.num_episodes_for_kmeans
        n_steps = self.config.data.num_steps_per_episode_for_kmeans

        if self.config.rank == 0:
            encoding_for_kmeans_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(encoding_for_kmeans_path, 'a') as f:
                f.attrs["num_episodes_for_kmeans"] = num_episodes
                f.attrs["num_steps_per_episode_for_kmeans"] = n_steps

                # add encoder config to the h5 file
                encoder_config_group = f.create_group("encoder_config")
                for key, value in dataclasses.asdict(self.config.encoder).items():
                    if isinstance(value, (int, float, bool)):
                        encoder_config_group.attrs[key] = value  # Store scalars as attributes
                    elif isinstance(value, str):
                        dtype = h5py.string_dtype(encoding="utf-8")
                        encoder_config_group.create_dataset(key, data=value, dtype=dtype)  # Store strings as datasets
                    else:
                        encoder_config_group.create_dataset(key, data=value)  # Store arrays as datasets

        if self.config.distributed:
            # let all processes wait until the file is created
            dist.barrier()

        data = {
            "slots": [],
            "episode_idx": [],
            "timestamp_idx": [],
            "slot_idx": [],
        }
        with h5py.File(encoding_path, 'r', libver="latest", swmr=True) as f:
            num_total_episodes = f.attrs["num_episodes"]

        for i in tqdm(range(num_episodes), desc=f"Precomputing encodings for K-means", disable=self.config.rank != 0):
            epi_idx = i % num_total_episodes

            with h5py.File(encoding_path, 'r', libver="latest", swmr=True) as f:
                encoding_ep_group = f[f"episode_{epi_idx}"]
                total_steps = encoding_ep_group.attrs["total_steps"]

                if n_steps == -1 or n_steps >= total_steps:
                    idxes = slice(0, total_steps)
                else:
                    start_idx = np.random.randint(0, total_steps - n_steps)
                    idxes = slice(start_idx, min(start_idx + n_steps, total_steps))

                slots = np.ascontiguousarray(encoding_ep_group["slots"][idxes])                     # (T, num_slots, slot_dim)
                slots_visible = np.ascontiguousarray(encoding_ep_group["slots_visible"][idxes])     # (T, num_slots)

            T, num_slots = slots_visible.shape

            epi_data = {
                "slots": slots,
                "episode_idx": np.full((T, num_slots), epi_idx, dtype=int),
                "timestamp_idx": np.arange(T).reshape(-1, 1).repeat(num_slots, axis=1),
                "slot_idx": np.arange(num_slots).reshape(1, -1).repeat(T, axis=0),
            }
            for k, v in epi_data.items():
                v = v[slots_visible]
                data[k].append(v)
            if self.config.distributed and i % 5 == 0:
                # sync up precomputation
                dist.barrier()

        data = {k: np.concatenate(v, axis=0) for k, v in data.items()}
        if self.config.rank == 0:
            with h5py.File(encoding_for_kmeans_path, 'a') as f:
                for k, v in data.items():
                    f.create_dataset(
                        k,
                        data=v,
                        dtype=None,
                        chunks=(1, *v.shape[1:]),
                        maxshape=(None, *v.shape[1:]),
                        compression="gzip",
                        shuffle=True,
                    )
        if self.config.distributed:
            # let all processes wait until the precomputation is done
            dist.barrier()

    def load_reward_termination_stats(self, reward_termination_stats_path):
        if not reward_termination_stats_path.exists():
            print(
                f"Reward termination stats {reward_termination_stats_path} does not exist. Re-computing."
            )
            return None

        with open(reward_termination_stats_path, 'r') as f:
            reward_termination_stats = json.load(f)

        return reward_termination_stats

    def precompute_reward_termination_stats(self, dataset, dataset_path, reward_termination_stats_path):
        """
        prepare reward termination stats, which will be used by dynamics model to compute label weights
        """
        if reward_termination_stats_path.exists():
            raise ValueError(f"Precomputed reward termination stats {reward_termination_stats_path} already exists.")

        stats = {
            "rewards": defaultdict(int),
            "terminations": defaultdict(int),
        }
        with h5py.File(dataset_path, "r", libver="latest", swmr=True) as file:
            for ep_idx in dataset.episode_indices:
                ep_group = file[f"episode_{ep_idx}"]
                assert isinstance(ep_group, h5py.Group)

                rewards = ep_group["rewards"][()]                               # (T, )
                terminations = ep_group["terminations"][()]                     # (T, )

                # json expects double
                rewards = rewards.astype(np.float64)
                terminations = terminations.astype(np.float64)

                for r, t in zip(rewards, terminations):
                    stats["rewards"][r] += 1
                    stats["terminations"][t] += 1

        if self.config.rank == 0:
            with open(reward_termination_stats_path, 'w') as f:
                json.dump(stats, f, indent=4)

        if self.config.distributed:
            dist.barrier()

        return stats

    def num_episodes(self, split: str):
        if split == "train":
            return int(len(self.train_episode_paths) * self.config.data.train_prop)
        elif split == "valid":
            return len(self.valid_episode_paths)
        elif split == "all":
            return len(self.episode_paths)
        else:
            raise ValueError(f"Invalid split {split}")

    def get_episode(
        self,
        split: Literal["train", "valid", "all"],
        epi_idx: int,
        seq_len: int = -1,
        load_obs: bool = False,
        load_sam_mask: bool = False,
        load_enc_feat: bool = False,
        load_slot: bool = False,
        patch_as_slot: bool = False,
        termination_prob: float = 0.2,
        deterministic: bool = False,
    ):
        """

        :param split: dataset split
        :param epi_idx:
        :param seq_len: -1 uses all steps in the episode, otherwise sample an seq_len chunk
        :param load_obs: load rgb observations
        :param load_sam_mask: load sam mask
        :param load_enc_feat: load patch-level encoder-extracted features
        :param load_slot: load slots
        :param patch_as_slot: use patch-level encoder-extracted features instead object-level slots (for ablation studies)
        :param termination_prob:
            when seq_len != -1, the probability for a sampled chunk to have termination = True
            notice that such a chunk could have length < seq_len
            use > 0 value to avoid the sampled data to have too few termination = True labels during training
        :param deterministic: generate fixed-length chunks deterministically
        :return:
        """
        if split == "train":
            episode_paths = self.train_episode_paths
        elif split == "valid":
            episode_paths = self.valid_episode_paths
        elif split == "all":
            episode_paths = self.episode_paths
        else:
            raise ValueError(f"Invalid split {split}")
        dataset_path, sam_mask_path, encoding_path, epi_idx = episode_paths[epi_idx]
        with ExitStack() as stack:
            dataset_f = stack.enter_context(h5py.File(dataset_path, "r", libver="latest", swmr=True))
            dataset_ep_group = dataset_f[f"episode_{epi_idx}"]
            total_steps = dataset_ep_group.attrs["total_steps"]

            # sample a sub-trajectory from the episode
            if termination_prob < 0:
                start_idx = np.random.randint(total_steps)
                idxes = slice(start_idx, min(start_idx + seq_len, total_steps))
            else:
                if seq_len == -1 or seq_len >= total_steps:
                    idxes = slice(0, total_steps)
                else:
                    if np.random.rand() < termination_prob:
                        start_idx = np.random.randint(total_steps - seq_len, total_steps)
                    else:
                        if deterministic:
                            rng = np.random.RandomState(epi_idx)  # Fixed seed
                            start_idx = rng.randint(0, total_steps - seq_len)
                            start_idx = int((total_steps - seq_len) * 0.5)
                        else:
                            start_idx = np.random.randint(0, total_steps - seq_len)
                    idxes = slice(start_idx, min(start_idx + seq_len, total_steps))

            # load rgb observation
            observations = None
            if load_obs:
                if total_steps == 1:
                    observations = np.ascontiguousarray(dataset_ep_group["observations"][()])       # (H, W, C)
                    observations = observations[None, ...]                                          # (1, H, W, C)
                else:
                    observations = np.ascontiguousarray(dataset_ep_group["observations"][idxes])    # (T, H, W, C)

            sam_masks = None
            if load_sam_mask:
                sam_mask_f = stack.enter_context(h5py.File(sam_mask_path, 'r', libver="latest", swmr=True))
                sam_mask_ep_group = sam_mask_f[f"episode_{epi_idx}"]
                if total_steps == 1:
                    sam_masks = np.ascontiguousarray(sam_mask_ep_group["sam_masks"][()])            # (H, W)
                    sam_masks = sam_masks[None, ...]                                                # (1, H, W)
                else:
                    sam_masks = np.ascontiguousarray(sam_mask_ep_group["sam_masks"][idxes])         # (T, H, W)

            # load encoding: slot / encoder feature
            if load_enc_feat or load_slot:
                encoding_f = stack.enter_context(h5py.File(encoding_path, 'r', libver="latest", swmr=True))
                encoding_ep_group = encoding_f[f"episode_{epi_idx}"]
                assert (
                    dataset_ep_group.attrs["total_steps"] == encoding_ep_group.attrs["total_steps"] or
                    dataset_ep_group.attrs["total_steps"] + 1 == encoding_ep_group.attrs["total_steps"]
                ), (
                    f"total_steps in dataset and encoding do not match for {epi_idx}:"
                    f"{dataset_ep_group.attrs['total_steps']} vs {encoding_ep_group.attrs['total_steps']}"
                )

                if load_enc_feat:
                    enc_feat = np.ascontiguousarray(encoding_ep_group["enc_feat"][idxes])           # (T, num_tokens, token_dim)

                if load_slot:
                    slots = np.ascontiguousarray(encoding_ep_group["slots"][idxes])                 # (T, num_slots, slot_dim)

                    # pad on the 2nd axis to make all episodes have the same number of slots
                    slots = np.pad(
                        slots,
                        ((0, 0), (0, self.config.dynamics.num_slots - slots.shape[1]), (0, 0)),
                        mode='constant',
                        constant_values=0,
                    )

                # load slots visibility and existence
                if patch_as_slot:
                    assert load_enc_feat

                    T, num_tokens, _ = enc_feat.shape
                    slots_visible = slots_exist = np.ones((T, num_tokens), dtype=bool)              # (T, num_tokens)

                    # will not be used, just to keep the same format as object-level slots
                    temporal_random_idxes = slot_obj_id = np.zeros((T, num_tokens), dtype=int)
                else:
                    slots_visible = np.ascontiguousarray(encoding_ep_group["slots_visible"][idxes]) # (T, num_slots)
                    slots_exist = np.ascontiguousarray(encoding_ep_group["slots_exist"][idxes])     # (T, num_slots)

                    temporal_random_idxes = np.zeros_like(slots_exist, dtype=int)
                    slot_obj_id = -np.ones_like(slots_exist, dtype=int)
                    temporal_random_idxes, slot_obj_id = get_temporal_idxes(
                        slots_exist, slots_visible,
                        temporal_random_idxes, slot_obj_id,
                    )

                    # pad on the 2nd axis to make all episodes have the same number of slots
                    slots_visible, slots_exist, temporal_random_idxes, slot_obj_id = [
                        np.pad(
                            arr,
                            ((0, 0), (0, self.config.dynamics.num_slots - arr.shape[1])),
                            mode='constant',
                            constant_values=0,
                        )
                        for arr in (slots_visible, slots_exist, temporal_random_idxes, slot_obj_id)
                    ]

            # load actions, rewards, terminations, truncations
            if total_steps == 1:
                actions = np.array(dataset_ep_group["actions"][()])                                 # (1, )
                rewards = np.array(dataset_ep_group["rewards"][()])                                 # (1, )
                terminations = np.array(dataset_ep_group["terminations"][()])                       # (1, )
                truncations = np.array(dataset_ep_group["truncations"][()])                         # (1, )
            else:
                actions = np.ascontiguousarray(dataset_ep_group["actions"][idxes])                  # (T, )
                rewards = np.ascontiguousarray(dataset_ep_group["rewards"][idxes])                  # (T, )
                terminations = np.ascontiguousarray(dataset_ep_group["terminations"][idxes])        # (T, )
                truncations = np.ascontiguousarray(dataset_ep_group["truncations"][idxes])          # (T, )

            # compute weights for rewards and terminations
            rewards_weights = np.zeros_like(rewards)
            terminations_weights = np.zeros_like(terminations)
            for i, (r, t) in enumerate(zip(rewards, terminations)):
                rewards_weights[i] = self.reward_termination_weights["rewards"][r]
                terminations_weights[i] = self.reward_termination_weights["terminations"][t]

        episode = {
            "actions": torch.tensor(actions, dtype=self.action_torch_dtype),
            "rewards": torch.tensor(rewards, dtype=self.dtype),
            "terminations": torch.tensor(terminations, dtype=torch.bool),
            "truncations": torch.tensor(truncations, dtype=torch.bool),
            "rewards_weights": torch.tensor(rewards_weights, dtype=self.dtype),
            "terminations_weights": torch.tensor(terminations_weights, dtype=self.dtype),
        }

        if load_obs:
            observations = rearrange(observations, "t h w c -> t c h w")                            # (T, C, H, W)
            observations = observations / 255.0
            if not isinstance(observations, torch.Tensor):
                observations = torch.tensor(observations, dtype=self.dtype)
            observations = self.resize(observations)
            episode["observations"] = observations

        if load_sam_mask:
            episode["sam_masks"] = torch.tensor(sam_masks, dtype=torch.int64)

        if load_enc_feat:
            episode["enc_feat"] = torch.tensor(enc_feat, dtype=self.dtype)

        if load_slot:
            episode["slots"] = torch.tensor(slots, dtype=self.dtype)

        if load_enc_feat or load_slot:
            episode.update({
                "slots_visible": torch.tensor(slots_visible, dtype=torch.bool),
                "slots_exist": torch.tensor(slots_exist, dtype=torch.bool),
                # for static-dynamic disentanglement training
                "temporal_random_idxes": torch.tensor(temporal_random_idxes, dtype=torch.long),
                "slot_obj_id": torch.tensor(slot_obj_id, dtype=torch.long),
            })

        return episode

    @staticmethod
    def collate_episodes(batch_list: List[Dict[str, torch.Tensor]]):
        # batch_list: list of dict

        num_episodes = len(batch_list)
        num_transitions = np.array([len(item["rewards"]) for item in batch_list])

        max_len = num_transitions.max()
        batch = {
            k: torch.zeros(num_episodes, max_len, *v.shape[1:], dtype=v.dtype)  # (num_episodes, max_len, ...)
            for k, v in batch_list[0].items()
        }
        padding_mask = torch.zeros(num_episodes, max_len, dtype=torch.bool)
        for i, (item, epi_len) in enumerate(zip(batch_list, num_transitions)):
            for k, v in item.items():
                batch[k][i, :epi_len] = v
            padding_mask[i, epi_len:] = True

        batch = Episode(padding_mask=padding_mask, **batch)

        return batch

    def as_episode_dataset(self, *args, **kwargs):
        return OfflineEpisodeWrapper(self, *args, **kwargs)
