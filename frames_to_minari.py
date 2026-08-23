"""Convert directories of image-sequence episodes into Minari datasets.

Expected input layout, one directory per split and one subdirectory per episode::

    <frames_root>/<split>/obs/<episode_dir>/
        s_0.png, s_1.png, ...   frames; the index is parsed out of the file name
        actions.npy             optional, length T - 1
        rewards.npy             optional, length T - 1

The produced datasets are what ``segment-anything-2/video_track_all_obj.py`` and
``encoder/solv_sam/train.py`` read. Both derive the environment name from the second
dash-separated token of the dataset id, so ids are built as
``<prefix>-<env_name>-v<version>``.

Example::

    python frames_to_minari.py \
        --frames_root /data/langroom_dataset \
        --data_root /data/minari \
        --env_name langroom
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
from unittest import mock

import gymnasium as gym
import h5py
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

import minari
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset.minari_storage import MinariStorage

IMAGE_SUFFIXES = (".png", ".jpeg", ".jpg", ".bmp", ".webp")


@dataclasses.dataclass
class Args:
    frames_root: str
    """Directory holding the per-split subdirectories."""
    data_root: str
    """Where the Minari datasets are written (used as MINARI_DATASETS_PATH)."""

    env_name: str = "langroom"
    """Second token of the dataset id; the SAM script keys its config off it."""
    dataset_prefix: str = "dyno"
    train_split: str = "train"
    valid_split: str = "val"
    train_version: int = 0
    valid_version: int = 1

    obs_subdir: str = "obs"
    """Subdirectory of a split that contains the episode directories."""
    frame_pattern: str = r"s_(\d+)"
    """Regex matched against the frame file stem; group 1 must be the time index."""

    resize_to: Optional[Tuple[int, int]] = None
    """Optional (height, width) to resize frames to. Native resolution if unset."""
    max_episodes: Optional[int] = None
    """Only convert the first N episodes of each split."""
    max_frames_per_episode: Optional[int] = None
    """Split longer episodes into consecutive chunks; SAM2 holds a whole episode on GPU."""
    drop_repeated_frames: bool = False
    """Drop frames identical to their predecessor. Breaks alignment with actions."""

    compression: Optional[str] = "gzip"
    """HDF5 compression for observations. Frames stay individually addressable."""
    last_step_terminates: bool = False
    """Mark the final step as a termination instead of a truncation."""
    overwrite: bool = False
    dry_run: bool = False


def _chunked_add_episode_to_group(episode_buffer: Dict, episode_group: h5py.Group, compression=None):
    """Replacement for Minari's writer that chunks observations one frame at a time.

    Encoder training reads a single random frame per sample, so a chunk must not span
    frames. Mirrors the patch that ``ppg_rollout_minari.py`` applies.
    """
    for key, data in episode_buffer.items():
        if data is None:
            continue

        if isinstance(data, dict):
            subgroup = episode_group.create_group(key) if key not in episode_group else episode_group[key]
            _chunked_add_episode_to_group(data, subgroup, compression=compression)
            continue

        data = np.asarray(data)
        dshape = data.shape[1:]
        episode_group.create_dataset(
            key,
            data=data,
            chunks=(1, *dshape) if key == "observations" else True,
            maxshape=(None, *dshape),
            compression=compression if key == "observations" else None,
        )


def natural_sort_key(name: str) -> Tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name))


def list_episode_dirs(split_dir: pathlib.Path, obs_subdir: str) -> List[pathlib.Path]:
    obs_dir = split_dir / obs_subdir
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"No episode directory at {obs_dir}")

    episode_dirs = [d for d in obs_dir.iterdir() if d.is_dir()]
    if not episode_dirs:
        raise FileNotFoundError(f"{obs_dir} contains no episode subdirectories")

    return sorted(episode_dirs, key=lambda d: natural_sort_key(d.name))


def list_frame_paths(episode_dir: pathlib.Path, frame_pattern: str) -> List[pathlib.Path]:
    """Order frames by the numeric index in their name, not lexicographically.

    Names like ``s_9`` / ``s_10`` sort the wrong way as strings, which would silently
    scramble the temporal order that SAM2 tracking depends on.
    """
    regex = re.compile(frame_pattern)
    indexed = []
    for path in episode_dir.iterdir():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = regex.fullmatch(path.stem)
        if match is None:
            continue
        indexed.append((int(match.group(1)), path))

    if not indexed:
        raise FileNotFoundError(
            f"{episode_dir} has no frames matching {frame_pattern!r} with a known image suffix"
        )

    indexed.sort(key=lambda item: item[0])
    idxes = [i for i, _ in indexed]
    if idxes != list(range(idxes[0], idxes[0] + len(idxes))):
        missing = sorted(set(range(idxes[0], idxes[-1] + 1)) - set(idxes))
        raise ValueError(f"{episode_dir} has gaps in its frame indices, missing {missing[:10]}")

    return [path for _, path in indexed]


def load_npy(path: pathlib.Path) -> Optional[np.ndarray]:
    return np.load(path, allow_pickle=True) if path.exists() else None


def to_array_dict(actions: np.ndarray) -> Dict[str, np.ndarray] | np.ndarray:
    """Turn an object array of per-step dicts into a dict of per-key arrays."""
    if actions.dtype != object or len(actions) == 0 or not isinstance(actions[0], dict):
        return actions
    return {
        key: np.asarray([np.asarray(step[key]) for step in actions])
        for key in sorted(actions[0].keys())
    }


def scan_action_space(episode_dirs: List[pathlib.Path]) -> gym.Space:
    """Derive a discrete action space from the actions stored next to the frames."""
    max_values: Dict[str, int] = {}
    scalar_max: Optional[int] = None
    found = False

    for episode_dir in episode_dirs:
        actions = load_npy(episode_dir / "actions.npy")
        if actions is None:
            continue
        found = True
        actions = to_array_dict(actions)
        if isinstance(actions, dict):
            for key, values in actions.items():
                max_values[key] = max(max_values.get(key, 0), int(values.max()))
        else:
            scalar_max = max(scalar_max or 0, int(np.asarray(actions).max()))

    if not found:
        return gym.spaces.Discrete(1)
    if max_values:
        return gym.spaces.Dict(
            {key: gym.spaces.Discrete(value + 1) for key, value in sorted(max_values.items())}
        )
    return gym.spaces.Discrete((scalar_max or 0) + 1)


def load_frames(paths: List[pathlib.Path], resize_to: Optional[Tuple[int, int]]) -> np.ndarray:
    frames = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if resize_to is not None and image.size != (resize_to[1], resize_to[0]):
            image = image.resize((resize_to[1], resize_to[0]), Image.BILINEAR)
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames)


def slice_step_data(
    data: Dict[str, np.ndarray] | np.ndarray | None,
    start: int,
    stop: int,
) -> Dict[str, np.ndarray] | np.ndarray | None:
    if data is None:
        return None
    if isinstance(data, dict):
        return {key: values[start:stop] for key, values in data.items()}
    return data[start:stop]


def drop_repeats(observations: np.ndarray, actions, rewards):
    """Drop frames identical to their predecessor, along with the steps leading to them."""
    keep = np.ones(len(observations), dtype=bool)
    keep[1:] = (observations[1:] != observations[:-1]).any(axis=(1, 2, 3))

    # step t is the transition into frame t + 1, so it survives iff that frame does
    step_keep = keep[1:]
    return observations[keep], _mask_step_data(actions, step_keep), _mask_step_data(rewards, step_keep)


def iter_episode_buffers(args: Args, episode_dirs: List[pathlib.Path]) -> Iterator[EpisodeBuffer]:
    for episode_dir in tqdm(episode_dirs, desc=f"converting {len(episode_dirs)} episodes"):
        frame_paths = list_frame_paths(episode_dir, args.frame_pattern)
        observations = load_frames(frame_paths, args.resize_to)

        actions = load_npy(episode_dir / "actions.npy")
        actions = to_array_dict(actions) if actions is not None else None
        rewards = load_npy(episode_dir / "rewards.npy")

        if args.drop_repeated_frames:
            observations, actions, rewards = drop_repeats(observations, actions, rewards)

        chunk = args.max_frames_per_episode or len(observations)
        for start in range(0, len(observations), chunk):
            stop = min(start + chunk, len(observations))
            if stop - start < 2:
                continue
            yield build_buffer(
                args,
                observations[start:stop],
                slice_step_data(actions, start, stop - 1),
                slice_step_data(rewards, start, stop - 1),
            )


def _mask_step_data(data, keep):
    if data is None:
        return None
    if isinstance(data, dict):
        return {key: values[keep] for key, values in data.items()}
    return data[keep]


def build_buffer(args: Args, observations: np.ndarray, step_actions, step_rewards) -> EpisodeBuffer:
    """Minari stores T observations against T - 1 transitions."""
    num_steps = len(observations) - 1

    if step_actions is None:
        step_actions = np.zeros(num_steps, dtype=np.int64)
    if step_rewards is None:
        step_rewards = np.zeros(num_steps, dtype=np.float32)

    _check_length(step_actions, num_steps, "actions")
    _check_length(step_rewards, num_steps, "rewards")

    terminations = np.zeros(num_steps, dtype=bool)
    truncations = np.zeros(num_steps, dtype=bool)
    if args.last_step_terminates:
        terminations[-1] = True
    else:
        truncations[-1] = True

    return EpisodeBuffer(
        id=None,
        observations=observations,
        actions=step_actions,
        rewards=np.asarray(step_rewards, dtype=np.float32),
        terminations=terminations,
        truncations=truncations,
    )


def _check_length(data, expected: int, name: str):
    lengths = (
        {len(v) for v in data.values()} if isinstance(data, dict) else {len(np.asarray(data))}
    )
    if lengths != {expected}:
        raise ValueError(
            f"{name} has length {lengths}, expected {expected} "
            f"(Minari stores T observations against T - 1 steps)"
        )


def convert_split(args: Args, split: str, version: int) -> Optional[str]:
    split_dir = pathlib.Path(args.frames_root) / split
    episode_dirs = list_episode_dirs(split_dir, args.obs_subdir)
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[: args.max_episodes]

    dataset_id = f"{args.dataset_prefix}-{args.env_name}-v{version}"
    dataset_path = pathlib.Path(args.data_root).resolve() / dataset_id

    probe = load_frames(list_frame_paths(episode_dirs[0], args.frame_pattern)[:1], args.resize_to)
    height, width = probe.shape[1:3]

    action_space = scan_action_space(episode_dirs)
    observation_space = gym.spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)

    print(
        f"\n=== {split} -> {dataset_id} ===\n"
        f"  episodes: {len(episode_dirs)}\n"
        f"  frame size: {height}x{width}\n"
        f"  observation space: {observation_space}\n"
        f"  action space: {action_space}\n"
        f"  output: {dataset_path}"
    )
    if args.dry_run:
        return None

    if dataset_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dataset_path} already exists; pass --overwrite to replace it"
            )
        import shutil

        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True)
    storage = MinariStorage.new(
        dataset_path / "data",
        observation_space=observation_space,
        action_space=action_space,
        data_format="hdf5",
    )
    storage.update_metadata(
        {
            "dataset_id": dataset_id,
            "minari_version": minari.__version__,
            "algorithm_name": "converted from image sequences",
        }
    )

    def writer(episode_buffer, episode_group):
        _chunked_add_episode_to_group(episode_buffer, episode_group, compression=args.compression)

    total_frames = 0
    with mock.patch(
        "minari.dataset._storages.hdf5_storage._add_episode_to_group", writer
    ):
        for buffer in iter_episode_buffers(args, episode_dirs):
            storage.update_episodes([buffer])
            total_frames += len(buffer.observations)

    size_mb = sum(p.stat().st_size for p in dataset_path.rglob("*") if p.is_file()) / 1e6
    print(f"  wrote {storage.total_episodes} episodes / {total_frames} frames ({size_mb:.1f} MB)")
    return dataset_id


def main(args: Args):
    os.environ["MINARI_DATASETS_PATH"] = str(pathlib.Path(args.data_root).resolve())

    created = []
    for split, version in (
        (args.train_split, args.train_version),
        (args.valid_split, args.valid_version),
    ):
        if not split:
            continue
        dataset_id = convert_split(args, split, version)
        if dataset_id is not None:
            created.append(dataset_id)

    if not created:
        return

    root = pathlib.Path(args.data_root).resolve()
    train_flags = (
        f"--root {root} --train_dataset_ids {created[0]} --valid_dataset_ids {created[-1]}"
    )

    print("\n=== next steps ===")
    print("with SAM supervision, first generate the masks:")
    for dataset_id in created:
        print(
            f"  python segment-anything-2/video_track_all_obj.py "
            f"--data_path {root / dataset_id} --cuda_ids 0"
        )
    print(f"  then train with {train_flags} --load_sam_masks")
    print(f"\nwithout masks, train directly with {train_flags} --no_load_sam_masks")


if __name__ == "__main__":
    main(tyro.cli(Args))
