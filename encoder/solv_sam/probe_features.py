"""
Cluster patch features to see what an encoder makes separable, before spending a run on it.

Slot attention can only cut a scene along boundaries that already exist in its input
features, so k-means over those features is an upper bound on what the slots could ever
discover, and unlike training it costs a couple of minutes. Clusters that trace furniture
mean the features carry objects; clusters that trace the floor tiling mean they carry
colour, and no amount of training will turn one into the other.

    python encoder/solv_sam/probe_features.py --data_path /data/minari/dyno-homegrid-v0
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from config import SolvSamConfig
from models.model import CosmosEncoder, DinoEncoder

PALETTE = np.array([
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
    (174, 199, 232), (255, 187, 120), (152, 223, 138), (255, 152, 150), (197, 176, 213),
], dtype=np.uint8)


def load_frames(data_path: Path, num_episodes: int, frames_per_episode: int) -> np.ndarray:
    frames = []
    with h5py.File(data_path / "data" / "main_data.hdf5", "r") as file:
        for ep in list(file.keys())[:num_episodes]:
            observations = file[ep]["observations"]
            idxes = np.linspace(0, len(observations) - 1, frames_per_episode, dtype=int)
            frames.extend(observations[i] for i in idxes)
    return np.stack(frames)


def kmeans(features: torch.Tensor, num_clusters: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """features: (n, d) -> (n,) cluster ids, Lloyd's algorithm from a random subset."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centroids = features[torch.randperm(len(features), generator=generator)[:num_clusters]].clone()

    for _ in range(iters):
        assignment = torch.cdist(features, centroids).argmin(dim=1)
        for k in range(num_clusters):
            members = features[assignment == k]
            if len(members) > 0:
                centroids[k] = members.mean(dim=0)

    return torch.cdist(features, centroids).argmin(dim=1)


def encode(encoder_name: str, frames: np.ndarray, resize_to: int, interpolation: str) -> torch.Tensor:
    """frames: (n, h, w, 3) uint8 -> (n, num_tokens, token_dim) on cpu."""
    args = SolvSamConfig(encoder=encoder_name, resize_to=[resize_to, resize_to])

    # the dataset feeds Cosmos raw [0, 1] and normalises only for the ViT encoders
    if encoder_name.startswith("Cosmos"):
        encoder = CosmosEncoder(args).cuda().eval()
        normalize = torch.nn.Identity()
    else:
        encoder = DinoEncoder(args).cuda().eval()
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    modes = {"bilinear": T.InterpolationMode.BILINEAR, "nearest": T.InterpolationMode.NEAREST_EXACT}
    images = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    images = T.Resize((resize_to, resize_to), interpolation=modes[interpolation])(images)
    images = normalize(images).cuda()

    with torch.no_grad():
        features = torch.cat([encoder(image[None]).float().cpu() for image in images])

    del encoder
    torch.cuda.empty_cache()
    return features


def cluster_overlay(frame: np.ndarray, assignment: np.ndarray, size: int) -> np.ndarray:
    """Paint one frame with its cluster ids, kept semi transparent to keep sprites readable."""
    grid = int(round(len(assignment) ** 0.5))
    colors = PALETTE[assignment % len(PALETTE)].reshape(grid, grid, 3)
    colors = np.array(Image.fromarray(colors).resize((size, size), Image.NEAREST))

    frame = np.array(Image.fromarray(frame).resize((size, size), Image.NEAREST))
    return (0.45 * frame + 0.55 * colors).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--encoders", nargs="+", default=["Cosmos-0.1-Tokenizer-CI16x16", "dinov2-vitb-14"])
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=3)
    parser.add_argument("--frames_per_episode", type=int, default=4)
    parser.add_argument("--resize_to", type=int, default=224)
    # a 96 px frame is stretched more than twofold, and bilinear turns every sprite edge
    # into a gradient; nearest keeps the flat colours pixel art is made of
    parser.add_argument("--interpolation", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument("--out", type=Path, default=Path("probe_features.png"))
    cfg = parser.parse_args()

    frames = load_frames(cfg.data_path, cfg.num_episodes, cfg.frames_per_episode)
    print(f"{len(frames)} frames of {frames.shape[1]}x{frames.shape[2]}")

    size = cfg.resize_to
    rows = [np.concatenate([
        np.array(Image.fromarray(frame).resize((size, size), Image.NEAREST)) for frame in frames
    ], axis=1)]

    for encoder_name in cfg.encoders:
        features = encode(encoder_name, frames, cfg.resize_to, cfg.interpolation)
        print(f"{encoder_name}: {tuple(features.shape)}")

        # clustered jointly over all frames, so a colour means the same thing everywhere and
        # an object keeping its colour across frames is itself a sign the features hold up
        num_frames, num_tokens, _ = features.shape
        assignment = kmeans(features.reshape(num_frames * num_tokens, -1), cfg.num_clusters)
        assignment = assignment.reshape(num_frames, num_tokens).numpy()

        rows.append(np.concatenate([
            cluster_overlay(frame, frame_assignment, size)
            for frame, frame_assignment in zip(frames, assignment)
        ], axis=1))

    Image.fromarray(np.concatenate(rows, axis=0)).save(cfg.out)
    print(f"saved {cfg.out}, rows: raw, " + ", ".join(cfg.encoders))


if __name__ == "__main__":
    main()
