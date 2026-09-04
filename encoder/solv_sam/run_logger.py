"""Experiment-tracking backends behind a single interface.

Every logging site in ``train.py`` used to branch on whether wandb could be imported and
fall back to mlflow inline, so adding a third backend would have tripled those branches.
Each backend here owns its own quirks instead: wandb needs metrics declared against a
step metric before use, mlflow takes a bare step, and comet distinguishes step from epoch.

Backends log the run configuration when they are constructed, so a caller only has to
build the right one and then log metrics and images against it.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np

Backend = Literal["auto", "wandb", "comet", "mlflow"]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Visualisations arrive as floats in [0, 1]; PIL-backed loggers want bytes."""
    if np.issubdtype(image.dtype, np.floating):
        return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    return image


# A full visualisation grid runs into megabytes, and asset uploads crawl at tens of KB/s
# on some clusters until they stall outright. The disk copy stays full size.
UPLOAD_MAX_SIDE = 768


def _downscale(image: np.ndarray, max_side: int = UPLOAD_MAX_SIDE) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_side / max(height, width)
    if scale >= 1:
        return image

    from PIL import Image

    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.BILINEAR))


class RunLogger:
    """Discards everything. Used for ranks that must not write to the tracking server."""

    def __init__(self, image_dir: Path | None = None):
        self._image_dir = Path(image_dir) if image_dir is not None else None

    def log_metric(self, name: str, value: float, step: int, step_kind: str = "step") -> None:
        pass

    def log_image(self, name: str, image: np.ndarray, step: int, step_kind: str = "step") -> None:
        self.save_image(name, image)

    def save_image(self, name: str, image: np.ndarray) -> None:
        """Mirror a visualisation next to the checkpoints.

        Asset uploads reach the tracking server over a different endpoint than metrics and
        are blocked outright on some clusters, while the pictures are the only qualitative
        signal training produces. Keeping a local copy makes them independent of that.
        """
        if self._image_dir is None:
            return

        from PIL import Image

        path = self._image_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(_to_uint8(image)).save(path)

    def finish(self) -> None:
        pass


class WandbLogger(RunLogger):
    def __init__(self, args, run_name: str, repo_path: Path, image_dir: Path | None = None):
        super().__init__(image_dir)

        import wandb

        self._wandb = wandb
        self._declared: set[str] = set()
        wandb.init(
            project=args.exp_name,
            name=run_name,
            config=dataclasses.asdict(args),
            dir=repo_path / "wandb",
            mode="online" if (args.rank == 0 and args.exp_name != "test") else "disabled",
        )

    def _declare(self, name: str, step_kind: str) -> None:
        if name in self._declared:
            return
        self._wandb.define_metric(step_kind)
        self._wandb.define_metric(name, step_metric=step_kind)
        self._declared.add(name)

    def log_metric(self, name, value, step, step_kind="step"):
        self._declare(name, step_kind)
        self._wandb.log({name: value, step_kind: step})

    def log_image(self, name, image, step, step_kind="step"):
        self.save_image(name, image)

        # one panel per group rather than per file, so the frames form a slider
        key = f"viz/{name.split('/')[0]}"
        self._declare(key, step_kind)
        self._wandb.log({key: self._wandb.Image(image), step_kind: step})

    def finish(self):
        self._wandb.finish()


class CometLogger(RunLogger):
    def __init__(self, args, run_name: str, image_dir: Path | None = None):
        super().__init__(image_dir)

        import comet_ml

        # comet_ml.login() falls back to prompting on stdin, which under torchrun hangs the
        # job with no output, so refuse up front instead
        if not comet_ml.get_config()["comet.api_key"]:
            raise RuntimeError(
                "No Comet API key found. Set COMET_API_KEY (and optionally COMET_WORKSPACE) "
                "or run `comet login` to write ~/.comet.config."
            )

        if args.comet_experiment_key:
            # no ExperimentConfig here, it would rename the experiment being continued
            self._experiment = comet_ml.start(
                experiment_key=args.comet_experiment_key, mode="get"
            )
        else:
            self._experiment = comet_ml.start(
                project_name=args.exp_name,
                experiment_config=comet_ml.ExperimentConfig(name=run_name),
            )
        self._experiment.log_parameters(dataclasses.asdict(args))
        self._upload_images = args.upload_images

    def log_metric(self, name, value, step, step_kind="step"):
        if step_kind == "epoch":
            self._experiment.log_metric(name=name, value=value, epoch=step)
        else:
            self._experiment.log_metric(name=name, value=value, step=step)

    def log_image(self, name, image, step, step_kind="step"):
        self.save_image(name, image)

        if not self._upload_images:
            return

        self._experiment.log_image(
            image_data=_downscale(_to_uint8(image)),
            name=f"viz/{name.split('/')[0]}",
            image_format="jpeg",
            step=step,
        )

    def finish(self):
        self._experiment.end()


class MlflowLogger(RunLogger):
    def __init__(self, args, image_dir: Path | None = None):
        super().__init__(image_dir)

        import mlflow

        self._mlflow = mlflow
        mlflow.log_params(dataclasses.asdict(args))

    def log_metric(self, name, value, step, step_kind="step"):
        self._mlflow.log_metric(name, value, step)

    def log_image(self, name, image, step, step_kind="step"):
        # mlflow keeps artifacts as files, so the full path is the useful name
        self._mlflow.log_image(image, artifact_file=name)


def resolve_backend(requested: Backend) -> str:
    """Turn ``auto`` into a concrete backend, preferring wandb as the code always has."""
    if requested != "auto":
        return requested

    try:
        import wandb  # noqa: F401
    except ImportError:
        return "mlflow"
    return "wandb"
