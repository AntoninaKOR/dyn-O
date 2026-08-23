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


class RunLogger:
    """Discards everything. Used for ranks that must not write to the tracking server."""

    def log_metric(self, name: str, value: float, step: int, step_kind: str = "step") -> None:
        pass

    def log_image(self, name: str, image: np.ndarray, step: int, step_kind: str = "step") -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger(RunLogger):
    def __init__(self, args, run_name: str, repo_path: Path):
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
        # one panel per group rather than per file, so the frames form a slider
        key = f"viz/{name.split('/')[0]}"
        self._declare(key, step_kind)
        self._wandb.log({key: self._wandb.Image(image), step_kind: step})

    def finish(self):
        self._wandb.finish()


class CometLogger(RunLogger):
    def __init__(self, args, run_name: str):
        import comet_ml

        # comet_ml.login() falls back to prompting on stdin, which under torchrun hangs the
        # job with no output, so refuse up front instead
        if not comet_ml.get_config()["comet.api_key"]:
            raise RuntimeError(
                "No Comet API key found. Set COMET_API_KEY (and optionally COMET_WORKSPACE) "
                "or run `comet login` to write ~/.comet.config."
            )

        self._experiment = comet_ml.start(
            project_name=args.exp_name,
            experiment_config=comet_ml.ExperimentConfig(name=run_name),
        )
        self._experiment.log_parameters(dataclasses.asdict(args))

    def log_metric(self, name, value, step, step_kind="step"):
        if step_kind == "epoch":
            self._experiment.log_metric(name=name, value=value, epoch=step)
        else:
            self._experiment.log_metric(name=name, value=value, step=step)

    def log_image(self, name, image, step, step_kind="step"):
        self._experiment.log_image(
            image_data=_to_uint8(image),
            name=f"viz/{name.split('/')[0]}",
            step=step,
        )

    def finish(self):
        self._experiment.end()


class MlflowLogger(RunLogger):
    def __init__(self, args):
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
