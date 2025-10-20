import os
import tyro
import time
import mlflow
import datetime
import warnings
import dataclasses
from typing import Tuple

# Suppress only FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import wandb
    wandb_available = True
except:
    wandb_available = False

import numpy as np

from tqdm import tqdm
from pathlib import Path
from einops import rearrange, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import utils
from models.model import SOLV_SAM, DinoEncoder, CosmosEncoder
from config import SolvSamConfig


def perturb_mask(patch_masks: torch.Tensor, args: SolvSamConfig, cur_epoch: float, num_iter: int) -> Tuple[torch.Tensor, float]:
    # :arg patch_masks: (bs, num_slots, num_tokens)
    # :arg args: SolvSamConfig
    # :arg cur_epoch: float
    # :arg num_iter: int
    # :return new_patch_masks: (bs, num_slots, num_tokens)

    # get fill probability
    start, end = args.schedule_start_epoch, args.schedule_end_epoch
    if args.schedule_type == "log":
        k = args.log_schedule_k
        drop_prob = np.log(1 + k * max(cur_epoch - start, 0)) / np.log(1 + k * (end - start))
    elif args.schedule_type == "linear":
        drop_prob = (cur_epoch - start) / (end - start)
    else:
        raise ValueError(f"Invalid schedule type: {args.schedule_type}")
    drop_prob = np.clip(drop_prob, 0, 1)

    if wandb_available:
        wandb.define_metric("step")
        wandb.define_metric("batch/drop_prob", step_metric="step")
        wandb.log({"batch/drop_prob": drop_prob, "step": num_iter})
    else:
        mlflow.log_metric("batch/drop_prob", drop_prob, num_iter)

    new_patch_masks = patch_masks.clone()
    bs, num_slots, N = new_patch_masks.shape
    factory_kwargs = {"device": new_patch_masks.device, "dtype": torch.float32}

    # select frames to change
    change_frames = torch.rand(bs, **factory_kwargs) > args.no_drop_ratio
    change_frames = repeat(change_frames, "b -> b s n", s=num_slots, n=N)

    # perturb mask on the patch / slot / frame level
    if args.drop_type == "patch":
        false_entries = ~new_patch_masks
        change_entries = torch.rand_like(new_patch_masks.float(), **factory_kwargs) < drop_prob
        change_entries = false_entries & change_entries & change_frames
        new_patch_masks[change_entries] = True
    elif args.drop_type == "slot":
        change_slots = torch.rand((bs, num_slots), **factory_kwargs) < drop_prob
        change_slots = repeat(change_slots, "b s -> b s n", n=N)
        change_slots = change_frames & change_slots
        new_patch_masks[change_slots] = True
    elif args.drop_type == "frame":
        new_patch_masks[change_frames] = True
    else:
        raise ValueError(f"Invalid drop type: {args.drop_type}")

    return new_patch_masks, drop_prob


def train_epoch(args, encoder, model, optimizer, scheduler, train_dataloader, num_iter, total_iter):
    total_loss = 0.0
    last_save_time = time.time()
    encoder.eval()
    model.train()

    loader = tqdm(train_dataloader, disable=(args.gpu != 0))

    patch_size = args.patch_size
    num_epoch_iter = len(train_dataloader)
    num_batches_to_skip = num_iter % num_epoch_iter

    for i, (frames, masks) in enumerate(loader):

        # when resuming training, skip the batches that have been trained
        if i < num_batches_to_skip:
            continue

        frames = frames.cuda(non_blocking=True)                                                 # (bs, 3, h, w)
        masks = masks.cuda(non_blocking=True)                                                   # (bs, num_slots, h, w)

        # (bs, num_slots, h, w) -> (bs, num_slots, num_tokens, path_size ** 2)
        patch_masks = rearrange(masks, "b s (h p1) (w p2) -> b s (h w) (p1 p2)", p1=patch_size, p2=patch_size)
        patch_masks = patch_masks.any(dim=-1)                                                   # (bs, num_slots, num_tokens)

        if args.no_drop_ratio < 1:
            input_patch_masks, drop_prob = perturb_mask(patch_masks, args, cur_epoch=num_iter / num_epoch_iter, num_iter=num_iter)
        else:
            input_patch_masks = patch_masks
            drop_prob = 0.0

        with torch.amp.autocast('cuda'):
            enc_features = encoder(frames)
            assert torch.isfinite(enc_features).all()

        enc_features = enc_features.to(torch.float32)                                           # (bs, num_tokens, token_dim)
        reconstruction = model(
            enc_features,
            mask=input_patch_masks if args.encode_use_mask else None,
        )

        loss_terms = compute_loss(frames, masks, enc_features, reconstruction, patch_size)
        attn_loss = loss_terms["attn_loss"] * args.attn_loss_weight

        # if all patches are dropped, do not compute attn loss, as slots can be permuted in any way
        if drop_prob == 1.0:
            attn_loss = attn_loss.detach()
        loss_terms["attn_loss"] = attn_loss

        loss = loss_terms["loss"] = sum(loss_terms.values())

        total_loss += loss.item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if args.rank == 0:
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            mean_loss = total_loss / (i + 1)
            loader.set_description(f"lr: {lr:.6f} | loss: {mean_loss:.5f}")

            for loss_term_name, loss_term_value in loss_terms.items():
                if wandb_available:
                    wandb.define_metric("step")
                    wandb.define_metric(f"batch/{loss_term_name}", step_metric="step")
                    wandb.log({f"batch/{loss_term_name}": loss_term_value.item(), "step": num_iter})
                else:
                    mlflow.log_metric(f"batch/{loss_term_name}", loss_term_value.item(), num_iter)

            if num_iter % args.train_visualize_freq == 0:
                num_digits = len(str(total_iter))
                fname = f"train/iter_{num_iter:0{num_digits}d}.png"
                log_mask_visualization(
                    fname,
                    frames,
                    masks,
                    input_patch_masks.float(),
                    reconstruction,
                    patch_size,
                    train_dataloader.dataset.denormalize,
                    iteration=num_iter,
                )
        
        num_iter += 1

        # === Save Checkpoint ===
        if args.rank == 0 and time.time() - last_save_time > args.backup_per_second:
            save_dict = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "num_iter": num_iter,
                "config": dataclasses.asdict(args),
            }
            utils.save_on_master(args, save_dict, f"checkpoint_latest.pt")
            last_save_time = time.time()

    mean_loss = total_loss / (i + 1)
    return mean_loss, num_iter


@torch.inference_mode()
def val_epoch(args, encoder, model, val_dataloader, evaluator, epoch, use_mask=True):
    encoder.eval()
    model.eval()
    val_loader = tqdm(val_dataloader)

    bs = args.batch_size // args.gpus
    h, w = args.resize_to
    patch_size = args.patch_size
    num_slots = args.num_slots
    num_patch_h, num_patch_w = h // patch_size, w // patch_size

    mlflow_prefix = f"val_{'w' if use_mask else 'wo'}_mask"

    # compute which idxes to visualize
    num_videos = len(val_loader)
    num_viz_videos = args.val_visualize_num_videos
    num_viz_frames = args.val_visualize_num_frames_per_video

    viz_offset = epoch % num_viz_videos
    if num_videos < num_viz_videos:
        viz_idxes = np.arange(num_videos)
    else:
        viz_idxes = np.arange(viz_offset, num_videos, num_videos // num_viz_videos)

    losses = []

    for idx, (frames, gt_masks) in enumerate(val_loader):

        frames = frames[0].cuda(non_blocking=True)                                      # (T, 3, h, w)
        gt_masks = gt_masks[0].cuda(non_blocking=True)                                  # (T, num_slots, h, w)

        T, num_slots, _, _ = gt_masks.shape

        # (T, num_slots, h, w) -> (T, num_slots, num_tokens, path_size ** 2)
        patch_masks = rearrange(gt_masks, "b s (h p1) (w p2) -> b s (h w) (p1 p2)", p1=patch_size, p2=patch_size)
        patch_masks = patch_masks.any(dim=-1)                                           # (T, num_slots, num_tokens)

        # === encoder feature extraction ===
        output_features = []
        reconstructions = []
        for s in range(0, T, bs):
            e = min(s + bs, T)
            with torch.amp.autocast('cuda'):
                features = encoder(frames[s:e])                                     # (bs, token_num, token_dim)
                assert torch.isfinite(features).all()
                features = features.to(torch.float32)

            reconstruction = model(
                features,
                mask=patch_masks[s:e] if args.encode_use_mask and use_mask else None,
            )

            output_features.append(features)
            reconstructions.append(reconstruction)

        output_features = torch.cat(output_features, dim=0)

        # list of dictionaries -> dictionary of lists
        reconstruction = {}
        for k in reconstructions[0]:
            if reconstructions[0][k] is None:
                reconstruction[k] = None
            else:
                reconstruction[k] = torch.cat([r[k] for r in reconstructions], dim=0)

        loss_terms = compute_loss(frames, gt_masks, output_features, reconstruction, patch_size)

        if args.rank == 0:
            losses.append(
                {k: v.item() for k, v in loss_terms.items()}
            )

        # visualize
        if idx in viz_idxes:
            start_frame = 0
            end_frame = min(T, 15 * num_viz_frames)         # procgen is 15hz, so at most skip 15 frames per viz
            if end_frame - start_frame < num_viz_frames:
                frame_idxes = np.arange(start_frame, end_frame)
            else:
                frame_idxes = np.linspace(start_frame, end_frame, num_viz_frames, dtype=int, endpoint=False)

            # original images
            log_mask_visualization(
                f"{mlflow_prefix}/epoch_{epoch:02d}_ep_{idx}.png",
                frames[frame_idxes],
                gt_masks[frame_idxes],
                patch_masks[frame_idxes].float(),
                {
                    k: None if v is None else v[frame_idxes]
                    for k, v in reconstruction.items()
                },
                patch_size,
                val_dataloader.dataset.denormalize,
                epoch=epoch,
            )

        if "segmentation" not in reconstruction:
            segmentation = repeat(reconstruction["patch_mask"], "t s n -> t s n p", p=patch_size * patch_size)
        else:
            segmentation = reconstruction["segmentation"]

        segmentation = rearrange(
            segmentation, "t s (h_p w_p) (p1 p2) -> t s (h_p p1) (w_p p2)",
            h_p=num_patch_h, w_p=num_patch_w, p1=patch_size, p2=patch_size,
        )                                                                           # (T, num_slots, h, w)

        pixel_valid = gt_masks.any(dim=1, keepdim=True)                             # (T, 1, h, w)
        segmentation[~pixel_valid.expand_as(segmentation)] = 0

        predictions = F.interpolate(segmentation, size=(h, w), mode="bilinear")         # (T, num_slots, h, w)
        predictions = torch.argmax(predictions, dim=1)                                  # (T, h, w)
        gt_masks = torch.argmax(gt_masks.float(), dim=1)                                # (T, h, w)

        # === Instance Segmentation Evaluation ===
        miou = evaluator.update(predictions, gt_masks) if num_slots > 1 else 1.0
        loss_desc = f"mIoU: {miou * 100:.3f}"

        # === Logger ===
        val_loader.set_description(loss_desc)

    # === Loss Logging ===
    if args.rank == 0:
        for loss_term_name, loss_term_value in losses[0].items():
            loss_term_value = np.mean([loss[loss_term_name] for loss in losses])
            if wandb_available:
                wandb.define_metric("epoch")
                wandb.define_metric(f"{mlflow_prefix}/{loss_term_name}", step_metric="epoch")
                wandb.log({f"{mlflow_prefix}/{loss_term_name}": loss_term_value, "epoch": epoch})
            else:
                mlflow.log_metric(f"{mlflow_prefix}/{loss_term_name}", loss_term_value, epoch)

    # === Evaluation Results ====
    miou, fg_ari = evaluator.get_results() if num_slots > 1 else (1.0, 1.0)

    # === Logger ===
    print("\n=== Results ===")
    print(f"\tmIoU: {miou * 100:.3f}")
    print(f"\tFG-ARI: {fg_ari * 100:.3f}\n")

    if args.rank == 0:
        if wandb_available:
            wandb.define_metric("epoch")
            wandb.define_metric(f"{mlflow_prefix}/mIoU", step_metric="epoch")
            wandb.define_metric(f"{mlflow_prefix}/FG-ARI", step_metric="epoch")
            wandb.log({f"{mlflow_prefix}/mIoU": miou, f"{mlflow_prefix}/FG-ARI": fg_ari, "epoch": epoch})
        else:
            mlflow.log_metric(f"{mlflow_prefix}/mIoU", miou, epoch)
            mlflow.log_metric(f"{mlflow_prefix}/FG-ARI", fg_ari, epoch)

    dist.barrier()

    return miou, fg_ari


def compute_loss(frames, gt_segmentations, enc_features, reconstruction, patch_size):
    _, _, h, w = frames.shape
    num_patch_h, num_patch_w = h // patch_size, w // patch_size

    # === enc_feat loss ===
    token_valid = reconstruction["token_valid"]                                             # (bs, num_tokens)
    enc_feat_rec = reconstruction["enc_feat_rec"]                                           # (bs, num_tokens, token_dim)

    # if mask is provided, only compute loss on patches that are occupied
    if token_valid is not None:
        enc_features, enc_feat_rec = enc_features[token_valid], enc_feat_rec[token_valid]

    enc_feat_loss = F.mse_loss(enc_feat_rec, enc_features)
    enc_feat_abs_error = (enc_features - enc_feat_rec).abs().mean()
    enc_feat_rel_error = enc_feat_abs_error / enc_features.std(dim=0).mean()

    # === rgb loss ===
    masks_flat = rearrange(gt_segmentations, "b s h w -> (b h w) s")                        # (bs * h * w, num_slots)
    pixel_valid = masks_flat.any(dim=1)                                                     # (bs * h * w)

    rgb_rec = reconstruction["rgb_rec"]                                                     # (bs, C, h, w)
    rgb_rec_flat = rearrange(rgb_rec, "b c h w -> (b h w) c")                               # (bs * h * w, C)
    frames_flat = rearrange(frames, "b c h w -> (b h w) c")                                 # (bs * h * w, C)

    rgb_rec_loss = F.mse_loss(rgb_rec_flat[pixel_valid], frames_flat[pixel_valid])

    # === attn loss ===
    patch_masks = rearrange(gt_segmentations, "b s (h p1) (w p2) -> b s (h w) (p1 p2)", p1=patch_size, p2=patch_size)
    patch_masks = patch_masks.any(dim=-1)                                                   # (bs, num_slots, num_tokens)
    attn = reconstruction["unweighted_attn"]
    attn_loss = (attn[~patch_masks]).abs().mean()

    loss = {
        "enc_feat_loss": enc_feat_loss,
        "rgb_rec_loss": rgb_rec_loss,
        "attn_loss": attn_loss,
        # just for logging, detach to remove gradient
        "enc_feat_rel_error": enc_feat_rel_error.detach(),
    }

    # === segmentation Loss ===
    if "segmentation_logits" in reconstruction:
        segmentation_logits = reconstruction["segmentation_logits"]                         # (bs, num_slots, num_tokens, path_size ** 2)
        segmentation_logits = rearrange(
            segmentation_logits, "b s (h_p w_p) (p1 p2) -> b s (h_p p1) (w_p p2)",
            h_p=num_patch_h, w_p=num_patch_w, p1=patch_size, p2=patch_size,
        )                                                                                   # (bs, num_slots, h, w)
        segmentation_logits_flat = rearrange(segmentation_logits, "b s h w -> (b h w) s")   # (bs * h * w, num_slots)
        segmentation_loss = F.cross_entropy(
            segmentation_logits_flat[pixel_valid],
            masks_flat[pixel_valid].long().argmax(dim=-1),
        )
        loss["segmentation_loss"] = segmentation_loss

    return loss


def log_mask_visualization(
    fname,
    frames,
    gt_segmentations,
    gt_patch_segmentations,
    reconstruction,
    patch_size,
    denormalize_fn,
    iteration=None,
    epoch=None,
):
    _, _, h, w = frames.shape
    num_patch_h, num_patch_w = h // patch_size, w // patch_size

    frames = denormalize_fn(frames)
    rgb_rec = denormalize_fn(reconstruction["rgb_rec"]).clip(0, 1)

    images = [frames, rgb_rec, frames, frames, frames]
    masks = [None, None, gt_patch_segmentations, reconstruction["patch_mask"], gt_segmentations]

    if "segmentation" in reconstruction:
        segmentations = rearrange(
            reconstruction["segmentation"],
            "b s (h_p w_p) (p1 p2) -> b s (h_p p1) (w_p p2)",
            h_p=num_patch_h, w_p=num_patch_w, p1=patch_size, p2=patch_size,
        )  # (bs, num_slots, h, w)

        pixel_valid = gt_segmentations.any(dim=1, keepdim=True)
        rgb_rec[~pixel_valid.expand_as(rgb_rec)] = 0
        segmentations[~pixel_valid.expand_as(segmentations)] = 0

        images.append(frames)
        masks.append(segmentations)

    viz = []
    for img, mask in zip(images, masks):
        viz.append(utils.visualize_mask(img, mask))  # [3, h, w]

    try:
        if wandb_available:
            fname = fname.split("/")[0]
            if iteration is not None:
                wandb.define_metric("step")
                wandb.define_metric(f"viz/{fname}", step_metric="step")
                wandb.log({f"viz/{fname}": wandb.Image(torch.cat(viz, dim=1).permute(1, 2, 0).cpu().numpy()), "step": iteration})
            elif epoch is not None:
                wandb.define_metric("epoch")
                wandb.define_metric(f"viz/{fname}", step_metric="epoch")
                wandb.log({f"viz/{fname}": wandb.Image(torch.cat(viz, dim=1).permute(1, 2, 0).cpu().numpy()), "epoch": epoch})
        else:
            mlflow.log_image(
                torch.cat(viz, dim=1).permute(1, 2, 0).cpu().numpy(),
                artifact_file=fname,
            )
    except:
        print(f"Failed to log image {fname} to mlflow")


def main_worker(args):
    if args.rank == 0 and not wandb_available:
        mlflow.log_params(dataclasses.asdict(args))
    # === Dataloaders ====
    train_dataloader, val_dataloader = utils.get_dataloaders(args)

    # === Model ===
    if args.encoder.startswith("Cosmos"):
        encoder = CosmosEncoder(args).cuda()
    else:
        encoder = DinoEncoder(args).cuda()
    model = SOLV_SAM(args, encoder).cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Training Items ===
    optimizer = torch.optim.Adam(utils.get_params_groups(model), lr=args.learning_rate)
    scheduler = utils.get_scheduler(args, optimizer, train_dataloader)

    # === Misc ===
    evaluator = utils.Evaluator()

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"num_iter": 0}
    utils.restart_from_checkpoint(args,
                                  run_variables=to_restore,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler)
    num_iter = to_restore["num_iter"]

    start_time = time.time()
    dist.barrier()

    print("Starting SOLV_SAM training!")

    best_miou = -1
    start_epoch = num_iter // len(train_dataloader) + 1
    print(f"Starting SOLV_SAM training from epoch {start_epoch} iter {num_iter}!")
    total_iter = len(train_dataloader) * args.num_epochs
    for epoch in range(start_epoch, args.num_epochs + 1):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        mean_loss, num_iter = train_epoch(args, encoder, model, optimizer, scheduler, train_dataloader, num_iter, total_iter)

        if args.rank == 0:
            if wandb_available:
                wandb.define_metric("epoch")
                wandb.define_metric("epoch/train-loss", step_metric="epoch")
                wandb.log({"epoch/train-loss": mean_loss, "epoch": epoch})
            else:
                mlflow.log_metric("epoch/train-loss", mean_loss, epoch)

        # === Save Checkpoint ===
        save_dict = {
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "num_iter": num_iter,
            "config": dataclasses.asdict(args),
        }
        utils.save_on_master(args, save_dict, f"checkpoint_epoch_{epoch}.pt")

        # === Validate ===
        if (epoch == 1) or (epoch % args.validation_epoch == 0):
            if args.encode_use_mask:
                miou, _ = val_epoch(args, encoder, model, val_dataloader, evaluator, epoch, use_mask=True)
                if args.no_drop_ratio < 1:
                    _, _ = val_epoch(args, encoder, model, val_dataloader, evaluator, epoch, use_mask=False)
            else:
                miou, _ = val_epoch(args, encoder, model, val_dataloader, evaluator, epoch, use_mask=False)

            if miou > best_miou:
                best_miou = miou
                utils.save_on_master(args, save_dict, f"best_checkpoint.pt")

        print("===== ===== ===== ===== =====")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    dist.destroy_process_group()


if __name__ == "__main__":
    args = tyro.cli(SolvSamConfig)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    if wandb_available:
        args_dict = dataclasses.asdict(args)

        run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        repo_path = Path(__file__).resolve().parents[2]
        wandb.init(
            project=args.exp_name,
            name=run_name,
            config=args_dict,
            dir=repo_path / "wandb",
            mode="online" if (args.rank == 0 and args.exp_name != "test") else "disabled",
        )
        args.ckpt_path = repo_path / "results" / args.exp_name / run_name
        main_worker(args)
    else:
        if os.environ.get("AMLT_JOB_NAME", None) is None:
            if args.rank == 0:
                # Set Up MLflow Logging to Local File
                repo_path = Path(__file__).resolve().parents[2]
                args.logging_path = repo_path / "mlruns"                            # oc_ssm/mlruns
                mlflow.set_tracking_uri(args.logging_path)

                experiment = mlflow.set_experiment(args.exp_name)
                run_name = f"{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name) as run:
                    mlflow_run_path = args.logging_path / experiment.experiment_id / run.info.run_id
                    args.ckpt_path = mlflow_run_path / "artifacts"
                    main_worker(args)

                    # create a human-readable symlink for the run
                    symlink_path = repo_path / "results" / args.exp_name / run_name
                    symlink_path.parent.mkdir(parents=True, exist_ok=True)
                    symlink_path.symlink_to(mlflow_run_path, target_is_directory=True)
            else:
                args.ckpt_path = None
                main_worker(args)
        else:
            # AzureML running, need to prepare for getting killed and restarted
            run_name = f"{args.run_name}"
            args.ckpt_path = Path("/mnt/default/oc_ssm/encoder") / args.exp_name / run_name
            mlflow.enable_system_metrics_logging()                  # Enable system metrics logging on Azure
            mlflow.set_system_metrics_samples_before_logging(12)    # Log every 12*10=120 seconds
            main_worker(args)
