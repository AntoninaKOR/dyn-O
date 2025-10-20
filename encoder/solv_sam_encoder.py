# a wrap-up of Visual_Encoder and SOLV for object-centric state-space model training

import dataclasses
from pathlib import Path
from typing import Union, Iterable, Literal, Optional

from einops import rearrange

import torch
import torch.nn as nn
import torchvision.transforms as T

from encoder.solv_sam.models.model import DinoEncoder, CosmosEncoder, SOLV_SAM
from oc_utils.utils import REPO_PATH, are_dicts_equal, pack_one, unpack_one


class SOLV_SAM_Encoder(nn.Module):
    def __init__(self, config):
        super(SOLV_SAM_Encoder, self).__init__()
        self.config = config
        self.dtype = config.dtype
        self.device = config.device

        # === Model ===
        self.args = config.encoder
        self.args.decode_use_mask = False

        self.decode_segmentation = self.args.decode_segmentation

        if self.args.encoder.startswith("Cosmos"):
            self.encoder = CosmosEncoder(self.args).to(self.device)
        else:
            self.encoder = DinoEncoder(self.args).to(self.device)
        self.solv = SOLV_SAM(self.args, self.encoder).to(self.device)

        if not self.args.finetune:
            for param in self.parameters():
                param.requires_grad = False

        # === Transformations ===
        self.resize = T.Resize(self.args.resize_to)
        self.resize_mask = T.Resize(self.args.resize_to, T.InterpolationMode.NEAREST_EXACT)

        if self.args.encoder.startswith("Cosmos"):
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        else:
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.normalize = T.Normalize(mean=mean, std=std)
            self.denormalize = T.Normalize(mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std])

        self.transform = T.Compose([self.resize, self.normalize])

        self.load_checkpoint(config.encoder.checkpoint_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.encoder.learning_rate)

    def load_checkpoint(self, checkpoint_path):
        assert checkpoint_path is not None, "=> SOLV_Encoder checkpoint_path is None"
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_PATH / checkpoint_path
        assert checkpoint_path.exists(), f"=> no SOLV_Encoder checkpoint found at '{checkpoint_path}'"

        # open checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # verify config is the same as encoder training
        assert are_dicts_equal(
            dataclasses.asdict(self.args),
            checkpoint["config"] if "config" in checkpoint else checkpoint["args"],
            only_use_common_keys=True,
            exclude_keys=["checkpoint_path", "batch_size"],
        ), "config mismatch (see above)"

        # load model state dict
        model_state_dict = checkpoint["model"]

        # remove ddp prefix
        model_state_dict = {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in model_state_dict.items()
        }
        msg = self.solv.load_state_dict(model_state_dict, strict=False)
        print(f"=> SOLV_Encoder loaded model from checkpoint:\n"
              f"{checkpoint_path}\n"
              f"with msg {msg}")

    def forward(
        self,
        x: torch.Tensor,
        masks: torch.Tensor = None,
        modes: Union[
            Iterable[Literal["enc_feat", "slots"]],
            Literal["enc_feat", "slots"]
        ] = "slots",
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        # x: (..., H, W, C) or (..., C, H, W), masks: (..., num_slots, H, W)

        assert self.args.use_sam_mask == (masks is not None), \
            "Mismatch between use_sam_mask and provided masks"

        # preprocess modes
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            assert mode in ["enc_feat", "slots"], f"Unknown modes: {mode}, available: enc_feat, slot"

        # preprocess input
        x, ps = pack_one(x, "* c h w")
        x = x.to(self.device, self.dtype, non_blocking=True)

        if x.max() > 1:
            x = (x / 255.0)

        if x.shape[-1] == 3:
            x = rearrange(x, "b h w c -> b c h w")

        x = self.transform(x)                                                   # (bs, C, H, W)

        # extract features
        with torch.amp.autocast("cuda"):
            features = self.encoder(x)                                          # (bs, num_tokens, token_dim)
            features = features.to(self.dtype)

        outputs = {
            "enc_feat": unpack_one(features, ps, "* num_tokens token_dim"),
        }

        if "slots" in modes:
            # preprocess masks
            if masks is not None:
                masks = masks.to(self.device, non_blocking=True)
                masks = self.resize_mask(masks)                                     # (bs, num_slots, H, W)

                # pixel-level mask -> token-level mask
                # (bs, num_slots, H, W) -> (B, num_slots, num_tokens)
                patch_size = self.args.patch_size
                masks = rearrange(masks, "b s (h p1) (w p2) -> b s (h w) (p1 p2)", p1=patch_size, p2=patch_size)
                masks = masks.any(dim=-1)                                           # (B, num_slots, num_tokens)

            # extract slots
            slots = self.solv.get_slots(features, masks)[0]                         # (bs, num_slots, slot_dim)

            outputs["slots"] = unpack_one(slots, ps, "* num_slots slot_dim")

        outputs = [outputs[mode] for mode in modes]

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def forward_episode(
        self,
        x: torch.Tensor,
        masks: torch.Tensor = None,
        modes: Union[
            Iterable[Literal["enc_feat", "slots"]],
            Literal["enc_feat", "slots"]
        ] = "slots",
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:

        # x: (T, H, W, C) or (T, C, H, W), masks: (T, num_slots, H, W)
        T = x.shape[0]
        bs = self.config.encoder.batch_size

        # preprocess modes
        if isinstance(modes, str):
            modes = [modes]

        outputs = [[] for _ in modes]
        for s in range(0, T, bs):
            output = self.forward(
                x[s:s + bs],
                None if masks is None else masks[s:s + bs],
                modes=modes,
            )
            for i in range(len(outputs)):
                outputs[i].append(output[i] if isinstance(output, list) else output)

        outputs = [torch.cat(output_i, dim=0) for output_i in outputs]

        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def decode(
        self,
        slots: torch.Tensor,
        slots_valid: Optional[torch.Tensor] = None,
        modes: Union[
            Iterable[Literal["patch_mask", "segmentation", "rgb_rec", "enc_feat_rec"]],
            Literal["patch_mask", "segmentation", "rgb_rec", "enc_feat_rec"]
        ] = "rgb_rec",
        requires_grad: bool = False,
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        # slots: (..., num_slots, slot_dim), slots_valid: (..., num_slots)

        # preprocess modes
        if isinstance(modes, str):
            modes = [modes]

        for mode in modes:
            assert mode in ["patch_mask", "segmentation", "rgb_rec", "enc_feat_rec"], \
                f"Unknown mode: {mode}, available: patch_mask, segmentation, rgb_rec, enc_feat_rec"

        if not self.args.decode_segmentation and "segmentation" in modes:
            raise ValueError("Segmentation output is not supported in this model.")

        with torch.set_grad_enabled(requires_grad):
            # preprocess slots, slots_valid
            slots, ps = pack_one(slots, "* s d")
            slots = slots.to(self.device, non_blocking=True)

            if slots_valid is not None:
                slots_valid, _ = pack_one(slots_valid, "* s")                       # (bs, num_slots)
                slots_valid = slots_valid.to(self.device, non_blocking=True)

            num_frames = slots.shape[0]
            patch_size = self.args.patch_size
            H, W = self.args.resize_to
            num_h_patches, num_w_patches = H // patch_size, W // patch_size

            bs = self.args.batch_size if not requires_grad else num_frames

            # === reconstruction extraction ===
            reconstructions = []
            for s in range(0, num_frames, bs):
                reconstruction = self.solv.decode(
                    slots=slots[s:s + bs],
                    slots_valid=None if slots_valid is None else slots_valid[s:s + bs],
                    decode_rgb="rgb_rec" in modes,
                )
                reconstructions.append(reconstruction)

            output = []
            for mode in modes:
                output_i = torch.cat(
                    [reconstruction[mode] for reconstruction in reconstructions],
                    dim=0,
                )

                if mode == "patch_mask":
                    # output_i: (bs, num_slots, num_tokens)
                    output_i = unpack_one(output_i, ps, "* num_slots num_tokens")

                elif mode == "segmentation":
                    # output_i: (bs, num_slots, num_tokens, path_size ** 2) -> (bs, num_slots, H, W)
                    output_i = rearrange(
                        output_i, "b s (h_p w_p) (p1 p2) -> b s (h_p p1) (w_p p2)",
                        h_p=num_h_patches, w_p=num_w_patches, p1=patch_size, p2=patch_size,
                    )

                    output_i = unpack_one(output_i, ps, "* num_slots h w")

                elif mode == "rgb_rec":
                    # output_i: (bs, num_tokens, path_size ** 2, 3) -> (bs, 3, H, W)
                    if output_i.shape[-2] == patch_size ** 2 and output_i.shape[-1] == 3:
                        output_i = rearrange(
                            output_i,
                            "b (h_p w_p) (p1 p2) c -> b c (h_p p1) (w_p p2)",
                            h_p=num_h_patches, w_p=num_w_patches, p1=patch_size, p2=patch_size,
                        )
                    output_i = self.denormalize(output_i)

                    if not requires_grad:
                        output_i = output_i.clip(0, 1)

                    output_i = unpack_one(output_i, ps, "* c h w")

                elif mode == "enc_feat_rec":
                    # output_i: (bs, num_tokens, token_dim)
                    output_i = unpack_one(output_i, ps, "* num_tokens token_dim")

                else:
                    raise ValueError(f"Unknown mode: {mode}, available: mask, segmentation, rec")

                output.append(output_i)

        if len(output) == 1:
            output = output[0]

        return output

    def decode_enc_feat_to_rgb(
        self,
        enc_feat: torch.Tensor,
        requires_grad: bool = False,
    ) -> torch.Tensor:

        with torch.set_grad_enabled(requires_grad):
            # enc_feat: (..., num_tokens, token_dim)
            enc_feat, ps = pack_one(enc_feat, "* num_tokens token_dim")

            enc_feat = enc_feat.to(self.device, non_blocking=True)

            num_frames = enc_feat.shape[0]
            bs = self.config.encoder.batch_size if not requires_grad else num_frames

            patch_size = self.config.encoder.patch_size
            H, W = self.config.encoder.resize_to
            num_h_patches, num_w_patches = H // patch_size, W // patch_size

            # === slot extraction ===
            rgb_recs = []
            for s in range(0, num_frames, bs):
                rgb_rec = self.solv.rgb_dec(enc_feat[s:s + bs])
                rgb_recs.append(rgb_rec)

            rgb_rec = torch.cat(rgb_recs, dim=0)                                    # (bs, num_tokens, patch_size ** 2, 3)

            # rgb_recs: (bs, num_tokens, path_size ** 2, 3) -> (bs, 3, H, W)
            if rgb_rec.shape[-2] == patch_size ** 2 and rgb_rec.shape[-1] == 3:
                rgb_rec = rearrange(
                    rgb_rec,
                    "b (h_p w_p) (p1 p2) c -> b c (h_p p1) (w_p p2)",
                    h_p=num_h_patches, w_p=num_w_patches, p1=patch_size, p2=patch_size,
                )

            rgb_rec = self.denormalize(rgb_rec)

            if not requires_grad:
                rgb_rec = rgb_rec.clip(0, 1)

            rgb_rec = unpack_one(rgb_rec, ps, "* c h w")

        return rgb_rec

    def update(self, replay_buffer, step):
        raise NotImplementedError
