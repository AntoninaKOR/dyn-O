import copy
from einops import rearrange
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init
from transformers import AutoModel

import os
os.environ['LOGURU_LEVEL'] = 'WARNING'

import cosmos_tokenizer
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks.configs import continuous_image

EPS = 1e-7  # small constant to avoid division by zero for float16


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, residual=False, layer_order="none"):
        super().__init__()
        self.residual = residual
        self.layer_order = layer_order
        if residual:
            assert input_dim == output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)

        if layer_order in ["pre", "post"]:
            self.norm = nn.LayerNorm(input_dim)
        else:
            assert layer_order == "none"

    def forward(self, x):
        input = x

        if self.layer_order == "pre":
            x = self.norm(x)

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)

        if self.residual:
            x = x + input
        if self.layer_order == "post":
            x = self.norm(x)

        return x


class CosmosEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Set default model name and resize dimensions if not provided in args
        self.model_name = args.encoder
        self.feat_res = [size // args.patch_size for size in args.resize_to]

        # Set checkpoint paths for encoder and decoder
        cosmos_path = Path(cosmos_tokenizer.__file__).parent.parent
        self.encoder_ckpt = f"{cosmos_path}/pretrained_ckpts/{self.model_name}/encoder.jit"
        self.decoder_ckpt = f"{cosmos_path}/pretrained_ckpts/{self.model_name}/decoder.jit"

        # Initialize the ImageTokenizer with specified checkpoints
        self.tokenizer = ImageTokenizer(
            checkpoint_enc=self.encoder_ckpt,
            checkpoint_dec=self.decoder_ckpt,
            tokenizer_config=continuous_image,
            device="cuda",
            dtype="float32",
        )
        for p in self.tokenizer.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        # :arg x:  (bs, 3, h, w)

        # :return x:  (bs, num_tokens, 16)
        token = self.tokenizer.encode(x)[0]
        token = rearrange(token, "b c h w -> b (h w) c")
        return token

    @torch.no_grad()
    def forward(self, x):
        return self.encode(x)

    def decode(self, x):
        # :arg x:  (bs, num_tokens, 16)
        # :return x:  (bs, 3, h, w)
        x = rearrange(x, "b (h w) c -> b c h w", h=self.feat_res[0], w=self.feat_res[1])
        x = self.tokenizer.decode(x)
        return x


class DinoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.resize_to = args.resize_to
        self.feat_res = [size // args.patch_size for size in args.resize_to]
        self.token_num = args.token_num
        self.token_dim = args.token_dim

        self.encoder = args.encoder

        if dist.is_initialized():
            # avoid conflicts caused by multiple torch hub download
            rank = dist.get_rank()
            if rank == 0:
                self.model = self.load_model(args)
            dist.barrier()
            if rank != 0:
                self.model = self.load_model(args)
        else:
            self.model = self.load_model(args)

        self.dino = AutoModel.from_pretrained('facebook/dinov2-base')
        for p in self.dino.parameters():
            p.requires_grad = False

    def load_model(self, args):
        assert args.resize_to[0] % args.patch_size == 0
        assert args.resize_to[1] % args.patch_size == 0

        if args.encoder == "dinov2-vitb-14":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        elif args.encoder == "dinov2-vitb-14-reg":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        else:
            assert False

        for p in model.parameters():
            p.requires_grad = False

        return model

    @torch.no_grad()
    def forward(self, x):
        # :arg x:  (bs, 3, h, w)
        #
        # :return x:  (bs, num_tokens, token_dim)

        self.dino.eval()
        out = self.dino(x).last_hidden_state[:, 1:, :]
        return out

        self.model.eval()

        if self.encoder.startswith("dinov2"):
            x = self.model.prepare_tokens_with_masks(x)
        elif self.encoder.startswith("mae"):
            x = self.model.patch_embed(x)
            x = x + self.model.pos_embed[:, 1:, :]
            cls_token = self.model.cls_token + self.model.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            x = self.model.prepare_tokens(x)

        bs = x.shape[0]

        for blk in self.model.blocks:
            x = blk(x)

        x = x[:, 1:]                                    # remove cls token
        if "reg" in self.encoder:
            x = x[:, 4:]                                # remove register tokens

        assert x.shape[0] == bs
        assert x.shape[1] == self.token_num
        assert x.shape[2] == self.token_dim

        return x


class Spatial_Binder(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()

        self.num_slots = args.num_slots
        self.scale = args.slot_dim ** -0.5
        self.iters = args.slot_att_iter
        self.slot_dim = args.slot_dim

        self.res_h = args.resize_to[0] // args.patch_size
        self.res_w = args.resize_to[1] // args.patch_size
        self.num_tokens = int(self.res_h * self.res_w)

        # === abs_grid ===
        self.sigma = 5
        xs = torch.linspace(-1, 1, steps=self.res_w)  # (C_x)
        ys = torch.linspace(-1, 1, steps=self.res_h)  # (C_y)

        xs, ys = torch.meshgrid(xs, ys, indexing='xy')  # (C_x, C_y), (C_x, C_y)
        xs = xs.reshape(1, 1, -1, 1)  # (1, 1, C_x * C_y, 1)
        ys = ys.reshape(1, 1, -1, 1)  # (1, 1, C_x * C_y, 1)
        self.abs_grid = nn.Parameter(torch.cat([xs, ys], dim=-1), requires_grad=True)  # (1, 1, num_tokens, 2)
        assert self.abs_grid.shape[2] == self.num_tokens

        self.h = nn.Linear(2, self.slot_dim)
        # === === ===

        # === Slot related ===
        self.slots = nn.Parameter(torch.Tensor(1, self.num_slots, self.slot_dim))

        self.S_s = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, num_slots, 1, 2)
        self.S_p = nn.Parameter(torch.Tensor(1, self.num_slots, 1, 2))  # (1, num_slots, 1, 2)

        init.xavier_uniform_(self.slots)
        init.normal_(self.S_s, mean=0., std=.02)
        init.normal_(self.S_p, mean=0., std=.02)

        # avoid zero division
        sign = torch.sign(self.S_s.data)
        sign[sign == 0] = 1
        self.S_s.data = sign * (self.S_s.data.abs() + 1e-3)
        # === === ===

        # === Slot Attention related ===
        self.Q = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.norm = nn.LayerNorm(self.slot_dim)
        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)
        self.mlp = MLP(self.slot_dim, 4 * self.slot_dim, self.slot_dim,
                       residual=True, layer_order="pre")
        # === === ===

        # === Query & Key & Value ===
        self.K = nn.Linear(self.slot_dim, self.slot_dim, bias=False)
        self.V = nn.Linear(self.slot_dim, self.slot_dim, bias=False)

        self.g = nn.Linear(2, self.slot_dim)
        self.f = nn.Sequential(nn.Linear(self.slot_dim, self.slot_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.slot_dim, self.slot_dim))
        # === === ===

        # Note: starts and ends with LayerNorm
        self.initial_mlp = nn.Sequential(nn.LayerNorm(input_dim),
                                         nn.Linear(input_dim, input_dim),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(input_dim, self.slot_dim),
                                         nn.LayerNorm(self.slot_dim))

        self.final_layer = nn.Linear(self.slot_dim, self.slot_dim)

    def forward(self, inputs, mask=None):
        # :arg inputs:              (bs, num_tokens, token_dim)
        # :arg mask:                (bs, num_slots, num_tokens)
        #
        # :return slots:            (bs, num_slots, slot_dim)
        # :return attn:             (bs, num_slots, num_tokens)
        # :return all_attn:         (bs, F, num_slots, num_tokens)

        bs, num_tokens, token_dim = inputs.shape
        num_slots = self.num_slots
        slot_dim = self.slot_dim

        slots = self.slots.expand(bs, num_slots, slot_dim)                      # (bs, num_slots, slot_dim)
        if mask is not None:
            slot_valid = mask.any(dim=-1, keepdims=True)                        # (bs, num_slots, 1)
            token_valid = mask.any(dim=1, keepdims=True)                        # (bs, 1, num_tokens)
            attn_invalid = ~(slot_valid & token_valid)                          # (bs, num_slots, num_tokens)

        inputs = self.initial_mlp(inputs).unsqueeze(dim=1)                      # (bs, 1, num_tokens, slot_dim)
        inputs = inputs.expand(bs, num_slots, num_tokens, slot_dim)             # (bs, num_slots, num_tokens, slot_dim)

        abs_grid = self.abs_grid.expand(bs, num_slots, self.num_tokens, 2)      # (bs, num_slots, num_tokens, 2)

        S_s = self.S_s.expand(bs, num_slots, 1, 2)                              # (bs, num_slots, 1, 2)
        S_p = self.S_p.expand(bs, num_slots, 1, 2)                              # (bs, num_slots, 1, 2)

        S_s_sign = torch.sign(S_s)
        S_s_sign[S_s_sign == 0] = 1
        S_s = S_s_sign * (S_s.abs() + EPS)                                      # (bs, num_slots, 1, 2)

        for t in range(self.iters + 1):

            assert torch.isfinite(slots).all(), f"Iteration {t}"
            assert torch.isfinite(S_s).all(), f"Iteration {t}"
            assert torch.isfinite(S_p).all(), f"Iteration {t}"

            slots_prev = slots
            slots = self.norm(slots)

            # === key and value calculation using rel_grid ===
            rel_grid = (abs_grid - S_p) / (S_s * self.sigma)                    # (bs, num_slots, num_tokens, 2)
            k = self.f(self.K(inputs) + self.g(rel_grid))                       # (bs, num_slots, num_tokens, slot_dim)
            v = self.f(self.V(inputs) + self.g(rel_grid))                       # (bs, num_slots, num_tokens, slot_dim)

            # === Calculate attention ===
            q = self.Q(slots).unsqueeze(dim=-1)                                 # (bs, num_slots, slot_dim, 1)

            # (bs, num_slots, slot_dim, 1) x (bs, num_slots, num_tokens, slot_dim) -> (bs, num_slots, num_tokens)
            dots = torch.einsum('bsdi,bsjd->bsj', q, k)
            dots *= self.scale                                                  # (bs, num_slots, num_tokens)

            if mask is not None:
                dots[~mask] = float('-inf')                                     # (bs, num_slots, num_tokens)

            attn = dots.softmax(dim=1)                                          # (bs, num_slots, num_tokens)

            if mask is not None:
                attn = attn.masked_fill(attn_invalid, 0)

            unweighted_attn = attn

            # === Weighted mean ===
            attn = attn / (attn.sum(dim=-1, keepdim=True) + EPS)                # (bs, num_slots, num_tokens)
            attn = attn.unsqueeze(dim=2)                                        # (bs, num_slots, 1, num_tokens)

            # (bs, num_slots, num_tokens, slot_dim) x (bs, num_slots, 1, num_tokens) -> (bs, num_slots, slot_dim)
            updates = torch.einsum('bsjd,bsij->bsd', v, attn)

            # === Update S_p and S_s ===
            # (bs, num_slots, num_tokens, 2) x (bs, num_slots, 1, num_tokens) -> (bs, num_slots, 2)
            S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)
            S_p = S_p.unsqueeze(dim=2)                                          # (bs, num_slots, 1, 2)

            values_ss = torch.pow(abs_grid - S_p, 2)                            # (bs, num_slots, num_tokens, 2)
            # (bs, num_slots, num_tokens, 2) x (bs, num_slots, 1, num_tokens) -> (bs, num_slots, 2)
            S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)

            S_s_sign = torch.sign(S_s)
            S_s_sign[S_s_sign == 0] = 1
            S_s = S_s_sign * (S_s.abs() + EPS)  # (bs, num_slots, 1, 2)

            S_s = torch.sqrt(S_s)                                               # (bs, num_slots, 2)
            S_s = S_s.unsqueeze(dim=2)                                          # (bs, num_slots, 1, 2)

            # === Update ===
            if t != self.iters:
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                )

                slots = slots.reshape(bs, -1, self.slot_dim)
                slots = self.mlp(slots)

        slots = self.final_layer(slots)                                         # (bs, num_slots, slot_dim)
        attn = attn.reshape(bs, num_slots, num_tokens)                          # (bs, F, num_slots, num_tokens)

        return slots, attn, unweighted_attn

    def get_rel_grid(self, attn):
        # :arg attn: (bs, num_slots, num_tokens)
        #
        # :return: (bs * num_slots, num_tokens, slot_dim)

        bs, num_slots = attn.shape[:2]
        attn = attn.unsqueeze(dim=2)                                            # (bs, num_slots, 1, num_tokens)

        abs_grid = self.abs_grid.expand(bs, num_slots, self.num_tokens, 2)      # (bs, num_slots, num_tokens, 2)

        # (bs, num_slots, num_tokens, 2) x (bs, num_slots, 1, num_tokens) -> (bs, num_slots, 2)
        S_p = torch.einsum('bsjd,bsij->bsd', abs_grid, attn)
        S_p = S_p.unsqueeze(dim=2)                                              # (bs, num_slots, 1, 2)

        values_ss = torch.pow(abs_grid - S_p, 2)                                # (bs, num_slots, num_tokens, 2)

        # (bs, num_slots, num_tokens, 2) x (bs, num_slots, 1, num_tokens) -> (bs, num_slots, 2)
        S_s = torch.einsum('bsjd,bsij->bsd', values_ss, attn)
        S_s = torch.sqrt(S_s)                                                   # (bs, num_slots, 2)
        S_s = S_s.unsqueeze(dim=2)                                              # (bs, num_slots, 1, 2)

        rel_grid = (abs_grid - S_p) / (S_s * self.sigma)                        # (bs, num_slots, num_tokens, 2)
        rel_grid = rel_grid.reshape(bs * num_slots, self.res_h, self.res_w, 2)  # (bs * num_slots, h, w, 2)
        rel_grid = rel_grid.reshape(bs * num_slots, -1, 2)                      # (bs * num_slots, num_tokens // 16, 2)

        rel_grid = self.h(rel_grid)                                             # (bs * num_slots, num_tokens // 16, slot_dim)

        return rel_grid


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # === Token calculations ===
        slot_dim = args.slot_dim
        hidden_dim = 1024

        # === MLP Based Decoder ===
        self.layer1 = nn.Linear(slot_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        self.dino_layer = nn.Linear(hidden_dim, args.token_dim + 1)                 # enc_feat feature + score

        # pixel-level segmentation
        self.decode_segmentation = args.decode_segmentation
        if self.decode_segmentation:
            self.segment_layer = nn.Linear(hidden_dim + args.token_dim + 1, args.patch_size ** 2)

    def forward(self, slot_maps):
        # :arg slot_maps: (bs, num_slots, num_tokens, slot_dim)

        slot_maps = self.relu(self.layer1(slot_maps))                               # (bs, num_slots, num_tokens, hidden_dim)
        slot_maps = self.relu(self.layer2(slot_maps))                               # (bs, num_slots, num_tokens, hidden_dim)
        slot_maps = self.relu(self.layer3(slot_maps))                               # (bs, num_slots, num_tokens, hidden_dim)
        slot_maps = self.relu(self.layer4(slot_maps))                               # (bs, num_slots, num_tokens, hidden_dim)

        feat_rec_w_score = self.dino_layer(slot_maps)                               # (bs, num_slots, num_tokens, token_dim + 1)

        out = {
            "enc_feat_rec": feat_rec_w_score[..., :-1],                             # (bs, num_slots, num_tokens, token_dim + 1)
            "enc_feat_rec_logits": feat_rec_w_score[..., -1:],                      # (bs, num_slots, num_tokens, 1)
        }
        if self.decode_segmentation:
            slot_and_feat = torch.cat([slot_maps, feat_rec_w_score], dim=-1)        # (bs, num_slots, num_tokens, token_dim + 1 + hidden_dim)
            segmentation_logits = self.segment_layer(slot_and_feat)                 # (bs, num_slots, num_tokens, 196)
            out["segmentation_logits"] = segmentation_logits

        return out


class RgbDecoder(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()

        if hasattr(encoder, "decode"):
            self.forward = encoder.decode
        else:
            self.patch_size = args.patch_size
            self.hp, self.wp = args.resize_to[0] // args.patch_size, args.resize_to[1] // args.patch_size

            # === Token calculations ===
            token_dim = args.token_dim
            self.decoder_pos_embed = nn.Parameter(torch.zeros(1, args.token_num, token_dim))

            self.decoder_blocks = nn.ModuleList([
                copy.deepcopy(encoder.model.blocks[0])
                for _ in range(args.decoder_depth)
            ])

            self.decoder_norm = nn.LayerNorm(token_dim)
            self.decoder_pred = nn.Linear(token_dim, args.patch_size ** 2 * 3, bias=True) # decoder to patch

            # reinitialize the transformer weights
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    torch.nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x):
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        x = rearrange(
            x, 'b (hp wp) (p1 p2 c) -> b c (hp p1) (wp p2)',
            hp=self.hp, wp=self.wp, p1=self.patch_size, p2=self.patch_size, c=3,
        )
        return x


class SOLV_SAM(nn.Module):
    def __init__(self, args, encoder):
        super().__init__()

        self.slot_dim = args.slot_dim
        self.slot_num = args.num_slots
        self.patch_size = args.patch_size
        self.token_num = args.token_num
        self.token_dim = args.token_dim

        self.decode_use_mask = args.decode_use_mask
        self.decode_segmentation = args.decode_segmentation

        self.s_bind = Spatial_Binder(args, input_dim=self.token_dim)

        self.dec = Decoder(args)
        self.pos_dec = nn.Parameter(torch.Tensor(1, self.token_num, self.slot_dim))
        init.normal_(self.pos_dec, mean=0., std=.02)

        self.rgb_dec = RgbDecoder(args, encoder)

    def forward(self, frames, mask=None):

        # :arg frames: (bs, num_tokens, token_dim)
        # :arg mask:   (bs, num_slots, num_tokens)

        if mask is not None:
            assert self.slot_num == mask.shape[1]
        assert self.token_dim == frames.shape[-1]

        slots, attn, unweighted_attn = self.get_slots(frames, mask)
        reconstruction = self.decode(
            slots,
            mask=mask if self.decode_use_mask else None,
        )
        reconstruction["attn"] = attn
        reconstruction["unweighted_attn"] = unweighted_attn
        return reconstruction

    def get_slots(self, frames, mask=None):
        # :arg frames: (bs, num_tokens, token_dim)
        # :arg mask: (bs, num_slots, num_tokens)

        # === === Get Slots === ===
        slots, attn, unweighted_attn = self.s_bind(frames, mask)             # (bs, num_slots, slot_dim), (bs, num_slots, num_tokens), (bs, num_slots, num_tokens)
        assert torch.isfinite(slots).all()
        assert torch.isfinite(attn).all()
        assert torch.isfinite(unweighted_attn).all()

        return slots, attn, unweighted_attn

    def decode(self, slots, mask=None, slots_valid=None, decode_rgb=True):
        """
            
        :param slots: (bs, num_slots, slot_dim)
        :param mask: (bs, num_slots, num_tokens)
        :param slots_valid: (bs, num_slots), only used when visualizing dynamics rollouts
        :return: 
        """

        slot_maps = self.sbd_slots(slots)
        assert torch.isfinite(slot_maps).all()

        out = self.dec(slot_maps)

        for k, out_v in out.items():
            assert torch.isfinite(out_v).all()

        reconstruction = self.reconstruct_feature_map(out, mask, slots_valid)
        reconstruction["slots"] = slots
        if decode_rgb:
            reconstruction["rgb_rec"] = self.rgb_dec(reconstruction["enc_feat_rec"])

        return reconstruction

    def sbd_slots(self, slots):
        # :arg slots: (bs, num_slots, slot_dim)

        slots = slots.unsqueeze(dim=-2)                     # (bs, num_slots, 1, slot_dim)
        slots = slots.tile(1, 1, self.token_num, 1)         # (bs, num_slots, num_tokens, slot_dim)

        pos_embed = self.pos_dec.expand_as(slots)
        slots = slots + pos_embed                           # (bs, num_slots, num_tokens, slot_dim)

        return slots

    def reconstruct_feature_map(self, rec, sam_mask=None, slot_valid=None):
        # :arg sam_mask: (bs, num_slots, num_tokens)
        # :arg slot_valid: (bs, num_slots)

        # (bs, num_slots, num_tokens, token_dim), (bs, num_slots, num_tokens, 1), (bs, num_slots, num_tokens, 196)
        enc_feat_rec = rec["enc_feat_rec"]
        enc_feat_rec_logits = rec["enc_feat_rec_logits"]

        # post-process enc_feat_rec_logits and segmentation_logits according to sam_mask and slot_valid
        if sam_mask is not None:
            assert slot_valid is None
            slot_valid = sam_mask.any(dim=-1)                                       # (bs, num_slots)

        if slot_valid is not None:
            has_valislot_dim = slot_valid.any(dim=-1, keepdim=True)
            enc_feat_rec_logits[~slot_valid & has_valislot_dim] = float('-inf')

            if self.decode_segmentation:
                segmentation_logits = rec["segmentation_logits"]
                segmentation_logits[~slot_valid & has_valislot_dim] = float('-inf')
                rec["segmentation_logits"] = segmentation_logits

        # mix enc_feat_rec according to enc_feat_rec_logits
        patch_mask = f.softmax(enc_feat_rec_logits, dim=1)                          # (bs, num_slots, num_tokens, 1)
        enc_feat_rec = torch.sum(enc_feat_rec * patch_mask, dim=1)                  # (bs, num_tokens, token_dim)

        assert torch.isfinite(patch_mask).all()
        assert torch.isfinite(enc_feat_rec).all()
        assert torch.isfinite(patch_mask).all()

        # for loss computation
        token_valid = None

        # for visualization only
        patch_mask = patch_mask.squeeze(dim=-1)                                     # (bs, num_slots, num_tokens)

        if sam_mask is not None:
            token_valid = sam_mask.any(dim=1)                                       # (bs, num_tokens)
            patch_mask = patch_mask.masked_fill(~token_valid.unsqueeze(dim=1).expand(-1, self.slot_num, -1), 0)

        reconstruction_dict = {
            "enc_feat_rec": enc_feat_rec,                                           # (bs, num_tokens, token_dim)
            "token_valid": token_valid,                                             # (bs, num_tokens)
            "patch_mask": patch_mask,                                               # (bs, num_slots, num_tokens)
        }

        if self.decode_segmentation:
            segmentation_logits = rec["segmentation_logits"]
            segmentation = f.softmax(segmentation_logits, dim=1)                    # (bs, num_slots, num_tokens, 196)
            reconstruction_dict["segmentation_logits"] = segmentation_logits        # (bs, num_slots, num_tokens, 196)
            reconstruction_dict["segmentation"] = segmentation                      # (bs, num_slots, num_tokens, 196)

        return reconstruction_dict
