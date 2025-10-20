from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Union, List, Optional


@dataclass
class LinearSchedule:
    start_step: int
    end_step: int
    start_value: float = 0.0
    end_value: float = 1.0

    def __post_init__(self):
        assert self.start_step <= self.end_step

    def get_value(self, step):
        if self.start_step == self.end_step:
            return self.start_value if step < self.start_step else self.end_value

        ratio = (step - self.start_step) / (self.end_step - self.start_step)
        ratio = min(max(ratio, 0.0), 1.0)
        return self.start_value + ratio * (self.end_value - self.start_value)


@dataclass
class SsmConfig:
    d_state: int = 64
    d_conv: int = 4


@dataclass
class MHAConfig:
    num_blocks: int = 2
    num_heads: int = 8


@dataclass
class Mamba:
    d_model: int = 512
    d_intermediate: int = 0
    n_layer: int = 2
    layer: Literal["Mamba1", "Mamba2", "MHA"] = "Mamba2"
    ssm_cfg: SsmConfig = field(default_factory=SsmConfig)
    attn_cfg: Dict[str, Any] = field(default_factory=lambda: {"num_heads": 8})
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True


@dataclass
class Sinkhorn:
    num_iters: int = 10
    epsilon: float = 0.1


@dataclass
class SlotPred:
    use_assignment: bool = False
    soft_assignment_method: Literal[
        "vanilla",      # using prediction error to assign predicted slots to ground truth slots
        "sinkhorn",     # further using sinkhorn algorithm to refine the assignment
    ] = "sinkhorn"
    hard_assignment_method: Literal[
        "greedy",       # for each ground truth slot, assign the prediction with the smallest error
    ] = "greedy"
    sinkhorn_cfg: Sinkhorn = field(default_factory=Sinkhorn)


@dataclass
class Pred:
    mlp_use_norm: bool | None = True

    # pre-processing part
    # proj_in_mlp_dims slot -> ssm input
    proj_in_mlp_dims: List[int] = field(default_factory=lambda: [512])
    pre_ssm_mixer_cfg: MHAConfig = field(default_factory=lambda: MHAConfig(num_blocks=1))

    # slot prediction part
    # if True, each slot uses a separate mamba
    num_slot_modes: int = 1
    pred_delta: bool | None = True
    slot_pred: SlotPred = field(default_factory=SlotPred)
    pred_mlp_dims: List[int] = field(default_factory=lambda: [512])

    # reward and termination prediction part
    num_reward_bins: int = 255
    pred_mha_cfg: MHAConfig = field(default_factory=lambda: MHAConfig(num_blocks=1))

    # ==================== Static-Dynamic Disentanglement ====================
    # whether to disentangle the static and dynamic subparts of the slot
    disentangle_static_dynamic: bool | None = False
    static_dynamic_merge_method: Literal["concat", "add", "dynamic_only"] = "concat"
    normalize_dynamic_embeddings: bool | None = False

    dynamic_use_vq: bool | None = False
    dynamic_vq_type: Literal["vq", "sim_vq", "fsq"] = "vq"
    codebook_size: int = 16384
    codebook_levels: List[int] = field(default_factory=lambda: [16, 16, 16, 16])
    codebook_transform_mlp_multi: Optional[List[int]] = None

    # project slot to static and dynamic subparts
    proj_mlp_dims: List[int] = field(default_factory=lambda: [1024, 1024])
    # reconstruct slot from the static and dynamic subparts
    recon_mlp_dims: List[int] = field(default_factory=lambda: [1024, 1024])

    # dynamic subpart + static subpart -> 0/1: to minimize mutual information between the static and dynamic subparts
    dynamic_to_static_discriminator_mlp_dims: List[int] = field(default_factory=lambda: [512])

    """
    Training behavior:
    - predict slot existence and visibility and use them as inputs
    if self.slot_pred.use_assignment is True:
        - use dynamics assignment to assign predicted slots to ground truth slots
        if pred_appearing_slots is True:
            - predict the appearing slots and use dynamics assignment to assign them to ground truth slots
            if pred_appearing_slots_exist_visible is True:
                - predict the appearing slots that do not exist at t but exist at t+1
                if pred_appearing_slots_exist_visible is True:
                    - predict the visibility and existence of the appearing slots
                else:
                    - do not predict the visibility and existence
        else:
            - only predict the slots that exist at t
    else:
        - do not use dynamics assignment, assume predictions are in the same order as ground truth slots
    """
    pred_appearing_slots: bool = False
    pred_appearing_slots_exist_visible: bool = False

    def __post_init__(self):
        if self.pred_appearing_slots:
            assert self.slot_pred.use_assignment
        if self.pred_appearing_slots_exist_visible:
            assert self.pred_appearing_slots


@dataclass
class Training:
    discriminator_lr: float = 1e-4
    static_lr: float = 1e-4
    dynamic_lr: float = 3e-4
    dynamic_disentanglement_lr: float = 1e-4
    prediction_lr: float = 1e-4

    grad_norm_clip: float = 1.0
    batch_size: int = 64
    pred_seq_len: int = 10
    disentangle_seq_len: int = 3
    termination_prob: float = -1.0

    n_pred_steps: int = 5
    deterministic: bool = True
    loss_only_use_last_pred: bool | None = False

    update_pred: bool | None = False
    use_pretrained_dynamics: bool | None = False

    slot_loss_fn: Literal["l1", "mse"] = "mse"

    use_reward_termination_weight: bool | None = False
    # when gamma = 0, it is equivalent to no focal loss
    reward_focal_loss_gamma: float = 0.0
    termination_focal_loss_gamma: float = 2.0

    pred_enc_feat: bool | None = True
    pred_obs: bool | None = True
    pred_ratio: float = 0.1

    valid_freq: int = 1000

    # ==================== Static-Dynamic Disentanglement ====================
    discriminator_type: Literal["logistic", "wasserstein"] = "wasserstein"
    discriminator_regularization_type: Literal["gradient_penalty", "lecam"] = "lecam"
    discriminator_gradient_penalty_coef: float = 10.0
    discriminator_lecam_coef: float = 0.05

    n_discriminator_steps: int = 1
    n_static_steps: int = 1
    n_dynamic_steps: int = 1
    n_dynamic_disentanglement_steps: int = 1

    proj_in_weight_decay: float = 0.0

    contrastive_temperature: float = 0.1
    contrastive_num_negatives: int = 256

    static_contrastive_coef: float = 1.0
    static_invariance_coef: float = 0.0

    # dynamic subpart + static subpart -> 0/1: to minimize mutual information between the static and dynamic subparts
    dynamic_contrastive_threshold: float = 0.9
    dynamic_contrastive_coef: float = 0.0

    dynamic_recon_dropout: float = 0.0

    # vq
    commit_weight: float = 1.0
    input_to_quantize_commit_loss_weight: float = 0.25
    vq_ema_update: bool | None = False

    dynamic_adversarial_loss_weight_schedule: LinearSchedule = field(
        default_factory=lambda: LinearSchedule(
            start_step=10000,
            end_step=50000,
            start_value=0.0,
            end_value=1.0
        )
    )

    iterative_dynamic_training: bool | None = True

    use_ca_grad: bool | None = False
    ca_grad_alpha: float = 0.05
    ca_grad_rescale: bool | None = True


@dataclass
class DataLoading:
    num_workers: int = 0
    prefetch_factor: Union[int, None] = None
    pin_memory: bool = True


@dataclass
class Rollout:
    n_kmeans_clusters: int = 32
    n_warmup_steps: int = 10
    deterministic: bool = True
    cg: bool = False


@dataclass
class MambaConfig:
    patch_as_slot: bool | None = False
    num_slots: int = 31
    checkpoint_path: Union[str, None] = None

    mamba: Mamba = field(default_factory=Mamba)
    pred: Pred = field(default_factory=Pred)
    training: Training = field(default_factory=Training)
    data_loading: DataLoading = field(default_factory=DataLoading)
    rollout: Rollout = field(default_factory=Rollout)

    def __post_init__(self):
        if self.patch_as_slot:
            # num_slots will be overridden by the number of patches in the input image
            if self.pred.disentangle_static_dynamic:
                self.pred.disentangle_static_dynamic = False
                print("using patch-level encoder feature as slot does not support static-dynamic disentanglement."
                      "overriding it to False.")

        assert not (self.training.use_ca_grad and self.training.iterative_dynamic_training), \
            "Cannot use CA grad and iterative dynamic training at the same time."
