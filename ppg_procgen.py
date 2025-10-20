# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppg/#ppg_procgenpy
import dataclasses
import os
import random
import time
from typing import List, Literal, Optional
from pathlib import Path

try:
    import wandb
    wandb_available = True
except:
    wandb_available = False

import gym
import mlflow
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import tyro
from mlflow.client import MlflowClient
from mlflow.entities import Metric
from mlflow.utils.time import get_current_time_millis
from procgen import ProcgenEnv
from torch import distributions as td
from torch.distributions.categorical import Categorical
from torchvision import transforms as T

from encoder.solv_sam_encoder import SOLV_SAM_Encoder
from oc_utils.layers.transformer_block import Transformer


@dataclasses.dataclass
class EncoderArgs:
    type: Literal["cnn", "cosmos", "oc"] = "cnn"

    # === Misc ===
    checkpoint_path: Optional[str] = None

    # === ViT Related Parameters ===
    resize_to: List[int] = dataclasses.field(default_factory=lambda: [224, 224])
    encoder: Literal[
        "dinov2-vitb-14", "mae-vitb-16",
        "Cosmos-0.1-Tokenizer-CI8x8", "Cosmos-0.1-Tokenizer-CI16x16",
    ] = "Cosmos-0.1-Tokenizer-CI16x16"

    use_sam_mask: bool = False

    # === Slot Attention Related Parameters ===
    num_slots: int = 31
    slot_att_iter: int = 3
    slot_dim: int = 256

    # === Decode Related Parameters ===
    decode_segmentation: bool = False
    decoder_depth: int = 4

    # === Training Related Parameters ===
    finetune: bool = False
    learning_rate: float = 4e-4
    batch_size: int = 32

    # Derived attributes
    patch_size: int = dataclasses.field(init=False)
    token_num: int = dataclasses.field(init=False)

    def __post_init__(self):
        if self.encoder.startswith("Cosmos"):
            self.patch_size = int(self.encoder.split("x")[-1])
            self.token_dim = 16
        else:
            self.patch_size = int(self.encoder.split("-")[2])
            self.token_dim = 768
        self.token_num = (self.resize_to[0] * self.resize_to[1]) // (self.patch_size ** 2)


@dataclasses.dataclass
class Args:
    # === Run related ===
    exp_name: str = "test"

    # Run related
    seed: int = 1
    """seed of the experiment"""
    th_deterministic: bool = True
    """if toggled, `th.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    cuda_id: int = 0
    """the id of the cuda device"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    ckpt_every: Optional[int] = None
    "Save model checkpoints every ckpt_every phases"
    ckpt_start: int = 0
    "Save model checkpoints starting from ckpt_start phase"

    # Environment specific arguments
    env_id: str = "starpilot"
    """the id of the environment"""
    num_envs: int = 64
    """the number of parallel game environments"""
    num_levels: int = 0
    """the number of levels used, 0 means all levels"""
    start_level: int = 0
    """the starting level id, all level ids will be no smaller than this"""
    distribution_mode: Literal["easy", "hard"] = "easy"
    """difficulty of the game"""
    use_backgrounds: bool = False
    """whether to use background images"""

    # Algorithm specific arguments
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 5e-4
    """the learning rate of the optimizer"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.999
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    adv_norm_fullbatch: bool = True
    """Toggle full batch advantage normalization as used in PPG code"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # PPG specific arguments
    n_iteration: int = 32
    """N_pi: the number of policy update in the policy phase """
    e_policy: int = 1
    """E_pi: the number of policy update in the policy phase """
    v_value: int = 1
    """E_V: the number of policy update in the policy phase """
    e_auxiliary: int = 6
    """E_aux:the K epochs to update the policy"""
    beta_clone: float = 1.0
    """the behavior cloning coefficient"""
    num_aux_rollouts: int = 4
    """the number of mini batch in the auxiliary phase"""
    n_aux_grad_accum: int = 1
    """the number of gradient accumulation in mini batch"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_phases: int = 0
    """the number of phases (computed in runtime)"""
    aux_batch_rollouts: int = 0
    """the number of rollouts in the auxiliary phase (computed in runtime)"""

    encoder: EncoderArgs = dataclasses.field(default_factory=EncoderArgs)


def layer_init_normed(layer, norm_dim, scale=1.0):
    with th.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = th.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert th.equal(a, c)


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_th.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale)
        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        conv = nn.Conv2d(
            in_channels=self._input_shape[0],
            out_channels=self._out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0)
        nblocks = 2  # Set to the number of residual blocks
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        self.args = args

        h, w, c = envs.single_observation_space.shape

        if args.encoder.type == "cnn":
            shape = (c, h, w)
            conv_seqs = []
            chans = [16, 32, 32]
            scale = 1 / np.sqrt(
                len(chans)
            )  # Not fully sure about the logic behind this but its used in PPG code
            for out_channels in chans:
                conv_seq = ConvSequence(shape, out_channels, scale=scale)
                shape = conv_seq.get_output_shape()
                conv_seqs.append(conv_seq)

            encodertop = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
            encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4)
            conv_seqs += [
                nn.Flatten(),
                nn.ReLU(),
                encodertop,
                nn.ReLU(),
            ]
            self.network = nn.Sequential(*conv_seqs)
        elif args.encoder.type in ["cosmos", "oc"]:
            self.encoder = SOLV_SAM_Encoder(args)

            d_model = 256
            d_feat = args.encoder.token_dim if args.encoder.type == "cosmos" else args.encoder.slot_dim
            if d_feat != d_model:
                self.proj = nn.Linear(d_feat, d_model)
            else:
                self.proj = nn.Identity()

            # actor and critic embeddings
            self.actor_critic_token = nn.Parameter(th.zeros(1, 1, d_model))
            nn.init.normal_(self.actor_critic_token, std=1e-6)
            self.actor_critic_mha = Transformer(
                d_model=d_model,
                num_blocks=1,
                num_heads=8,
            )
        else:
            raise ValueError(f"Invalid encoder type: {args.encoder.type}")

        self.actor = layer_init_normed(
            nn.Linear(256, envs.single_action_space.n), norm_dim=1, scale=0.1
        )
        self.critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)
        self.aux_critic = layer_init_normed(nn.Linear(256, 1), norm_dim=1, scale=0.1)

    def get_feature(self, x):
        if self.args.encoder.type == "cnn":
            return self.network(x.permute((0, 3, 1, 2)) / 255.0)
        elif self.args.encoder.type in ["cosmos", "oc"]:

            with th.no_grad():
                x = self.encoder.forward_episode(x, modes="enc_feat" if self.args.encoder.type == "cosmos" else "slots")

            x = self.proj(x)                                                        # (bs, num_tokens, d_model)
            x = th.cat([
                self.actor_critic_token.expand(x.shape[0], -1, -1),
                x,
            ], dim=1)                                                               # (bs, num_tokens + 1, d_model)
            x = self.actor_critic_mha(x)                                            # (bs, num_tokens + 1, d_model)
            x = x[:, 0, :]                                                          # (bs, d_model)

            return x

    def get_action_and_value(self, x, action=None):
        hidden = self.get_feature(x)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden.detach())

    def get_value(self, x):
        return self.critic(self.get_feature(x))  # "bhwc" -> "bchw"

    # PPG logic:
    def get_pi_value_and_aux_value(self, x):
        hidden = self.get_feature(x)
        return (
            Categorical(logits=self.actor(hidden)),
            self.critic(hidden.detach()),
            self.aux_critic(hidden),
        )

    def get_pi(self, x):
        return Categorical(logits=self.actor(self.get_feature(x)))


def train(args):
    # Compute runtime args
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)
    args.num_phases = int(args.num_iterations // args.n_iteration)
    args.aux_batch_rollouts = int(args.num_envs * args.n_iteration)
    assert args.v_value == 1, "Multiple value epoch (v_value != 1) is not supported yet"

    # Log the hyperparameters
    if not wandb_available:
        mlflow.log_params(dataclasses.asdict(args))

    # Set up ckpt folder (in case we want to run on 2 games in a single job)
    ckpt_dir = os.path.join(os.environ.get("AMLT_OUTPUT_DIR", args.logging_path), args.env_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    flatten_unflatten_test()  # Try not to mess with the flatten unflatten logic

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.th_deterministic

    device = th.device(f"cuda:{args.cuda_id}" if th.cuda.is_available() and args.cuda else "cpu")
    args.dtype = th.float32
    args.device = device

    # Env setup
    envs = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        use_backgrounds=args.use_backgrounds,
    )
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space["rgb"]
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    agent = Agent(envs, args).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)

    # ALGO Logic: Storage setup
    obs = th.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = th.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = th.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = th.zeros((args.num_steps, args.num_envs)).to(device)
    dones = th.zeros((args.num_steps, args.num_envs)).to(device)
    values = th.zeros((args.num_steps, args.num_envs)).to(device)
    aux_obs = th.zeros(
        (args.num_steps, args.aux_batch_rollouts) + envs.single_observation_space.shape,
        dtype=th.uint8,
    )  # Saves lot system RAM
    aux_returns = th.zeros((args.num_steps, args.aux_batch_rollouts))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs = th.Tensor(envs.reset()).to(device)
    next_done = th.zeros(args.num_envs).to(device)

    metrics = []  # Store the metrics temporarily to reduce mlflow overhead
    for phase in range(1, args.num_phases + 1):
        start_time = time.time()
        # POLICY PHASE
        for update in range(1, args.n_iteration + 1):
            print(f"--- Phase {phase}/{args.num_phases}, update {update}/{args.n_iteration} ---")
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with th.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = envs.step(action.cpu().numpy())
                rewards[step] = th.tensor(reward).to(device).view(-1)
                next_obs, next_done = th.Tensor(next_obs).to(device), th.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        if wandb_available:
                            episode_metrics = {
                                "episodic_return": item["episode"]["r"],
                                "episodic_length": item["episode"]["l"],
                            }
                            wandb.define_metric("step")
                            for k in episode_metrics.keys():
                                wandb.define_metric(k, step_metric="step")

                            wandb.log({**episode_metrics, "step": global_step})
                        else:
                            ts = get_current_time_millis()
                            metrics.append(
                                Metric("episodic_return", item["episode"]["r"], ts, global_step)
                            )
                            metrics.append(
                                Metric("episodic_length", item["episode"]["l"], ts, global_step)
                            )
                        break

            # Bootstrap value if not done (GAE calculation)
            with th.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = th.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # Flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # PPG code does full batch advantage normalization
            if args.adv_norm_fullbatch:
                b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.e_policy):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with th.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * th.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = th.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + th.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            training_metrics = {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": v_loss.item(),
                "policy_loss": pg_loss.item(),
                "entropy": entropy_loss.item(),
                "old_approx_kl": old_approx_kl.item(),
                "approx_kl": approx_kl.item(),
                "clipfrac": np.mean(clipfracs),
                "explained_variance": explained_var,
            }
            if wandb_available:
                wandb.define_metric("step")
                for k in training_metrics.keys():
                    wandb.define_metric(k, step_metric="step")

                wandb.log({**training_metrics, "step": global_step})
            else:
                ts = get_current_time_millis()
                for k, v in training_metrics.items():
                    metrics.append(Metric(k, v, ts, global_step))
                MlflowClient().log_batch(mlflow.active_run().info.run_id, metrics)
                metrics = []

            # PPG Storage - Rollouts are saved without flattening for sampling full rollouts later:
            storage_slice = slice(args.num_envs * (update - 1), args.num_envs * update)
            aux_obs[:, storage_slice] = obs.cpu().clone().to(th.uint8)
            aux_returns[:, storage_slice] = returns.cpu().clone()

        # AUXILIARY PHASE
        aux_inds = np.arange(args.aux_batch_rollouts)

        # Build the old policy on the aux buffer before distilling to the network
        aux_pi = th.zeros((args.num_steps, args.aux_batch_rollouts, envs.single_action_space.n))
        for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
            end = start + args.num_aux_rollouts
            aux_minibatch_ind = aux_inds[start:end]
            m_aux_obs = aux_obs[:, aux_minibatch_ind].to(th.float32).to(device)
            m_obs_shape = m_aux_obs.shape
            m_aux_obs = flatten01(m_aux_obs)
            with th.no_grad():
                pi_logits = agent.get_pi(m_aux_obs).logits.cpu().clone()
            aux_pi[:, aux_minibatch_ind] = unflatten01(pi_logits, m_obs_shape[:2])
            del m_aux_obs

        for auxiliary_update in range(1, args.e_auxiliary + 1):
            print(f"aux epoch {auxiliary_update}")
            np.random.shuffle(aux_inds)
            for i, start in enumerate(range(0, args.aux_batch_rollouts, args.num_aux_rollouts)):
                end = start + args.num_aux_rollouts
                aux_minibatch_ind = aux_inds[start:end]
                try:
                    m_aux_obs = aux_obs[:, aux_minibatch_ind].to(device)
                    m_obs_shape = m_aux_obs.shape
                    # Sample full rollouts for PPG instead of random indexes
                    m_aux_obs = flatten01(m_aux_obs)
                    m_aux_returns = aux_returns[:, aux_minibatch_ind].to(th.float32).to(device)
                    m_aux_returns = flatten01(m_aux_returns)

                    new_pi, new_values, new_aux_values = agent.get_pi_value_and_aux_value(m_aux_obs)

                    new_values = new_values.view(-1)
                    new_aux_values = new_aux_values.view(-1)
                    old_pi_logits = flatten01(aux_pi[:, aux_minibatch_ind]).to(device)
                    old_pi = Categorical(logits=old_pi_logits)
                    kl_loss = td.kl_divergence(old_pi, new_pi).mean()

                    real_value_loss = 0.5 * ((new_values - m_aux_returns) ** 2).mean()
                    aux_value_loss = 0.5 * ((new_aux_values - m_aux_returns) ** 2).mean()
                    joint_loss = aux_value_loss + args.beta_clone * kl_loss

                    loss = (joint_loss + real_value_loss) / args.n_aux_grad_accum
                    loss.backward()

                    if (i + 1) % args.n_aux_grad_accum == 0:
                        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()  # This cannot be outside, else gradients won't accumulate

                except RuntimeError as e:
                    raise Exception(
                        "if running out of CUDA memory, try a higher --n-aux-grad-accum, which trades more time for less gpu memory"
                    ) from e

                del m_aux_obs, m_aux_returns

        aux_metrics = {
            "aux/kl_loss": kl_loss.mean().item(),
            "aux/aux_value_loss": aux_value_loss.item(),
            "aux/real_value_loss": real_value_loss.item(),
            "SPS": int(global_step / (time.time() - start_time)),
        }
        if wandb_available:
            wandb.define_metric("step")
            for k in aux_metrics.keys():
                wandb.define_metric(k, step_metric="step")
            wandb.log({**aux_metrics, "step": global_step})
        else:
            ts = get_current_time_millis()
            for k, v in aux_metrics.items():
                mlflow.log_metric(k, v, global_step)

        # Save model checkpoints
        if (
            args.ckpt_every is not None
            and phase % args.ckpt_every == 0
            and phase >= args.ckpt_start
        ):
            th.save(agent.state_dict(), os.path.join(ckpt_dir, f"agent@phase{phase}.pt"))

    envs.close()

    # Save final model
    th.save(agent.state_dict(), os.path.join(ckpt_dir, "agent@final.pt"))


if __name__ == "__main__":
    args = tyro.cli(Args)

    if wandb_available:
        run_name = f"{args.env_id}_{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="ppg_procgen",
            name=run_name,
            config=dataclasses.asdict(args),
            dir=Path("wandb").resolve(),
            mode="online" if (args.exp_name != "test") else "disabled",
        )
        args.logging_path = Path("results") / args.exp_name / run_name
        train(args)
    elif os.environ.get("AMLT_JOB_NAME", None) is None:
        logging_path = Path("mlruns").resolve()
        mlflow.set_tracking_uri(logging_path)
        experiment = mlflow.set_experiment(experiment_name="ppg_procgen")
        run_name = f"{args.env_id}_{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            args.logging_path = logging_path / experiment.experiment_id / run.info.run_id
            train(args)
    else:
        mlflow.enable_system_metrics_logging()  # Enable system metrics logging on Azure
        mlflow.set_system_metrics_samples_before_logging(12)  # Log every 12*10=120 seconds
        args.logging_path = None
        train(args)
