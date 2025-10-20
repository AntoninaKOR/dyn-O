import os
import time
import warnings

# Suppress pytorch FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import wandb
    wandb_available = True
except:
    wandb_available = False

import dataclasses
import tyro
import mlflow
from PIL import Image
import tempfile

import gym
import random
import numpy as np

from tqdm import tqdm
from pathlib import Path
from procgen import ProcgenEnv

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.set_printoptions(precision=3, sci_mode=False)


from configs.config import Config
from encoder import encoder_library
from dynamics import dynamics_library
from policy import policy_library
from trainer.trainer import Trainer
from trainer.world_model_env import WorldModelObs
from oc_utils.replay_buffer import replay_buffer_library
from oc_utils.dtype import torch_to_numpy_dtype_dict, torch_dtype_dict
from oc_utils.visualize_rollout import visualize_rollout
from oc_utils.dist import init_distributed_mode
from oc_utils.utils import log_params


mlflow.config.enable_async_logging(enable=True)


def set_remaining_online_config(config, env):
    # ================== MISC ==================
    config.device = torch.device(f"cuda:{config.gpu}" if torch.cuda.is_available() else "cpu")
    config.dtype = torch_dtype_dict[config.dtype_name]
    config.np_dtype = torch_to_numpy_dtype_dict[config.dtype]
    torch.set_default_dtype(config.dtype)

    # ================== FOR and FROM ENV ==================
    # decide whether env use stack frame wrapper from encoder config
    config.action_space = env.action_space
    config.observation_space = env.observation_space

    return config


def set_remaining_offline_config(config, rb):
    # ================== FROM OFFLINE BUFFER ==================
    # overwrite the observation space and action space with the ones loaded by the offline buffer
    config.action_space = rb.action_space
    config.observation_space = rb.observation_space


def init_env(config, train=True):
    envs = ProcgenEnv(
        num_envs=config.env.num_train_envs if train else config.env.num_eval_envs,
        num_levels=200 if train else 0,
        start_level=0 if train else 250,
        env_name=config.env.env_id,
        distribution_mode=config.env.distribution_mode,
        use_backgrounds=config.env.use_backgrounds,
        rand_seed=config.seed,
    )
    envs.is_vector_env = True
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"])
    envs.observation_space = envs.observation_space["rgb"]
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    set_remaining_online_config(config, envs)

    record_episode_statistics_kwargs = {}
    if not train:
        record_episode_statistics_kwargs["deque_size"] = config.env.num_eval_envs
    envs = gym.wrappers.RecordEpisodeStatistics(envs, **record_episode_statistics_kwargs)

    # initialize RecordEpisodeStatistics wrapper
    envs.reset()

    return envs


def main(config: Config):
    if not wandb_available:
        log_params(config)

    # TRY NOT TO MODIFY: seeding
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = config.torch_deterministic

    # ================== ENV (will not be used if offline training) ==================
    train_envs = init_env(config, train=True)
    eval_envs = init_env(config, train=False)

    # ================== MODEL and REPLAY BUFFER ==================
    encoder = encoder_library[config.encoder_type](config).to(config.device).eval()

    rb = replay_buffer_library[config.replay_buffer_type](config, encoder=encoder)
    if config.offline:
        set_remaining_offline_config(config, rb)

    dynamics = dynamics_library[config.dynamics_type](config, encoder=encoder).to(config.device)

    if config.distributed:
        if config.encoder.finetune:
            encoder = DDP(encoder, device_ids=[config.gpu], find_unused_parameters=False)
        dynamics = DDP(dynamics, device_ids=[config.gpu], find_unused_parameters=True)

    policy = None
    if config.update_policy or config.eval_policy:
        policy = policy_library[config.policy_type](config).to(config.device)
        if config.distributed:
            policy = DDP(policy, device_ids=[config.gpu], find_unused_parameters=False)

    trainer = Trainer(config, encoder, dynamics, policy, rb)

    # start_step = 1 if no checkpoint is found in config.ckpt_path
    if config.ckpt_path is not None:
        config.ckpt_path.mkdir(parents=True, exist_ok=True)
    start_step = trainer.load_last_checkpoint()

    last_save_time = time.time()
    for global_step in tqdm(
        range(start_step, config.total_timesteps + 1),
        initial=start_step,
        total=config.total_timesteps,
        disable=config.rank != 0,
    ):
        num_digits = len(str(config.total_timesteps))
        global_step_str = f"{global_step:0{num_digits}d}"

        # ALGO LOGIC: training.
        learning_starts = global_step > config.learning_starts or config.offline
        if not learning_starts:
            continue

        # dynamics training
        if config.update_dynamics:
            trainer.update_dynamics(global_step)

            if global_step % config.dynamics.training.valid_freq == 0:
                for split in ["train", "valid"]:
                    visualize = global_step % config.logging.viz_rollout_per_timestep == 0
                    rollout_kwargs = {
                        "rollout": {
                            "rollout": True,
                            "random_static": False,
                        },
                    }
                    if visualize:
                        rollout_kwargs["one_step"] = {
                            "rollout": False,
                        }
                        if config.dynamics.pred.disentangle_static_dynamic:
                            rollout_kwargs["rollout_random_all"] = {
                                "rollout": True,
                                "random_static": True,
                            }
                            rollout_kwargs["rollout_random_one"] = {
                                "rollout": True,
                                "random_static": True,
                                "random_one_slot": True,
                            }

                    inputs, predictions = trainer.rollout_dynamics(split, global_step, rollout_kwargs)

                    if not visualize:
                        continue

                    # inputs, predictions: dict of str -> prediction with different kwargs
                    for postfix in rollout_kwargs:
                        images = visualize_rollout(config, encoder, inputs[postfix], predictions[postfix])
                        if config.rank != 0:
                            continue
                        for i, image in enumerate(images):
                            if wandb_available:
                                image_name = f"viz_{split}_{postfix}/sample_{i}"
                                wandb.define_metric("step")
                                wandb.define_metric(image_name, step_metric="step")
                                wandb.log({image_name: wandb.Image(image), "step": global_step})
                            else:
                                fname = f"viz_{split}_{postfix}/step_{global_step_str}_{i}.png"
                                try:
                                    mlflow.log_image(image, fname, synchronous=False)
                                except:
                                    print(f"Failed to log image {fname} to mlflow")

        # policy training
        if (
            config.update_policy and
            global_step > config.policy_learning_starts and
            global_step % config.policy.update_frequency == 0
        ):
            trainer.update_policy(global_step)

        # policy evaluation
        if (
            (config.update_policy and global_step % config.logging.eval_policy_per_timestep == 0) or
            (not config.update_policy and config.eval_policy)
        ):
            for env_prefix, envs in zip(
                ["train", "eval"],
                [train_envs, eval_envs]
            ):
                # Force-reset Procgen by using action == -1
                # https://github.com/openai/procgen/blob/master/procgen/src/game.cpp#L124-L127
                eval_obs, _, _, _ = envs.step(np.array([-1] * envs.num_envs))

                num_of_gifs = min(config.logging.num_of_gifs, len(eval_obs))
                frames = [[] for _ in range(num_of_gifs)]

                episode_stats = []
                while len(episode_stats) < envs.num_envs:
                # ALGO LOGIC: put action logic here
                    with torch.inference_mode():
                        if config.policy_type == "actor_critic":
                            mode = config.policy.actor_critic_model.input_mode
                        elif config.policy_type == "bc":
                            mode = config.policy.model.input_mode
                        else:
                            raise ValueError(f"Unsupported policy type: {config.policy_type}")

                        assert config.encoder_type == "solv_sam"

                        encoder_kwargs = {"modes": mode}
                        eval_obs = torch.tensor(eval_obs, device=config.device, dtype=config.dtype)
                        encoding = encoder(eval_obs, **encoder_kwargs)

                        eval_obs = WorldModelObs(**{mode: encoding})
                        policy.eval()
                        actions = policy(eval_obs).cpu().numpy()

                    next_eval_obs, rewards, dones, infos = envs.step(actions)
                    eval_obs = np.array(next_eval_obs)
                    for item in infos:
                        if "episode" in item.keys():
                            episode_stats.append(item["episode"])

                    # gif logging
                    for i in range(num_of_gifs):
                        frames[i].append(eval_obs[i])
                        if dones[i]:
                            imgs = [Image.fromarray(frame) for frame in frames[i]]
                            frames[i] = []

                            if config.rank == 0:
                                with tempfile.TemporaryDirectory() as tmp_dir:
                                    gif_fname = f"step_{global_step_str}_epi_{i}.gif"
                                    tmp_gif_path = Path(tmp_dir) / gif_fname
                                    try:
                                        imgs[0].save(
                                            tmp_gif_path,
                                            save_all=True,
                                            append_images=imgs[1:],
                                            duration=int(1000 / 15),        # 15 FPS
                                            loop=0,
                                        )
                                        mlflow.log_artifact(str(tmp_gif_path), f"policy_{env_prefix}")
                                    except:
                                        print(f"Failed to log gif {gif_fname} to mlflow")

                if config.rank == 0:
                    mlflow.log_metric(
                        f"policy_{env_prefix}/episodic_return",
                        np.mean([item["r"] for item in episode_stats]),
                        global_step,
                        synchronous=False,
                    )
                    mlflow.log_metric(
                        f"policy_{env_prefix}/episodic_length",
                        np.mean([item["l"] for item in episode_stats]),
                        global_step,
                        synchronous=False,
                    )

        if config.rank == 0:
            # Save model checkpoints
            if global_step % config.logging.ckpt_per_timestep == 0:
                if config.update_dynamics:
                    torch.save(trainer.dynamics_save_dict(global_step), config.ckpt_path / f"dynamics_{global_step_str}.pt")
                if config.update_policy:
                    torch.save(trainer.policy_save_dict(global_step), config.ckpt_path / f"policy_{global_step_str}.pt")

            if time.time() - last_save_time > config.logging.ckpt_per_second:
                if config.update_dynamics:
                    torch.save(trainer.dynamics_save_dict(global_step), config.ckpt_path / f"dynamics_latest.pt")
                if config.update_policy:
                    torch.save(trainer.policy_save_dict(global_step), config.ckpt_path / f"policy_latest.pt")
                last_save_time = time.time()

    train_envs.close()
    eval_envs.close()
    if config.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.distributed:
        init_distributed_mode(config)
    else:
        config.rank, config.gpu, config.world_size = 0, config.cuda_id, 1
        torch.cuda.set_device(config.cuda_id)

    if wandb_available:

        config_dict = dataclasses.asdict(config)
        config_dict = {k: v for k, v in config_dict.items() if not k.endswith("_all")}

        run_name = f"{config.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"

        wandb.init(
            project=config.exp_name,
            name=run_name,
            config=config_dict,
            dir=Path("wandb").resolve(),
            mode="online" if (config.rank == 0 and config.exp_name != "test") else "disabled",
        )
        config.ckpt_path = Path("results").resolve() / config.exp_name / run_name
        main(config)

    else:
        if os.environ.get("AMLT_JOB_NAME", None) is None:
            # Local running
            if config.rank == 0:
                config.logging_path = Path("mlruns").resolve()
                mlflow.set_tracking_uri(config.logging_path)

                experiment = mlflow.set_experiment(config.exp_name)
                run_name = f"{config.run_name}_seed_{config.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
                with mlflow.start_run(run_name=run_name) as run:
                    mlflow_run_path = config.logging_path / experiment.experiment_id / run.info.run_id
                    config.ckpt_path = mlflow_run_path / "artifacts"

                    # create a human-readable symlink for the run
                    symlink_path = Path("results").resolve() / config.exp_name / run_name
                    symlink_path.parent.mkdir(parents=True, exist_ok=True)
                    symlink_path.symlink_to(mlflow_run_path, target_is_directory=True)

                    main(config)
            else:
                config.ckpt_path = None
                main(config)
        else:
            # AzureML running, need to prepare for getting killed and restarted
            run_name = f"{config.run_name}"
            config.ckpt_path = Path("/mnt/default/oc_ssm") / config.exp_name / run_name
            mlflow.enable_system_metrics_logging()                  # Enable system metrics logging on Azure
            mlflow.set_system_metrics_samples_before_logging(12)    # Log every 12*10=120 seconds
            main(config)
