from __future__ import annotations

import collections
import dataclasses
import logging
import os
import pathlib
import random
import secrets
import shutil
from typing import Any, Iterable, List, Literal, SupportsFloat, Type
from unittest import mock

import gym3
import gymnasium as gym
import h5py
import minari
import mlflow
import numpy as np
import procgen
import torch as th
import tyro
from minari.data_collector.callbacks import EpisodeMetadataCallback
from minari.data_collector.data_collector import AUTOSEED_BIT_SIZE
from minari.data_collector.episode_buffer import EpisodeBuffer
from minari.dataset._storages.hdf5_storage import _get_from_h5py
from minari.dataset.step_data import StepData
from numpy.typing import NDArray
from torch.distributions.categorical import Categorical

from ppg_procgen import Agent, EncoderArgs


CHUNK_SIZE = 1


def vt2space(vt: gym3.types.ValType):

    def tt2space(tt: gym3.types.TensorType):
        if isinstance(tt.eltype, gym3.types.Discrete):
            if tt.ndim == 0:
                return gym.spaces.Discrete(tt.eltype.n)
            else:
                return gym.spaces.Box(
                    low=0,
                    high=tt.eltype.n - 1,
                    shape=tt.shape,
                    dtype=gym3.types_np.dtype(tt),
                )
        elif isinstance(tt.eltype, gym3.types.Real):
            return gym.spaces.Box(
                shape=tt.shape,
                dtype=gym3.types_np.dtype(tt),
                low=float("-inf"),
                high=float("inf"),
            )
        else:
            raise NotImplementedError

    space = gym3.types.multimap(tt2space, vt)

    def dict2dict_space(d):
        if isinstance(d, dict):
            return gym.spaces.Dict({k: dict2dict_space(v) for k, v in d.items()})
        else:
            return d

    return dict2dict_space(space)


def _add_episode_to_group(episode_buffer: dict, episode_group: h5py.Group):
    for key, data in episode_buffer.items():
        if isinstance(data, dict):
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(data, episode_group_to_clear)
        elif isinstance(data, tuple):
            dict_data = {f"_index_{i}": subdata for i, subdata in enumerate(data)}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)
        elif isinstance(data, List) and all(
            isinstance(entry, collections.OrderedDict) for entry in data
        ):  # list of OrderedDict
            dict_data = {key: [entry[key] for entry in data] for key in data[0].keys()}
            episode_group_to_clear = _get_from_h5py(episode_group, key)
            _add_episode_to_group(dict_data, episode_group_to_clear)

        # leaf data
        elif key in episode_group:
            dataset = episode_group[key]
            assert isinstance(dataset, h5py.Dataset)
            dataset.resize((dataset.shape[0] + len(data), *dataset.shape[1:]))
            dataset[-len(data) :] = data
        elif not isinstance(data, Iterable):
            if data is not None:
                episode_group.create_dataset(key, data=data)
        else:
            dtype = None
            if all(map(lambda elem: isinstance(elem, str), data)):
                dtype = h5py.string_dtype(encoding="utf-8")
            dshape = ()
            if hasattr(data[0], "shape"):
                dshape = data[0].shape

            episode_group.create_dataset(
                key,
                data=data,
                dtype=dtype,
                chunks=(CHUNK_SIZE, *dshape) if key == "observations" else True,
                maxshape=(None, *dshape),
                compression="gzip",
                shuffle=True,
            )


class ProcgenGymnasiumVectorEnv(gym.vector.VectorEnv):
    def __init__(
        self,
        num_envs: int,
        env_name: str,
        num_levels: int = 0,
        start_level: int = 0,
        center_agent: bool = True,
        use_backgrounds: bool = True,
        use_monochrome_assets: bool = False,
        restrict_themes: bool = False,
        use_generated_assets: bool = False,
        paint_vel_info: bool = False,
        distribution_mode: Literal["hard", "easy", "exploration", "extreme", "memory"] = "hard",
        **kwargs,
    ):
        self.procgen_env = procgen.ProcgenGym3Env(
            num=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            center_agent=center_agent,
            use_backgrounds=use_backgrounds,
            use_monochrome_assets=use_monochrome_assets,
            restrict_themes=restrict_themes,
            use_generated_assets=use_generated_assets,
            paint_vel_info=paint_vel_info,
            distribution_mode=distribution_mode,
            **kwargs,
        )
        observation_space = vt2space(self.procgen_env.ob_space)
        action_space = vt2space(self.procgen_env.ac_space)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed: int | List[int] | None = None, options: dict | None = None):
        _rew, ob, first = self.procgen_env.observe()
        if not first.all():
            logging.warning("Warning: manual reset ignored")
        return ob, [{} for _ in range(self.num_envs)]

    def step_async(self, actions):
        self.procgen_env.act(actions)

    def step_wait(self, **kwargs) -> tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        rew, ob, first = self.procgen_env.observe()
        return (
            ob,
            rew,
            first,
            np.zeros_like(first),
            [{} for _ in range(self.num_envs)],
        )  # discard info dict
        # return ob, rew, first, np.zeros_like(first), self.procgen_env.get_info()


class VectorFilterObservation(gym.vector.VectorEnvWrapper):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        filter_key: str,
    ):
        super().__init__(env)
        self.filter_key = filter_key
        self.observation_space = self.env.observation_space[filter_key]
        self.single_observation_space = self.env.single_observation_space[filter_key]

    def reset(self, *, seed: int | List[int] | None = None, options: dict | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.vector_observation(observation), info

    def step(self, actions) -> tuple[Any, NDArray[Any], NDArray[Any], NDArray[Any], dict]:
        observation, reward, terminated, truncated, info = self.env.step(actions)
        return self.vector_observation(observation), reward, terminated, truncated, info

    def vector_observation(self, observation):
        return observation[self.filter_key]


class VectorStepDataCallback:
    def __call__(
        self,
        env: gym.vector.VectorEnv,
        observations: gym.core.ObsType,
        actions: gym.core.ActType,
        rewards: List[Any],
        terminations: List[bool],
        truncations: List[bool],
        infos: List[dict[str, Any]],
    ) -> StepData:
        steps_data = [
            {
                "action": actions[i],
                "observation": observations[i],
                "reward": rewards[i],
                "termination": terminations[i],
                "truncation": truncations[i],
                "info": infos[i],
            }
            for i in range(env.num_envs)
        ]
        return steps_data


class VectorDataCollector(minari.DataCollector):
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        vector_step_data_callback: Type[VectorStepDataCallback] = VectorStepDataCallback,
        episode_metadata_callback: Type[EpisodeMetadataCallback] = EpisodeMetadataCallback,
        record_infos: bool = False,
        max_buffer_steps: int | None = None,
        single_observation_space=None,
        single_action_space=None,
        data_format: str | None = None,
    ):
        super().__init__(
            env,
            episode_metadata_callback=episode_metadata_callback,
            record_infos=record_infos,
            observation_space=single_observation_space,
            action_space=single_action_space,
            data_format=data_format,
        )
        self._vector_step_data_callback = vector_step_data_callback()
        self._episode_metadata_callback = episode_metadata_callback()

        # ------ Differences from minari.DataCollector ------
        self._ongoing_episodes: List[EpisodeBuffer] = [EpisodeBuffer() for _ in range(env.num_envs)]
        self._finished_episodes: List[EpisodeBuffer] = []
        self._finished_steps = 0
        self.total_episodes = 0
        self.total_steps = 0
        self.max_buffer_steps = max_buffer_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        # Global seeding (only support same seed for all envs in VectorEnv)
        autoseed_enabled = (not options) or options.get("minari_autoseed", True)
        if seed is None and autoseed_enabled:
            seed = secrets.randbits(AUTOSEED_BIT_SIZE)
        self.obs, info = self.env.reset(seed=seed, options=options)
        return self.obs, info

    def process_single_step(self, env_idx: int, step_data: StepData):
        """
        Add single step to ongoing episodes buffer and move finished episodes
        """
        self._ongoing_episodes[env_idx] = self._ongoing_episodes[env_idx].add_step_data(step_data)

        # Episode end
        if step_data["termination"] or step_data["truncation"]:
            # Save finished episode
            ep = dataclasses.replace(self._ongoing_episodes[env_idx], id=self.total_episodes)
            self._finished_episodes.append(ep)
            # Update counters
            episode_length = len(self._ongoing_episodes[env_idx])
            self._finished_steps += episode_length
            self.total_steps += episode_length
            self.total_episodes += 1
            # Reset the ongoing episode buffer
            self._ongoing_episodes[env_idx] = EpisodeBuffer()

    def step(
        self, actions: gym.core.ActType
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

        steps_data = self._vector_step_data_callback(
            env=self.env,
            observations=self.obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
        )
        self.obs = next_obs

        # Check that the stored obs/act spaces comply with the dataset spaces
        if not self._storage.observation_space.contains(steps_data[0]["observation"]):
            logging.warning(
                "Observation is not in observation space.\n"
                f"Observation: {steps_data[0]['observation']}\nSpace: {self._storage.observation_space}"
            )
        if not self._storage.action_space.contains(steps_data[0]["action"]):
            logging.warning(
                "Action is not in action space.\n"
                f"Action: {steps_data[0]['action']}\nSpace: {self._storage.action_space}",
            )

        # Process step in each environment
        for env_idx, step_data in enumerate(steps_data):
            self.process_single_step(env_idx, step_data)

        # Check if buffer is full
        if self.max_buffer_steps is not None and self._finished_steps > self.max_buffer_steps:
            self._flush_to_storage()

        return next_obs, rewards, terminations, truncations, infos

    def _flush_to_storage(self):
        if self._finished_episodes:
            self._storage.update_episodes(self._finished_episodes)
            self._finished_steps = 0
            self._finished_episodes = []


@dataclasses.dataclass
class Args:
    # Data collection related
    ckpt_path: str | None = None
    "path to the checkpoint agent"
    data_root: str | None = None
    "root directory of the data folder"
    data_version: int = 0
    "dataset version"
    total_timesteps: int = int(1e6)
    """total timesteps to be collected"""
    max_buffer_steps: int = int(1e5)
    """number of timesteps in the buffer before saving to a temporary HDF5 file"""

    # Run related
    seed: int = 1
    """seed of the experiment"""
    th_deterministic: bool = True
    """if toggled, `th.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Environment specific arguments
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

    # to be filled in runtime
    env_id: str = ""
    """the id of the environment, to be inferred from the ckpt path"""

    encoder: EncoderArgs = dataclasses.field(default_factory=lambda: EncoderArgs(
        type="cnn",
    ))


def collect(args):
    # Extract env id from the checkpoint path and set data root
    if args.ckpt_path is not None and args.env_id == "":
        ckpt_path = pathlib.Path(args.ckpt_path)
        args.env_id = ckpt_path.parent.name
    assert args.env_id in procgen.env.ENV_NAMES

    minari_root = pathlib.Path(args.data_root)
    os.environ["MINARI_DATASETS_PATH"] = str(minari_root)
    print(f"Minari root: {minari_root}")
    mlflow.log_params(dataclasses.asdict(args))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.backends.cudnn.deterministic = args.th_deterministic
    device = th.device("cuda" if th.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    env = ProcgenGymnasiumVectorEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        num_levels=args.num_levels,
        start_level=args.start_level,
        distribution_mode=args.distribution_mode,
        use_backgrounds=args.use_backgrounds,
    )
    env = VectorFilterObservation(env, filter_key="rgb")
    env = VectorDataCollector(
        env,
        max_buffer_steps=args.max_buffer_steps,
        single_observation_space=env.single_observation_space,
        single_action_space=env.single_action_space,
    )

    # Update metadata
    env._storage.update_metadata(
        {
            "num_levels": args.num_levels,
            "start_level": args.start_level,
            "distribution_mode": args.distribution_mode,
            "use_backgrounds": args.use_backgrounds,
        }
    )

    # Load agent
    print("Start loading")
    agent = Agent(env, args).to(device)
    if args.ckpt_path is not None:
        agent.load_state_dict(th.load(pathlib.Path(args.ckpt_path), map_location=device))

    # Collect
    obs, infos = env.reset()
    iteration = 0
    print("Start collecting")
    with mock.patch(
        "minari.dataset._storages.hdf5_storage._add_episode_to_group", _add_episode_to_group
    ):  # patch minari
        while env.total_steps < args.total_timesteps:
            if (iteration + 1) % 100 == 0:
                print(f"Steps: {env.total_steps} | Episodes: {env.total_episodes}")
            # Forward the policy
            if args.ckpt_path is None:
                actions = np.array([env.action_space.sample() for _ in range(args.num_envs)])
            else:
                with th.no_grad():
                    obs_t = th.tensor(obs, device=device)
                    hidden = agent.network(obs_t.permute((0, 3, 1, 2)) / 255.0)  # "bhwc" -> "bchw"
                    logits = agent.actor(hidden)
                    probs = Categorical(logits=logits)
                    actions = probs.sample().cpu().numpy()

            # Step the environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            obs = next_obs
            iteration += 1

        # Save dataset
        print("Saving data ...")
        dataset = env.create_dataset(
            dataset_id=f"procgen-{args.env_id}-v{args.data_version}",
            algorithm_name="Phasic Policy Gradient",
            author="Kaixin",
            author_email="kaixin96.wang@gmail.com",
        )
        print("Dataset saved!")

    print("Moving data to blob storage ...")
    print("Finished!")


if __name__ == "__main__":
    args = tyro.cli(Args)
    collect(args)
