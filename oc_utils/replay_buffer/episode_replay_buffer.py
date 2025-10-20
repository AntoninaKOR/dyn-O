from collections import deque
import numpy as np


class DummnyEpisodesDataset:
    def __init__(self, config, **kwargs):
        self.config = config
        self.episodes = deque()

    def __len__(self) -> int:
        return len(self.episodes)

    def add(self, *args, **kwargs):
        pass

    def sample(self, num_transitions: int):
        encodings = np.random.randn(num_transitions, *self.config.encoding_space.shape).astype(self.config.np_dtype)
        actions = np.random.randint(self.config.action_space.n, size=num_transitions)
        rewards = np.random.uniform(-1, 1, num_transitions).astype(self.config.np_dtype)
        dones = np.random.rand(num_transitions) < 0.1
        next_encodings = np.random.randn(num_transitions, *self.config.encoding_space.shape).astype(self.config.np_dtype)
        return encodings, actions, rewards, dones, next_encodings

    def sample_episode(self, num_episodes: int):
        encodings, actions, rewards, dones, num_transitions = [], [], [], [], []

        for _ in range(num_episodes):
            num_transition = np.random.randint(10, 50)
            encoding, action, reward, done, _ = self.sample(num_transition)

            encodings.append(encoding)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            num_transitions.append(num_transition)

        max_len = max(num_transitions)
        encodings, actions, rewards, dones = map(
            lambda x: np.stack(
                [
                    np.pad(
                        e,
                        [(0, max_len - len(e))] + [(0, 0)] * (e.ndim - 1),      # pad each episode to max_len
                        mode='constant'
                    )
                    for e in x
                ],
                axis=0,
            ),
            [encodings, actions, rewards, dones]
        )
        padding_mask = np.zeros((num_episodes, max_len), dtype=bool)
        for i, num_transition in enumerate(num_transitions):
            padding_mask[i, num_transition:] = True

        return encodings, actions, rewards, dones, padding_mask
