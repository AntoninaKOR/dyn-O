from oc_utils.replay_buffer.episode_replay_buffer import DummnyEpisodesDataset
from oc_utils.replay_buffer.offline_episode_buffer import OfflineEpisodeBuffer

replay_buffer_library = {
    "online": DummnyEpisodesDataset,
    "offline": OfflineEpisodeBuffer,
}
