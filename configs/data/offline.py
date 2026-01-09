import dataclasses


@dataclasses.dataclass
class OfflineConfig:
    """
    Assume the following directory structure:
        config.data.data_path
        ├── dataset_id1
        │   ├── data
        │   │   ├── main_data.hdf5              # minari data
        │   │   ├── metadata.json               # minari metadata
        │   │   ├── encoding_fname              # precomputed encodings
    """
    data_path: str = "/scratch/cluster/zzwang_new/procgen_data"
    dataset_ids: list = dataclasses.field(
        default_factory=lambda: ["procgen-coinrun-v3"]
    )
    encoding_fname: str = "encoder_sam_attn_loss/encoding.h5"

    # a small portion of the data is used for kmeans clustering
    encoding_for_kmeans_fname: str = "encoder_sam_attn_loss/encoding_for_kmeans.h5"
    num_episodes_for_kmeans: int = 1000
    num_steps_per_episode_for_kmeans: int = 10

    reward_termination_stats_fname = "reward_termination_stats.json"

    # X% of the data is used for training, the rest is used for evaluation
    train_ratio: float = 0.95

    # use X% of the training data
    train_prop: float = 1.0

    num_envs: int = 1  # placeholder for env initialization
