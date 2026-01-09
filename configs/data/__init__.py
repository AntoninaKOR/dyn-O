from dataclasses import dataclass, field

from configs.data.online import OnlineConfig
from configs.data.offline import OfflineConfig


@dataclass
class DataConfig:
    online: OnlineConfig = field(default_factory=OnlineConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)

    def __post_init__(self):
        self.configs = {
            "online": self.online,
            "offline": self.offline
        }
