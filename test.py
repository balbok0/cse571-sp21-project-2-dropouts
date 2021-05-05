from gym_duckietown.envs.multimap_env import MultiMapEnv
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn


class ImageCritic(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(666, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        # (N, H, W, C) -> (N, C, H, W)
        x = torch.moveaxis(x, 3, 1).float()
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.linear(x) * 2 - 1



class RLLibDQNCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = ImageCritic()

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

# Start ray
ray.init()

ModelCatalog.register_custom_model("image-dqn", RLLibDQNCritic)
register_env("DuckieTown-MultiMap", lambda _: MultiMapEnv())

trainer = ImpalaTrainer(
    env="DuckieTown-MultiMap",
    config={
        "framework": "torch",
        "model": {
            "custom_model": "image-dqn",
        },
    }
)

result = trainer.train()
print(result)