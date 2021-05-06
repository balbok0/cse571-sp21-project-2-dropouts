from gym_duckietown.envs.multimap_env import MultiMapEnv
from gym_duckietown.envs.duckietown_env import DuckietownLF
from gym_duckietown.wrappers import PyTorchObsWrapper, SteeringToWheelVelWrapper, DiscreteWrapper
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn
from tqdm import trange


class ImageCritic(nn.Module):
    def __init__(self):
        """CNN that does the predictions.
        Data from is as follows:

                           > agent(x) -> a (R^256)
                          /
            x -> common(x)
                         |
                         L> critic(x) -> c (R^256)

        The reason for sharing common part is because images are large so that many Conv's is quite expensive.
        Also predicting value of a state and predicting next action to take is quite similar, and many papers use this approach.

        NOTE: Difference between this and DQN ImageCritic is that critic model return 256 elements.
            This is because DQN then uses 2 linear layers (400, 300 hidden sizes by default), to finally get the value.
        """
        super(self.__class__, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )
        self.agent = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )

    def forward(self, x):
        x = x.transpose(3, 1).float()
        x = self.common(x)
        a = self.agent(x)
        v = self.critic(x)
        return a * 2 - 1, v


class RLLibDQNCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = ImageCritic()
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return self._value

# Start ray
ray.init()

# We are using custom model and environment, which need to be registered in ray/rllib
# Names can be anything.
# NOTE: We are using DuckietownLF environment because SteeringToWheelVelWrapper does not cooperate with multimap.
ModelCatalog.register_custom_model("image-dqn", RLLibDQNCritic)
register_env("DuckieTown-MultiMap", lambda _: SteeringToWheelVelWrapper(DuckietownLF()))

# Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
# Here is a list of default config keys/values: https://docs.ray.io/en/master/rllib-training.html#common-parameters
# For DDPG specifically there are also additionally these keys: https://docs.ray.io/en/master/rllib-algorithms.html#ddpg
trainer = DDPGTrainer(
    env="DuckieTown-MultiMap",
    config={
        "framework": "torch",
        "model": {
            "custom_model": "image-dqn",
        },
        "learning_starts": 0,
        # "record_env": True,  # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.
    }
)

for _ in trange(10):  # Number of epochs
    result = trainer.train()
    print(result)
