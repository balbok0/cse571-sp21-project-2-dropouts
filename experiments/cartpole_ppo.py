import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn


class ImageCritic(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.agent = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.float()
        x = self.common(x)
        a = self.agent(x)
        v = self.critic(x)
        return a, v


class RLLibPPOCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = ImageCritic()
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        # Structure of this forward is due to TorchModelV2.
        # Note that we do not immediately return value, but rather save it for `value_function`
        model_out, self._value = self.base_model(input_dict["obs"])
        # l = np.array([last_r])
        # if l.shape == (1,):
        #     l = l.reshape((1, 1))
        return model_out, state

    def value_function(self):
        print(self._value.shape)
        return self._value


ModelCatalog.register_custom_model("image-ppo", RLLibPPOCritic)

ray.init()
trainer = PPOTrainer(
    env="CartPole-v0",
    config={
        "framework": "torch",
        "model": {
            "custom_model": "image-ppo",
        },
    }
)

trainer.train()
