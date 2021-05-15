from torch import nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


class RLLibTorchModel(TorchModelV2, nn.Module):
    """
    This is basically a wraper around ImageCritic that satisfies requirements of rllib.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = self.base_model_generator()
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        # Structure of this forward is due to TorchModelV2.
        # Note that we do not immediately return value, but rather save it for `value_function`
        model_out, self._value = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return self._value

    @staticmethod
    def base_model_generator():
        raise NotImplementedError()
