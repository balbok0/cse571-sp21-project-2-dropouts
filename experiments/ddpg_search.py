from gym_duckietown.envs.multimap_env import MultiMapEnv
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models import ModelCatalog
import numpy as np
from dropouts_project import ImageCritic, RLLibTorchModel, MultiMapSteeringToWheelVelWrapper, CustomRewardWrapper
from experiments.experiment import Experiment


class DDPGRLLibModel(RLLibTorchModel):
    @staticmethod
    def base_model_generator():
        return ImageCritic(256, 256)


if __name__ == "__main__":
    # Start ray
    ray.init()

    ModelCatalog.register_custom_model(
        "image-ddpg", DDPGRLLibModel,
    )

    wrappers = lambda x: CustomRewardWrapper(MultiMapSteeringToWheelVelWrapper(x))  # noqa: E731
    register_env(
        "DuckieTown-MultiMap",
        lambda _: wrappers(MultiMapEnv())
    )

    experiment = Experiment(
        wrappers=wrappers,
        env_name="DuckieTown-MultiMap",
        trainer_class=DDPGTrainer,
        trainer_config={
            "framework": "torch",
            "model": {
                "custom_model": "image-ddpg",
            },
            # "use_state_preprocessor": True,
            "learning_starts": 0,
            # "record_env": True,  # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.  # noqa: E501
            "train_batch_size": 16,
            # Use a very small buffer to reduce memory usage, default: 50_000.
            "buffer_size": 1000,
            # No hidden layers
            "actor_hiddens": [],
            "critic_hiddens": [],

            "compress_observations": True,
            "num_workers": 0,
            "num_gpus": 0.66,
            "log_level": "ERROR",
        },
        search_additional_config={
            "lr": lambda: 10 ** np.random.uniform(-5, -2.5),
            "learning_starts": lambda: np.random.choice(list(range(100, 1001, 100))),
            "grad_clip": lambda: np.random.choice([None, 10, 20, 40, 50, 60], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1]),
        },
    )

    experiment.run()
