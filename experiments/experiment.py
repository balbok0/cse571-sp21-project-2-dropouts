import argparse
import os
import time
import yaml
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from gym_duckietown import simulator

from ray.tune.registry import register_env


class Experiment:
    def __init__(self, wrappers, env_name, trainer_class, trainer_config, search_additional_config=None):
        self.wrappers = wrappers
        self.env_name = env_name

        self.trainer_class = trainer_class
        self.trainer_config = trainer_config

        self.search_additional_config = {} if search_additional_config is None else search_additional_config

    def train_model(self, epochs, model_path=None, trainer_config=None, use_model_path_config=True):
        # Start training from a checkpoint, if available.
        if trainer_config is None:
            if model_path and use_model_path_config:
                with open(Path(model_path).parent / "trainer_config.yaml", mode="r") as f:
                    trainer_config = yaml.safe_load(f)
            else:
                trainer_config = self.trainer_config

        trainer = self.trainer_class(
            env=self.env_name,
            config=trainer_config,
        )

        if model_path:
            trainer.restore(model_path)

        # TODO(balbok0): Start values from checkpoint, if available.
        best_mean_reward = -np.inf
        epoch_of_best_mean_reward = 0
        path_of_best_mean_reward = None

        for i in range(epochs):  # Number of episodes (basically epochs)
            print(f'----------------------- Starting epoch {i} ----------------------- ')
            # train() trains only a single episode
            result = trainer.train()
            # print(result)

            # Save model so far.
            checkpoint_path = trainer.save()

            if i == 0:
                with open(Path(checkpoint_path).parent / "trainer_config.yaml", mode="w") as f:
                    yaml.safe_dump(trainer_config, f)
            # print(f'Epoch {i}, checkpoint saved at: {checkpoint_path}')

            if result["episode_reward_mean"] > best_mean_reward:
                best_mean_reward = result["episode_reward_mean"]
                epoch_of_best_mean_reward = i
                path_of_best_mean_reward = checkpoint_path

            # Cleanup CUDA memory to reduce memory usage.
            torch.cuda.empty_cache()
            # Debug log to monitor memory.
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        return best_mean_reward, epoch_of_best_mean_reward, path_of_best_mean_reward

    def evaluate_model(self, model_path, wrappers, map="loop_empty", video_name=None, max_steps=2000, config=None):
        if model_path == '':
            print('Cannot evaluate model, no --model_path set')
            exit(1)

        if config is None:
            try:
                with open(Path(model_path).parent / "trainer_config.yaml", mode="r") as f:
                    config = yaml.safe_load(f)
            except Exception:
                config = self.trainer_config

        def get_env():
            # Simulator env uses a single map, so better for evaluation/testing.
            # return SteeringToWheelVelWrapper(DuckietownLF(
            # ))
            return wrappers(simulator.Simulator(
                map_name=map,
                max_steps=max_steps,
            ))

        if video_name is None:
            mode = "human"
        else:
            mode = "rgb_array"
            video_path = f"videos/{video_name}.avi"

        # Rather than reuse the env, another one is created later because I can't
        # figure out how to provide register_env with an object, th
        register_env('DuckieTown-Simulator', lambda _: get_env())
        trainer = self.trainer_class(
            env="DuckieTown-Simulator",
            config=config,
        )
        trainer.restore(model_path)

        sim_env = get_env()

        # Standard OpenAI Gym reset/action/step/render loop.
        # This matches how the `enjoy_reinforcement.py` script works, see: https://git.io/J3js2
        done = False
        observation = sim_env.reset()
        episode_reward = 0
        images = []
        while not done:
            action = trainer.compute_action(observation)
            observation, reward, done, _ = sim_env.step(action)
            episode_reward += reward
            img = sim_env.render(mode=mode)
            if mode == "rgb_array":
                images.append(img)

        if mode == "rgb_array":
            self.save_video(images, video_path,  round(len(images) / sim_env.timestamp))

        print(f"Last Timestamp: {sim_env.timestamp}")

        print(f'Episode complete, total reward: {episode_reward}')

    def search_models(self, search_name, n_searches, epochs):
        csv_path = f"searches/{search_name}_results.csv"

        if not Path(csv_path).parent.exists():
            os.makedirs(Path(csv_path).parent)

        for search_idx in range(n_searches):
            print(f'----------------------- Starting Search {search_idx} ----------------------- ')
            additional_params = {}

            config = self.trainer_config.copy()

            for k, v in self.search_additional_config.items():
                additional_params[k] = v()

            for k, v in additional_params.items():
                config[k] = v

            try:
                best_mean_reward, best_epoch, best_path = self.train_model(epochs, trainer_config=config)

                additional_params["best_mean_reward"] = best_mean_reward
                additional_params["best_epoch"] = best_epoch
                additional_params["best_path"] = best_path

                # No need to save lr_schedule, since it's deterministic on lr
                additional_params.pop("lr_schedule", None)

                params_for_df = {}
                for k, v in additional_params.items():
                    params_for_df[k] = [v]

                pd.DataFrame.from_dict(params_for_df).to_csv(
                    csv_path,
                    mode="a" if os.path.exists(csv_path) else "w",
                    header=not os.path.exists(csv_path)
                )
            except RuntimeError as e:
                if str(e).startswith("RuntimeError: CUDA out of memory."):
                    # Not enough memory on GPU. Might be bad config, or a CUDA not keeping up. Give it a minute.
                    time.sleep(20)
                else:
                    raise e
            finally:
                torch.cuda.empty_cache()

    @staticmethod
    def save_video(images, video_path, fps):
        height, width, layers = images[0].shape

        if os.path.exists(video_path):
            os.remove(video_path)

        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

        for image in images:
            video.write(image)

        video.release()
        # cv2.destroyAllWindows()
        print(f"Video saved to: {video_path}")

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()

        # Training options
        parser.add_argument('--epochs', default=10, type=int, help='Num of epochs to train for')

        # Evaluation options
        parser.add_argument('--eval', action='store_true', default=False, help='Evaluate model instead of train')
        parser.add_argument(
            '--video-name',
            default=None,
            type=str,
            help='Only effective in evaluation. Where to save the video. '
            'If None (default) renders image on the fly, without saving it into a file.'
        )
        parser.add_argument('--map', default='loop_empty', help='Which map to evaluate on, see https://git.io/J3jES')
        parser.add_argument('--hyper-search', default='', help='Name of hyperparameter search.')
        parser.add_argument(
            '--n_searches',
            default=50,
            type=int,
            help='Only effective if hyper-search is set. Num of epochs to train for.'
        )
        # Looks like, e.g. `DQN_DuckieTown-MultiMap_2021-05-10_20-43-32ndgfiq44/checkpoint_000009/checkpoint-9`
        parser.add_argument('--model_path', default='', help='Location to pre-trained model')

        return parser.parse_args()

    def run(self, args=None):
        if args is None:
            args = self.get_parser()

        if args.eval:
            self.evaluate_model(args.model_path, self.wrappers, args.map, args.video_name)
        elif args.hyper_search:
            self.search_models(args.hyper_search, args.n_searches, args.epochs)
        else:
            self.train_model(args.epochs, model_path=args.model_path)
