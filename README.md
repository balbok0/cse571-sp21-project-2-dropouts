# cse571-sp21-project-2-dropouts

## Setup

### Conda
```
# Create conda environment
conda create -n robotics-project python=3.8
conda activate robotics-project

# Assuming starting from root of this repo
git clone https://github.com/duckietown/gym-duckietown.git ../gym-duckietown
pip3 install -e ../gym-duckietown --use-feature=2020-resolver

# Install torch
# Note this is cpu only version you might want specific CUDA version:
# https://pytorch.org/
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install atari_py - dependency of rllib which is broken on pip
conda install -c conda-forge atari_py

# Install rllib and additional ray dependencies
pip install 'ray[rllib]'
pip install 'ray[default]'

# Get missing maps from duckietown repo
wget https://raw.githubusercontent.com/duckietown/gym-duckietown/daffy/src/gym_duckietown/maps/loop_only_duckies.yaml -P /home/$USER/anaconda3/envs/robotics-project/lib/python3.8/site-packages/duckietown_world/data/gd1/maps/
wget https://raw.githubusercontent.com/duckietown/gym-duckietown/daffy/src/gym_duckietown/maps/small_loop_only_duckies.yaml -P /home/$USER/anaconda3/envs/robotics-project/lib/python3.8/site-packages/duckietown_world/data/gd1/maps/

# Optional: Install Tensorboard (useful for visualizing training progression)
pip install tensorflow tensorboard
```

### Virtualenv
```
mkvirtualenv -p python3.8 project2_dropouts

pip install -r requirements.txt

wget https://raw.githubusercontent.com/duckietown/gym-duckietown/daffy/src/gym_duckietown/maps/loop_only_duckies.yaml -P $VIRTUAL_ENV/lib/python3.8/site-packages/duckietown_world/data/gd1/maps/
wget https://raw.githubusercontent.com/duckietown/gym-duckietown/daffy/src/gym_duckietown/maps/small_loop_only_duckies.yaml -P /$VIRTUAL_ENV/lib/python3.8/site-packages/duckietown_world/data/gd1/maps/
```

## Checking setup
If everything got installed properly
```
python -m examples.discrete_dqn
```

If you also installed tensorboard running:
```
tensorboard --logdir=~/ray_results
```
Will provide insights into training.

## Colab Training

There is an included notebook which can be imported to Colab to train using GPU, see `colab_notebook.ipynb`.

The notebook is also accessible here: https://colab.research.google.com/drive/1FJm_wAT3DyYX4pQMVrsUJXPdNZ06M1TZ?usp=sharing

## Evaluation

### Training

Each agent/model logs the results for each epoch of training using a `plotter.Plotter()` object. At the end of a training run there is a plot saved to the directory which you executed from.

You can also manually generate a plot from just the saved reward history data by running:
```
python plotter.py <path_to_file.txt>
```

### Model Evaluation
Evaluation should be done locally if you are rendering the simulator, though you can add support to ignore rendering for cloud evaluation.

To evaluate a trained model (only DQN supported right now) look at the available flags:
```
$ python -m examples.discrete_dqn -h

usage: discrete_dqn.py [-h] [--eval] [--epochs EPOCHS] [--model_path MODEL_PATH] [--map MAP]

optional arguments:
  -h, --help            show this help message and exit
  --eval                Evaluate model instead of train
  --epochs EPOCHS       Num of epochs to train for
  --model_path MODEL_PATH
                        Location to pre-trained model
  --map MAP             Which map to evaluate on, see https://git.io/J3jES
```

For example we can train on the default map using the provided baseline model with:
```
$ python -m examples.discrete_dqn --eval --model_path trained_models/dqn_v0/model
```

The baseline model is terrible, you should see it go around in circles slowly.
