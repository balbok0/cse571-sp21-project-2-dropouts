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
python examples/discrete_dqn.py
```

If you also installed tensorboard running:
```
tensorboard --logdir=~/ray_results
```
Will provide insights into training.
