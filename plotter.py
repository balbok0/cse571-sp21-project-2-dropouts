import sys
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class Plotter:
    def __init__(self, experiment_name: str = 'foo', prior_file: Optional[str] = None):
        if prior_file is not None:
            self.data_filename = prior_file
            self.plot_filename = 'your_plot'
        else:
            self.plot_filename = f'{experiment_name}_plot'
            self.data_filename = f'{experiment_name}_plot_data.txt'

        self.data_file = open(self.data_filename, 'a')

    def add_results(self, results: Dict[str, Any]) -> None:
        log_line = f"{results['training_iteration']},{results['episode_reward_mean']}\n"
        self.data_file.write(log_line)
        # In case training crashes, keep writing data out.
        self.data_file.flush()

    def plot(self, title: Optional[str] = None) -> None:
        self.data_file.close()
        with open(self.data_filename, 'r') as f:
            data = [x.split(',') for x in f.read().splitlines()]
            data = [[int(x[0]), float(x[1])] for x in data]
        df = pd.DataFrame(data)
        plot_data(df, title=title, plot_name=self.plot_filename)


def plot_data(data, xaxis=0, value=1, smooth=1, plot_name='your_plot', title=None):
    """
    Adapted from: https://git.io/JsYa0
    """
    if smooth > 1:
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)

    sns.lineplot(data=data, x=xaxis, y=value, ci='sd')

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Reward')

    if title is not None:
        plt.title(title)

    plt.savefig(plot_name, bbox_inches='tight')


if __name__ == '__main__':
    plot = Plotter(prior_file=sys.argv[1])
    plot.plot(title='PPO CartPole-v0')
