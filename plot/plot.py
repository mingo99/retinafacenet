import pandas as pd
import matplotlib.pyplot as plt
from .parse import parse_lr, parse_loss

def plot_lr(log_path: str, save_path: str = 'lr.png'):
    lrs = parse_lr(log_path)
    iterations = len(lrs)*50
    iteration_numbers = list(range(1, iterations + 1, 50))

    plt.plot(iteration_numbers, lrs, linestyle='-', color='b', label='Learning Rate', linewidth=1)
    plt.xlabel('Iters')
    plt.ylabel('Learning Rate')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_loss(log_path: str, save_path: str = 'loss.png'):
    losses = parse_loss(log_path)
    iterations = len(losses[0])*50
    iteration_numbers = list(range(1, iterations + 1, 50))

    for i, l in enumerate(losses):
        l_series = pd.Series(l)
        smooth_l_series = l_series.rolling(window=50).mean()
        losses[i] = smooth_l_series
        

    plt.plot(iteration_numbers, losses[0], linestyle='-', color='b', label='Total Loss', linewidth=1)
    plt.plot(iteration_numbers, losses[1], linestyle='-', color='r', label='Bbox Regression Loss', linewidth=1)
    plt.plot(iteration_numbers, losses[2], linestyle='-', color='g', label='Classification Loss', linewidth=1)
    plt.plot(iteration_numbers, losses[3], linestyle='-', color='y', label='Keypoint Regression Loss', linewidth=1)
    plt.xlabel('Iters')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
