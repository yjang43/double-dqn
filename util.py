import yaml
import random

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt



class Config:
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

    def __getattr__(self, attr):
        try:
            return self.cfg[attr]
        except KeyError:
            raise AttributeError

    def __setattr__(self, attr, value):
        # append unset hyperparameters to the list
        if attr in ['action_n']:
            self.cfg[attr] = value
        else:
            super(Config, self).__setattr__(attr, value)

# util
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def plot_graph(file_path, x, y):
    plt.ylabel("Score")
    plt.xlabel("Training steps")
    plt.plot(x, y)
    plt.title(file_path)
    plt.savefig(file_path)


def write_video(video_path, frames, fps=24):
    assert '.mp4' in video_path, "Only supports *.mp4 for codec"
    height, width, _ = frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)
    video.release()
