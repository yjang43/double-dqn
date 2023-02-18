# Double DQN

This is a PyTorch implementation of [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf).

The Reinforcement Learning agent is trained to play Atari games (ALE/Pong-v5) from pixels.


## Installation
```bash
pip install -r requirements.txt
pip install "gymnasium[atari]"

# If using Apple M1, to the following.
# AutoROM --accept-license
```

## Run
```bash
# Make adjustment to config files in config directory
python main.py --config_path ./config/base.yaml
```

## Result

### Score report
![Score Report](img/ALE%3APong-v5.png)

### Sampled episodes
##### Episode 1
![Episode 1](img/episode_1.gif)
##### Episode 10
![Episode 10](img/episode_10.gif)
##### Episode 100
![Episode 100](img/episode_100.gif)
##### Episode 1000
![Episode 1000](img/episode_1000.gif)
