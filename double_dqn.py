# import libraries
import os
import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np

from collections import deque
from tqdm import tqdm
from model import FunctionApproximation
from util import Config, write_video



class Agent:
    def __init__(self, func_approx, config):
        self.func_approx = func_approx
        self.action_n = config.action_n
        self.start_epsilon = config.start_epsilon
        self.end_epsilon = config.end_epsilon
        self.anneal_step = config.anneal_step
        self.step = 0

    def sample_action(self, phi_sequence, greedy=False):
        if greedy:
            epsilon = 0
        else:
            epsilon = self._anneal_epsilon()

        with torch.no_grad():
            q = self.func_approx(phi_sequence.unsqueeze(0))

        action = self._epsilon_greedy(q, epsilon)
        return action


    def _anneal_epsilon(self):
        rate = (self.end_epsilon - self.start_epsilon) / self.anneal_step
        return max(self.start_epsilon + rate * self.step, self.end_epsilon)


    def _epsilon_greedy(self, q, epsilon):
        # random action for epsilon otherwise argmax Q(a)
        if random.random() < epsilon:
            action = random.randrange(0, self.action_n)
        else:
            _, max_action = q.max(1)
            action = max_action.item()
        return action


class Environment:
    def __init__(self, config):
        self.env = gym.make(config.atari_id)
        setattr(config, 'action_n', self.env.action_space.n)

        self.skip_frame = config.skip_frame
        self.record_play = config.record_play

    def reset(self):
        # return tensor size of (N, H, W, C)
        if self.record_play:
            self.game_play = []

        observation, info = self.env.reset()
        sequence = np.tile(observation, (self.skip_frame, 1, 1, 1))

        if self.record_play:
            self.game_play.append(observation)

        return sequence, info

    def step(self, action):
        # return tensor size of (N, H, W, C)
        sequence = []
        for i in range(self.skip_frame):
            observation, reward, terminated, truncated, info = self.env.step(action)
            sequence.append(observation)
            if terminated or truncated:
                break

        if self.record_play:
            self.game_play.extend(sequence)

        # in case of termination/truncation during skip-frame, repeat the last frame
        for r_i in range(i + 1, self.skip_frame):
            sequence.append(observation)

        sequence = np.stack(sequence)

        return sequence, reward, terminated, truncated, info
        

class ReplayMemory:
    def __init__(self, config):
        # self.Transition = namedtuple('Transition', ['sequence', 'action', 'reward', 'next_sequence'])
        self.transition_queue = deque([], maxlen=config.replay_memory_size)
        self.batch_size = config.batch_size

    def store_transition(self, phi_sequence, action, reward, phi_next_sequence, termianted):
        self.transition_queue.append((
            phi_sequence,
            action,
            reward,
            phi_next_sequence,
            termianted
        ))
    
    def sample_batch(self, device='cpu'):
        s, a, r, ns, t = zip(*random.sample(self.transition_queue, self.batch_size))
        batch = {
            'phi_sequence': torch.stack(s).to(device),
            'action': torch.tensor(a).to(device),
            'reward': torch.tensor(r).to(device),
            'phi_next_sequence': torch.stack(ns).to(device),
            'terminated': torch.tensor(t).to(device),
        }
        
        return batch


class DoubleDQN:
    def __init__(self, config_path):
        # load config
        self.config = Config(config_path)

        # initialize environment and agent
        self.env = Environment(self.config)
        self.agent = Agent(FunctionApproximation(self.config), self.config)

    def evaluate(self):
        pass

    def train(self):
        policy_net = self.agent.func_approx.to(self.config.device)
        target_net = FunctionApproximation(self.config).to(self.config.device)
        self._update_target_weight(target_net, policy_net)
        

        # initialize replay memory
        replay_memory = ReplayMemory(self.config)

        # initialize criterion and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(
            policy_net.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum
        )

        # logging info
        accum_rewards = []
        episode = 1

        progbar = tqdm(total=self.config.total_step)
        while self.agent.step < self.config.total_step:
            sequence, _ = self.env.reset()
            done = False

            accum_reward = 0

            while not done:
                phi_sequence = self._preprocess(sequence)
                action = self.agent.sample_action(phi_sequence.to(self.config.device))
                next_sequence, reward, terminated, truncated, _ = self.env.step(action)

                # clip reward
                reward = reward > 0
                accum_reward += reward

                phi_next_sequence = self._preprocess(next_sequence)
                assert phi_sequence.is_cpu, "tensors in replay memory should be on CPU"
                replay_memory.store_transition(
                    phi_sequence,
                    action,
                    reward,
                    phi_next_sequence,
                    terminated
                )

                sequence = next_sequence
                done = terminated or truncated

                # sample only if can sample from replay memory
                if len(replay_memory.transition_queue) < self.config.batch_size:
                    continue

                # sample random mini-batch
                batch = replay_memory.sample_batch(device=self.config.device)
                
                # compute target and prediction
                target = self._make_target(batch, target_net, policy_net)
                pred = policy_net(batch['phi_sequence']).gather(1, batch['action'].unsqueeze(0))

                optimizer.zero_grad()
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

                if (self.agent.step + 1) % self.config.target_update_frequency == 0:
                    self._update_target_weight(target_net, policy_net)

                self.agent.step += 1
                progbar.update()

            if episode in self.config.record_episodes:
                write_video(os.path.join("video", f"episode_{episode}.mp4"), self.env.game_play)

            progbar.set_description(f"Episode {episode} reward: {accum_reward}")
            accum_rewards.append(accum_reward)
            episode += 1

    def _preprocess(self, sequence):
        # preprocess (N, H, W, C) to (C', H, W)
        transforms = T.Compose([
            # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.Grayscale(1),
            T.Resize((84, 84)),
        ])
        seq = torch.from_numpy(sequence)
        seq = seq.permute(0, 3, 1 ,2)
        seq = T.functional.convert_image_dtype(seq, dtype=torch.float)
        seq = transforms(seq).squeeze(1)
        return seq

    def _update_target_weight(self, target_net, policy_net):
        target_net.load_state_dict(policy_net.state_dict())
    
    def _make_target(self, batch, target_net, policy_net):
        # Core message of DoubleDQN:
        # Decomposing max operation into action selection and action evaluation
        with torch.no_grad():
            _, max_action = torch.max(policy_net(batch['phi_next_sequence']), 1)
            bootstrap = target_net(batch['phi_next_sequence']).gather(1, max_action.unsqueeze(0))
        
        # mask terminal
        target = batch['reward'] + ~batch['terminated'] * self.config.gamma * bootstrap
        return target


