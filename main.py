from double_dqn import DoubleDQN



if __name__ == '__main__':
    double_dqn = DoubleDQN("config/base.yaml")
    double_dqn.train()