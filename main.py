from double_dqn import DoubleDQN
from util import set_seed



if __name__ == '__main__':

    set_seed(42)
    double_dqn = DoubleDQN("config/base.yaml")
    double_dqn.train()
    