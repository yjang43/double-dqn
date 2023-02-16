from double_dqn import DoubleDQN
from util import set_seed
from argparse import ArgumentParser
from util import Config



def set_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="config/base.yaml",
        type=str
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    config = Config(args.config_path)

    set_seed(config.seed)

    double_dqn = DoubleDQN(config)
    double_dqn.train()
    