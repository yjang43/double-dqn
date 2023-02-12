import torch.nn as nn



class FunctionApproximation(nn.Module):
    def __init__(self, config):
        super(FunctionApproximation, self).__init__()
        # 4 x84 x 84
        self.first_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4,
                # bias=False
            ),
            nn.ReLU()
        )
        # 32 x 20 x 20
        self.second_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                # bias=False
            ),
            nn.ReLU()
        )
        # 64 x 9 x 9
        self.third_conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                # bias=False
            ),
            nn.ReLU()
        )
        # 64 x 7 x 7
        self.hidden_layer = nn.Sequential(
            nn.Linear(
                in_features=7*7*64,
                out_features=512,
                # bias=False
            ),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(
            in_features=512,
            out_features=config.action_n,
            # bias=False
        )
    
    def forward(self, inp):
        hid = self.first_conv_layer(inp)
        hid = self.second_conv_layer(hid)
        hid = self.third_conv_layer(hid)
        hid = self.hidden_layer(hid.view(hid.shape[0], -1))
        out = self.output_layer(hid)
        return out
