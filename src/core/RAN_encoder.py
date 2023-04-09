import torch
from torch import nn

from torch import Tensor


class RanEncoder(nn.Module):
    def __init__(self, per_dim, in_channel, out_channel):
        super().__init__()
        self.per_dim = per_dim
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(self.out_channel)
        self.drop_out = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor):
        """

        :param x: shape(bs,features_dim)
        :return:
        """
        
        x_nums = x.shape[0]
        x = torch.reshape(x, (-1, self.in_channel, self.per_dim, self.per_dim))
        out = self.conv1(x)
        out = self.bn(out)
        out = self.drop_out(out)
        out = self.relu(out)
        
        out = torch.reshape(out, (x_nums, self.out_channel, -1))

        return out
