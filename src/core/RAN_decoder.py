# 实现protein_feature和drug_feature的拼接

import torch
from torch import nn
from torch import Tensor


class RanDecoder(nn.Module):
    def __init__(self, in_channel, per_dim, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.per_dim = per_dim
        self.fc1 = nn.Linear(self.in_channel * self.per_dim, self.out_channel)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.out_channel * 2, self.out_channel)
        self.fc3 = nn.Linear(self.out_channel, 4)
        self.fc4 = nn.Linear(4, 2)

    def forward(self, x: Tensor, y: Tensor):
        """

        :param x:
        :param y:
        :return:
        """
        # x.shape(bs,in_channel,per_dim)
        x_num = x.shape[0]
        # ->(bs, 1, in_channel*per_dim)
        x1 = torch.reshape(x, (x_num, 1, -1))
        # ->(bs, in_channel*per_dim)
        x2 = torch.squeeze(x1, 1)
        x3 = self.fc1(x2)

        y_num = y.shape[0]
        # ->(bs, 1, in_channel*per_dim)
        y1 = torch.reshape(y, (y_num, 1, -1))
        # ->(bs, in_channel*per_dim)
        y2 = torch.squeeze(y1, 1)
        y3 = self.fc1(y2)

        out = torch.concat((x3, y3), dim=-1)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out
