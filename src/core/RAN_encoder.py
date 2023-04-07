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
        # 首先将其转化为(bs,out_channel,per_num,per_num)的形式
        x_nums = x.shape[0]
        x = torch.reshape(x, (-1, self.in_channel, self.per_dim, self.per_dim))
        out = self.conv1(x)
        out = self.bn(out)
        out = self.drop_out(out)
        out = self.relu(out)
        # 再降维(bs,out_channel,out)
        out = torch.reshape(out, (x_nums, self.out_channel, -1))

        return out


if __name__ == '__main__':
    exp = torch.randn(4, 16, 36)
    print(exp.shape)
    model = RanEncoder(6, 16, 8)
    y = model(exp)
    print(y.shape)
    # a = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
    #                   [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
    # b = torch.reshape(a, (2, -1))
    # print(b)

# a = torch.rand((4, 12))
# b = torch.rand((4, 12))
#
# c = torch.concat((a, b), dim=1)
#
# print(c.size())
