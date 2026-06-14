"""AlphaZero 风格的策略-价值网络 (ResNet)。

输入:  (B, INPUT_CHANNELS, 10, 9)
输出:
  policy_logits: (B, ACTION_SIZE)  走子策略 logits(未 softmax)
  value:         (B,)              当前走方视角的局面评估,tanh 输出 [-1, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import NUM_ROWS, NUM_COLS
from .encoding import ACTION_SIZE
from .state_encoder import INPUT_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class XiangqiNet(nn.Module):
    """策略-价值双头 ResNet。

    个人/中等规模项目默认 channels=128, blocks=10;云端多卡可调大。
    """

    def __init__(self, channels: int = 128, num_blocks: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # 策略头
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * NUM_ROWS * NUM_COLS, ACTION_SIZE)

        # 价值头
        self.value_conv = nn.Conv2d(channels, 16, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * NUM_ROWS * NUM_COLS, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


def masked_policy(policy_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对非法动作施加掩码后做 softmax。

    policy_logits, mask: (B, ACTION_SIZE),mask 为 0/1。
    返回合法动作上归一化的概率分布(非法动作概率为 0)。
    """
    neg_inf = torch.finfo(policy_logits.dtype).min
    masked = torch.where(mask.bool(), policy_logits, neg_inf)
    return F.softmax(masked, dim=-1)
