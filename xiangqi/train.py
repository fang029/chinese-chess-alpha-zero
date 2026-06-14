"""训练:经验回放缓冲 + 损失函数 + 训练循环。

损失:  L = (z - v)^2  -  pi . log(p)  +  L2(由优化器 weight_decay 实现)
  价值项:均方误差
  策略项:MCTS 分布 pi 与网络 softmax 后 log 概率的交叉熵
"""

from __future__ import annotations

import random
from collections import deque

import os
import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """固定容量的样本缓冲,存 (state_tensor, pi_vec, z)。"""

    def __init__(self, capacity: int = 200_000):
        self.buffer = deque(maxlen=capacity)

    def add_many(self, samples):
        self.buffer.extend(samples)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.stack([b[0] for b in batch])
        pis = np.stack([b[1] for b in batch])
        zs = np.array([b[2] for b in batch], dtype=np.float32)
        return states, pis, zs

    def save(self, path: str):
        """把缓冲内容存盘,供续训恢复(避免重启后重新暖机)。

        三个数组分别堆叠存入单个 .npz;空缓冲也写,便于无条件加载。
        """
        if self.buffer:
            states = np.stack([b[0] for b in self.buffer])
            pis = np.stack([b[1] for b in self.buffer])
            zs = np.array([b[2] for b in self.buffer], dtype=np.float32)
        else:
            states = np.empty((0,), dtype=np.float32)
            pis = np.empty((0,), dtype=np.float32)
            zs = np.empty((0,), dtype=np.float32)
        # np.savez 会给无后缀路径补 .npz;先写临时文件再原子替换到目标路径。
        tmp = path + ".tmp.npz"
        np.savez(tmp[:-4], states=states, pis=pis, zs=zs)
        os.replace(tmp, path)

    def load(self, path: str):
        """从 save() 写出的文件恢复缓冲内容(尊重当前 capacity 上限)。"""
        data = np.load(path)
        states, pis, zs = data["states"], data["pis"], data["zs"]
        self.buffer.clear()
        for i in range(len(zs)):
            self.buffer.append((states[i], pis[i], np.float32(zs[i])))
        return len(self.buffer)


def loss_fn(policy_logits, value, target_pi, target_z):
    """计算 AlphaZero 联合损失。返回 (total, policy_loss, value_loss)。"""
    # 价值:MSE
    value_loss = F.mse_loss(value, target_z)
    # 策略:交叉熵 -sum(pi * log_softmax(logits))
    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()
    total = value_loss + policy_loss
    return total, policy_loss, value_loss


class Trainer:
    def __init__(self, net, device, lr: float = 2e-3, weight_decay: float = 1e-4):
        self.net = net
        self.device = device
        self.net.to(device)  # 确保权重与输入同设备(GPU 训练必需)
        self.optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )

    def train_step(self, states, pis, zs):
        """单个 minibatch 的训练步,返回损失字典。"""
        self.net.train()
        x = torch.from_numpy(states).to(self.device)
        target_pi = torch.from_numpy(pis).to(self.device)
        target_z = torch.from_numpy(zs).to(self.device)

        policy_logits, value = self.net(x)
        total, p_loss, v_loss = loss_fn(policy_logits, value, target_pi, target_z)

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "total": total.item(),
            "policy": p_loss.item(),
            "value": v_loss.item(),
        }

    def train_epoch(self, buffer: ReplayBuffer, batch_size: int, steps: int):
        """从 buffer 采样训练若干步,返回平均损失。"""
        if len(buffer) < batch_size:
            return None
        agg = {"total": 0.0, "policy": 0.0, "value": 0.0}
        for _ in range(steps):
            states, pis, zs = buffer.sample(batch_size)
            losses = self.train_step(states, pis, zs)
            for k in agg:
                agg[k] += losses[k]
        return {k: v / steps for k, v in agg.items()}
