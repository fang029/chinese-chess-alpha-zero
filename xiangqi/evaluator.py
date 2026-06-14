"""神经网络评估器:封装 XiangqiNet 的推理,供 MCTS 调用。

把 GameState 编码、设备搬运、掩码 softmax 等细节集中在此,MCTS 只需调用
evaluate(state) -> (policy_dict, value)。
"""

from __future__ import annotations

import numpy as np
import torch

from .game import GameState
from .network import XiangqiNet, masked_policy
from .state_encoder import encode_state
from . import encoding


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Evaluator:
    """对单个或一批 GameState 做策略-价值评估。"""

    def __init__(self, net: XiangqiNet, device: torch.device | None = None):
        self.net = net
        self.device = device or pick_device()
        self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def evaluate(self, state: GameState):
        """返回 (policy, value)。

        policy: dict[move -> prob],仅含合法走法,已归一化。
        value:  float,当前走方视角的局面评估 [-1, 1]。
        """
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0

        x = torch.from_numpy(encode_state(state)).unsqueeze(0).to(self.device)
        mask_list = encoding.legal_action_mask(legal)
        mask = torch.tensor(mask_list, dtype=torch.float32,
                            device=self.device).unsqueeze(0)

        logits, value = self.net(x)
        probs = masked_policy(logits, mask)[0].cpu().numpy()

        policy = {m: float(probs[encoding.move_to_index(m)]) for m in legal}
        # 归一化(掩码 softmax 已保证,但防数值误差)
        total = sum(policy.values())
        if total > 0:
            policy = {m: p / total for m, p in policy.items()}
        else:
            # 退化情形:均匀分布
            u = 1.0 / len(legal)
            policy = {m: u for m in legal}

        return policy, float(value[0].item())

    @torch.no_grad()
    def evaluate_batch(self, states: list):
        """批量评估,返回 [(policy, value), ...]。用于并行自我对弈加速。"""
        if not states:
            return []
        legals = [s.legal_moves() for s in states]
        xs = np.stack([encode_state(s) for s in states])
        x = torch.from_numpy(xs).to(self.device)
        masks = torch.tensor(
            [encoding.legal_action_mask(lm) for lm in legals],
            dtype=torch.float32, device=self.device,
        )
        logits, values = self.net(x)
        probs = masked_policy(logits, masks).cpu().numpy()

        results = []
        for i, legal in enumerate(legals):
            if not legal:
                results.append(({}, 0.0))
                continue
            pol = {m: float(probs[i, encoding.move_to_index(m)]) for m in legal}
            total = sum(pol.values())
            if total > 0:
                pol = {m: p / total for m, p in pol.items()}
            else:
                u = 1.0 / len(legal)
                pol = {m: u for m in legal}
            results.append((pol, float(values[i].item())))
        return results
