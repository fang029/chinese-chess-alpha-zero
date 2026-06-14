"""自我对弈:用当前网络 + MCTS 对弈,生成训练样本。

每个样本为 (encoded_state, pi_vector, z):
  encoded_state: 状态张量 (np.float32)
  pi_vector:     长度 ACTION_SIZE 的 MCTS 改进策略(访问分布)
  z:             该状态轮走方视角的最终对局结果 (+1 胜 / -1 负 / 0 和)

温度调度:开局前若干步用 temperature=1 鼓励多样性,之后降到接近 0。
"""

from __future__ import annotations

import numpy as np

from .game import GameState
from .mcts import MCTS, action_probabilities, advance_root
from .state_encoder import encode_state
from . import encoding
from .constants import RED, BLACK


class SelfPlayConfig:
    def __init__(self,
                 num_simulations: int = 200,
                 temperature_moves: int = 30,
                 max_moves: int = 300,
                 c_puct: float = 1.5,
                 batch_size: int = 8):
        self.num_simulations = num_simulations
        self.temperature_moves = temperature_moves  # 前多少步用高温采样
        self.max_moves = max_moves                  # 超过则判和,避免无限局
        self.c_puct = c_puct
        self.batch_size = batch_size                # MCTS 批量推理叶节点数


def _pi_to_vector(pi: dict) -> np.ndarray:
    """把 {move: prob} 转为长度 ACTION_SIZE 的稠密向量。"""
    vec = np.zeros(encoding.ACTION_SIZE, dtype=np.float32)
    for move, prob in pi.items():
        vec[encoding.move_to_index(move)] = prob
    return vec


def _sample_move(pi: dict, rng):
    """按概率分布 pi 采样一个走法。"""
    moves = list(pi.keys())
    probs = np.array([pi[m] for m in moves], dtype=np.float64)
    probs /= probs.sum()
    idx = rng.choice(len(moves), p=probs)
    return moves[idx]


def play_game(evaluator, config: SelfPlayConfig, rng):
    """进行一局自我对弈,返回训练样本列表 [(state_tensor, pi_vec, z), ...]。"""
    mcts = MCTS(evaluator, c_puct=config.c_puct, batch_size=config.batch_size)
    state = GameState()

    # 暂存 (state_tensor, pi_vec, to_move),终局后再回填 z。
    trajectory = []
    move_count = 0
    root = None  # 复用上一步搜索的子树作为本步搜索起点

    while not state.is_terminal() and move_count < config.max_moves:
        root = mcts.run(state, config.num_simulations,
                        add_noise=True, rng=rng, root=root)

        temperature = 1.0 if move_count < config.temperature_moves else 1e-3
        pi = action_probabilities(root, temperature=temperature)

        trajectory.append((
            encode_state(state),
            _pi_to_vector(pi),
            state.to_move,
        ))

        move = _sample_move(pi, rng)
        state.push(move)
        move_count += 1
        root = advance_root(root, move)  # 提升子节点为新根,保留其子树统计

    # 计算最终结果(从红方视角的 z_red)
    result = state.result()
    if result == "red_win":
        z_red = 1.0
    elif result == "black_win":
        z_red = -1.0
    else:
        z_red = 0.0  # 和棋或超步

    samples = []
    for state_tensor, pi_vec, to_move in trajectory:
        z = z_red if to_move == RED else -z_red
        samples.append((state_tensor, pi_vec, np.float32(z)))

    return samples, result, move_count
