"""对弈接口:用训练好的模型选子,支持人机/机机对弈与命令行对局。

用法:
  python -m xiangqi.play --checkpoint checkpoints/latest.pt --simulations 400
  (红方为人类输入,黑方为 AI;输入格式 "r1 c1 r2 c2",如 "0 1 2 2")
"""

from __future__ import annotations

import argparse

import torch

from .game import GameState
from .network import XiangqiNet
from .evaluator import Evaluator, pick_device
from .mcts import MCTS, action_probabilities, advance_root
from .constants import RED, BLACK


class AIPlayer:
    """基于 MCTS + 网络的 AI 选子器。"""

    def __init__(self, evaluator, simulations: int = 400, c_puct: float = 1.5,
                 batch_size: int = 8):
        self.mcts = MCTS(evaluator, c_puct=c_puct, batch_size=batch_size)
        self.simulations = simulations
        self.root = None  # 跨步复用的搜索树根

    def select_move(self, state: GameState):
        """返回 AI 认为最佳的走法(贪心取访问最多者,不加噪声)。"""
        self.root = self.mcts.run(state, self.simulations,
                                  add_noise=False, root=self.root)
        pi = action_probabilities(self.root, temperature=0.0)
        return max(pi.items(), key=lambda kv: kv[1])[0]

    def advance(self, move):
        """对局推进一步后调用:把对应子节点提升为新根以复用子树。

        双方走子后都应调用,使 AI 的搜索树始终对齐当前局面。
        """
        if self.root is not None:
            self.root = advance_root(self.root, move)


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    channels = ckpt.get("channels", 128)
    # blocks 数无法从 state_dict 直接得知,按常见默认;若不匹配会在 load 时报错。
    # 为稳健起见,尝试从权重推断残差块数量。
    num_blocks = _infer_num_blocks(ckpt["model"])
    net = XiangqiNet(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(ckpt["model"])
    return net


def _infer_num_blocks(state_dict) -> int:
    """从 state_dict 的键推断残差块数量。"""
    idxs = set()
    for k in state_dict:
        if k.startswith("res_blocks."):
            idxs.add(int(k.split(".")[1]))
    return (max(idxs) + 1) if idxs else 10


def _parse_human_move(text: str):
    parts = text.strip().split()
    if len(parts) != 4:
        return None
    try:
        r1, c1, r2, c2 = (int(p) for p in parts)
    except ValueError:
        return None
    return ((r1, c1), (r2, c2))


def human_vs_ai(checkpoint_path, simulations, human_side=RED, device=None):
    device = device or pick_device()
    net = load_model(checkpoint_path, device)
    evaluator = Evaluator(net, device=device)
    ai = AIPlayer(evaluator, simulations=simulations)

    state = GameState()
    print(state)
    print()

    while not state.is_terminal():
        if state.to_move == human_side:
            legal = set(map(tuple, state.legal_moves()))
            move = None
            while move not in legal:
                raw = input("你的走法 (r1 c1 r2 c2),q 退出: ")
                if raw.strip().lower() == "q":
                    return
                move = _parse_human_move(raw)
                if move not in legal:
                    print("非法走法,请重输。")
            state.push(move)
            ai.advance(move)
        else:
            print("AI 思考中...")
            move = ai.select_move(state)
            print(f"AI 走子: {move[0]} -> {move[1]}")
            state.push(move)
            ai.advance(move)

        print(state)
        print()

    print(f"对局结束: {state.result()}")


def parse_args():
    p = argparse.ArgumentParser(description="与中国象棋 AI 对弈")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--human-side", choices=["red", "black"], default="red")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    side = RED if args.human_side == "red" else BLACK
    human_vs_ai(args.checkpoint, args.simulations, human_side=side, device=device)


if __name__ == "__main__":
    main()
