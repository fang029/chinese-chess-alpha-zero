"""人类棋谱监督预训练:用棋谱里的人类走法做策略监督,加速冷启动。

偏离纯 AlphaZero(后者完全从零自我对弈),但实用:先让网络模仿人类走法,
得到一个有基本棋感的初始权重,再转入自我对弈强化。

棋谱格式:每局是一串中文着法(如 "炮二平五","马8进7"),红黑交替。
从每个局面生成一条样本 (state_tensor, target_move_index, z):
  - target_move_index: 该局面人类所走着法在 2550 动作空间的索引(策略目标,
    one-hot 交叉熵)。
  - z: 该局面轮走方视角的最终对局结果(+1/-1/0),作为价值目标。

只解析能合法落地的着法;遇到无法解析或非法的着法即终止该局(返回已生成的
样本),避免脏数据污染训练。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .game import GameState
from .move_notation import parse_move
from .iccs_notation import parse_iccs
from .state_encoder import encode_state
from . import encoding
from .constants import RED


def _parse(board, text, to_move, notation):
    """按 notation 选择解析器。中文记谱需 board+to_move 推断,ICCS 是纯坐标。"""
    if notation == "iccs":
        return parse_iccs(text)
    return parse_move(board, text, to_move)


def game_to_samples(moves: list, result: str | None = None,
                    notation: str = "chinese"):
    """把一局着法序列转成监督样本 [(state, action_index, to_move), ...]。

    notation: "chinese"(炮二平五)或 "iccs"(C3-C4)。
    result: "red_win"/"black_win"/"draw";None 表示未知(价值目标记 0)。
    遇到无法解析/非法着法即停止,返回此前已成功的样本。
    """
    g = GameState()
    samples = []
    for text in moves:
        try:
            move = _parse(g.board, text, g.to_move, notation)
        except ValueError:
            break
        legal = set(map(tuple, g.legal_moves()))
        if tuple(move) not in legal:
            break
        idx = encoding.move_to_index(move)
        samples.append((encode_state(g), idx, g.to_move))
        g.push(move)
    return samples, g


def _z_for(result: str | None, to_move: int) -> float:
    if result == "red_win":
        z_red = 1.0
    elif result == "black_win":
        z_red = -1.0
    else:
        z_red = 0.0
    return z_red if to_move == RED else -z_red


def build_dataset(games: list, notation: str = "chinese"):
    """把多局棋谱转成训练张量。

    games: [{"moves": [着法...], "result": "red_win"/...}, ...]
    notation: "chinese"(炮二平五)或 "iccs"(C3-C4)。
    返回 (states[N,15,10,9], action_idx[N], zs[N]) 三个 numpy 数组。
    """
    all_states, all_idx, all_z = [], [], []
    for game in games:
        moves = game.get("moves", [])
        result = game.get("result")
        samples, _ = game_to_samples(moves, result, notation=notation)
        for state, idx, to_move in samples:
            all_states.append(state)
            all_idx.append(idx)
            all_z.append(_z_for(result, to_move))
    if not all_states:
        return (np.empty((0, 15, 10, 9), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32))
    return (np.stack(all_states),
            np.array(all_idx, dtype=np.int64),
            np.array(all_z, dtype=np.float32))


def pretrain(net, states, action_idx, zs, device, *, epochs=10,
             batch_size=256, lr=1e-3, value_weight=1.0, log_every=1):
    """监督预训练:策略用交叉熵(对人类走法),价值用 MSE(对对局结果)。

    返回每个 epoch 的平均损失列表。
    """
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    n = len(states)
    states_t = torch.from_numpy(states)
    idx_t = torch.from_numpy(action_idx)
    z_t = torch.from_numpy(zs)

    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        agg = {"total": 0.0, "policy": 0.0, "value": 0.0}
        nb = 0
        for start in range(0, n, batch_size):
            sel = perm[start:start + batch_size]
            x = states_t[sel].to(device)
            target_idx = idx_t[sel].to(device)
            target_z = z_t[sel].to(device)

            logits, value = net(x)
            policy_loss = F.cross_entropy(logits, target_idx)
            value_loss = F.mse_loss(value, target_z)
            total = policy_loss + value_weight * value_loss

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            agg["total"] += total.item()
            agg["policy"] += policy_loss.item()
            agg["value"] += value_loss.item()
            nb += 1

        avg = {k: v / max(1, nb) for k, v in agg.items()}
        history.append(avg)
        if log_every and (epoch + 1) % log_every == 0:
            print(f"[预训练 epoch {epoch+1}/{epochs}] "
                  f"total={avg['total']:.3f} policy={avg['policy']:.3f} "
                  f"value={avg['value']:.3f}")
    return history


def save_pretrained(path, net):
    """存为与 pipeline 兼容的 checkpoint(iteration=0,无优化器状态)。

    pipeline 的 load_checkpoint 只在存在 'optimizer' 键时加载它,这里省略,
    续训时优化器从头初始化即可(预训练用 Adam,正式训练用 SGD,本就不同)。
    """
    torch.save({
        "iteration": 0,
        "model": net.state_dict(),
        "channels": net.stem[0].out_channels,
    }, path)


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="人类棋谱监督预训练")
    p.add_argument("--games", required=True,
                   help="棋谱 JSON 文件:[{\"moves\":[...],\"result\":\"...\"}, ...]")
    p.add_argument("--out", default="checkpoints/pretrained.pt")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--notation", choices=["chinese", "iccs"], default="chinese",
                   help="棋谱着法记法:chinese(炮二平五)或 iccs(C3-C4)")
    return p.parse_args()


def main():
    import json
    import os
    from .network import XiangqiNet
    from .evaluator import pick_device

    args = parse_args()
    device = pick_device(args.device)
    with open(args.games, "r", encoding="utf-8") as f:
        games = json.load(f)

    states, idx, zs = build_dataset(games, notation=args.notation)
    print(f"[预训练] 载入 {len(games)} 局,生成 {len(states)} 条样本,设备 {device}")
    if len(states) == 0:
        print("[预训练] 无有效样本,退出")
        return

    net = XiangqiNet(channels=args.channels, num_blocks=args.blocks)
    pretrain(net, states, idx, zs, device,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_pretrained(args.out, net)
    print(f"[预训练] 已保存 {args.out};正式训练用 --resume {args.out} 接续")


if __name__ == "__main__":
    main()
