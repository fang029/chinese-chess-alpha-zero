"""AlphaZero 主训练循环编排。

迭代:自我对弈生成数据 -> 存入 replay buffer -> 训练网络 -> 存 checkpoint。

用法:
  python -m xiangqi.pipeline --iterations 100 --games-per-iter 50 \
      --simulations 200 --channels 128 --blocks 10

可调小参数在 CPU/MPS 上跑通流程;正式训练用云端多卡并加大规模。
多卡可在此基础上接入 DistributedDataParallel 与多进程自我对弈(见文末注释)。
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

from .network import XiangqiNet
from .evaluator import Evaluator, pick_device
from .selfplay import SelfPlayConfig, play_game
from .train import ReplayBuffer, Trainer
from .arena import ArenaConfig, compare
from .metrics import MetricsLogger


def parse_args():
    p = argparse.ArgumentParser(description="AlphaZero 中国象棋训练")
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--games-per-iter", type=int, default=50)
    p.add_argument("--simulations", type=int, default=200)
    p.add_argument("--mcts-batch", type=int, default=8,
                   help="MCTS 批量推理的叶节点数(virtual loss),非训练批量")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temperature-moves", type=int, default=30)
    p.add_argument("--max-moves", type=int, default=300)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true",
                   help="同时写 TensorBoard event(需安装 tensorboard)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="只保留最近 N 个 iter_*.pt(0=全部保留)")
    # 评估门控(arena):新网络需在与上一代对弈中达标才晋级为自弈网络。
    p.add_argument("--gating", action="store_true",
                   help="启用评估门控:训练后让新网络与上一代对弈,达标才晋级")
    p.add_argument("--eval-interval", type=int, default=1,
                   help="每隔多少轮做一次评估门控(默认每轮)")
    p.add_argument("--eval-games", type=int, default=20,
                   help="门控对弈局数")
    p.add_argument("--eval-simulations", type=int, default=None,
                   help="门控对弈每步模拟数(默认沿用 --simulations)")
    p.add_argument("--eval-threshold", type=float, default=0.55,
                   help="晋级所需得分率(胜1和0.5),默认 0.55")
    return p.parse_args()


def save_checkpoint(path, net, trainer, iteration):
    torch.save({
        "iteration": iteration,
        "model": net.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "channels": net.stem[0].out_channels,
    }, path)


def prune_checkpoints(checkpoint_dir, keep):
    """只保留最近 keep 个 iter_*.pt,删除更早的,避免长跑堆满磁盘。

    keep <= 0 表示不清理(全部保留)。latest.pt/champion.pt/buffer.npz 等
    非 iter_ 前缀文件不受影响。
    """
    import glob
    import re
    if keep is None or keep <= 0:
        return
    pattern = os.path.join(checkpoint_dir, "iter_*.pt")
    files = glob.glob(pattern)
    # 按文件名中的迭代号排序(iter_0007.pt -> 7),而非字典序。
    def iter_num(p):
        m = re.search(r"iter_(\d+)\.pt$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    files.sort(key=iter_num)
    for old in files[:-keep]:
        try:
            os.remove(old)
        except OSError:
            pass


def load_checkpoint(path, net, trainer, device):
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt["model"])
    if trainer is not None and "optimizer" in ckpt:
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("iteration", 0)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"[设备] {device}")

    net = XiangqiNet(channels=args.channels, num_blocks=args.blocks)
    trainer = Trainer(net, device, lr=args.lr)
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(args.resume, net, trainer, device)
        print(f"[恢复] 从 {args.resume} 第 {start_iter} 轮继续")

    # 评估门控:champion 网络负责自我对弈(数据生成),net 是被训练的
    # challenger。每个 eval-interval 让 challenger 与 champion 对弈,达标才把
    # champion 同步为 challenger,否则把 challenger 回退到 champion(丢弃这段
    # 训练,避免坏权重污染数据)。不启用门控时二者同一,即始终用最新网络。
    if args.gating:
        champion_net = XiangqiNet(channels=args.channels, num_blocks=args.blocks)
        champion_net.load_state_dict(net.state_dict())
        selfplay_eval = Evaluator(champion_net, device=device)
        challenger_eval = Evaluator(net, device=device)
        arena_config = ArenaConfig(
            num_games=args.eval_games,
            num_simulations=args.eval_simulations or args.simulations,
            win_threshold=args.eval_threshold,
            max_moves=args.max_moves,
            batch_size=args.mcts_batch,
        )
        print(f"[门控] 启用,每 {args.eval_interval} 轮评估 "
              f"{args.eval_games} 局,晋级阈值 {args.eval_threshold}")
    else:
        champion_net = net
        selfplay_eval = Evaluator(net, device=device)
        challenger_eval = None

    evaluator = selfplay_eval
    buffer = ReplayBuffer(capacity=args.buffer_size)
    sp_config = SelfPlayConfig(
        num_simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        max_moves=args.max_moves,
        batch_size=args.mcts_batch,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    buffer_path = os.path.join(args.checkpoint_dir, "buffer.npz")
    if args.resume and os.path.exists(buffer_path):
        n = buffer.load(buffer_path)
        print(f"[恢复] 加载 replay buffer {n} 条样本")
    rng = np.random.default_rng(args.seed)
    logger = MetricsLogger(args.checkpoint_dir, tensorboard=args.tensorboard)

    for it in range(start_iter, start_iter + args.iterations):
        t0 = time.time()

        # --- 自我对弈(用 champion 网络生成数据)---
        champion_net.eval()
        results = {"red_win": 0, "black_win": 0, "draw": 0}
        total_moves = 0
        for g in range(args.games_per_iter):
            samples, result, n_moves = play_game(evaluator, sp_config, rng)
            buffer.add_many(samples)
            results[result if result in results else "draw"] = \
                results.get(result if result in results else "draw", 0) + 1
            total_moves += n_moves
        sp_time = time.time() - t0

        # --- 训练 ---
        t1 = time.time()
        losses = trainer.train_epoch(buffer, args.batch_size, args.train_steps)
        train_time = time.time() - t1

        avg_moves = total_moves / max(1, args.games_per_iter)
        loss_str = (f"total={losses['total']:.3f} policy={losses['policy']:.3f} "
                    f"value={losses['value']:.3f}") if losses else "buffer不足,跳过训练"
        print(f"[轮 {it}] 对弈 {args.games_per_iter} 局 "
              f"(红胜{results['red_win']}/黑胜{results['black_win']}/和{results['draw']}, "
              f"均步{avg_moves:.0f}) | buffer={len(buffer)} | {loss_str} | "
              f"自弈{sp_time:.0f}s 训练{train_time:.0f}s")

        logger.log(it + 1, {
            "loss_total": losses["total"] if losses else "",
            "loss_policy": losses["policy"] if losses else "",
            "loss_value": losses["value"] if losses else "",
            "red_win": results["red_win"],
            "black_win": results["black_win"],
            "draw": results["draw"],
            "avg_moves": round(avg_moves, 1),
            "buffer": len(buffer),
            "selfplay_sec": round(sp_time, 1),
            "train_sec": round(train_time, 1),
        })

        # --- 评估门控 ---
        promoted = None
        if args.gating and (it + 1) % args.eval_interval == 0:
            net.eval()
            champion_net.eval()
            res = compare(challenger_eval, selfplay_eval, arena_config)
            promoted = res["promote"]
            tag = "晋级" if promoted else "保留旧网络"
            print(f"  [门控] challenger 胜{res['wins']}/负{res['losses']}/"
                  f"和{res['draws']} 得分率{res['score']:.2f} -> {tag}")
            if promoted:
                champion_net.load_state_dict(net.state_dict())
            else:
                # 回退:丢弃这段训练,challenger 重置为现任 champion。
                net.load_state_dict(champion_net.state_dict())

        # --- checkpoint ---
        # 门控启用时,latest/champion 始终指向受信任的对弈网络(champion),
        # 另存 challenger 便于排查;未启用时二者相同。
        ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
        save_checkpoint(ckpt_path, net, trainer, it + 1)
        latest = os.path.join(args.checkpoint_dir, "latest.pt")
        if args.gating:
            champ_path = os.path.join(args.checkpoint_dir, "champion.pt")
            save_checkpoint(champ_path, champion_net, trainer, it + 1)
            save_checkpoint(latest, champion_net, trainer, it + 1)
        else:
            save_checkpoint(latest, net, trainer, it + 1)
        buffer.save(buffer_path)  # 持久化 buffer,支持断点续训
        prune_checkpoints(args.checkpoint_dir, args.keep_checkpoints)

    logger.close()
    print("[完成] 训练结束")


# ---- 关于多卡/云端扩展(供后续实现参考)----
# 1. 自我对弈并行:用 torch.multiprocessing 启多个 worker 进程,各持网络副本
#    生成对局,经队列汇总到主进程的 ReplayBuffer。自我对弈是吞吐瓶颈。
# 2. 训练并行:用 DistributedDataParallel 包装 net,多卡数据并行训练。
# 3. 评估门控:已实现,见 arena.py 与 --gating 系列参数。每隔若干轮让新网络与
#    上一代对弈,得分率达标才晋级为自弈网络(AlphaGo Zero 做法;AlphaZero 简化
#    为始终用最新网络,即不加 --gating)。
#    (AlphaGo Zero 做法;AlphaZero 简化为始终用最新网络)。


if __name__ == "__main__":
    main()
