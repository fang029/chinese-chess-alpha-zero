"""DDP 数据并行训练循环(多卡/多进程)。

用 torchrun 拉起,每个 rank 一块卡:

    torchrun --nproc_per_node=4 -m xiangqi.pipeline_ddp \\
        --iterations 200 --games-per-iter 25 --simulations 400 \\
        --channels 128 --blocks 10 --batch-size 512

设计:每个 rank 各自自我对弈生成数据、各自从本地 buffer 采样训练,DDP 在
反向传播时 all-reduce 梯度,等价于把有效 batch 放大 world_size 倍的数据并行。
games-per-iter 是每个 rank 的局数,总吞吐 = world_size × games-per-iter。
只有 rank 0 落盘 checkpoint 与打印日志。单进程(无 torchrun)下退化为普通
单卡训练,便于本地调试。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .network import XiangqiNet
from .evaluator import Evaluator
from .selfplay import SelfPlayConfig, play_game
from .train import ReplayBuffer, Trainer
from .pipeline import save_checkpoint, load_checkpoint, prune_checkpoints
from .metrics import MetricsLogger
from . import distributed as D


def parse_args():
    p = argparse.ArgumentParser(description="DDP 多卡 AlphaZero 中国象棋训练")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=25,
                   help="每个 rank 每轮自我对弈局数(总数 = world_size × 此值)")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--mcts-batch", type=int, default=8)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=200)
    p.add_argument("--buffer-size", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temperature-moves", type=int, default=30)
    p.add_argument("--max-moves", type=int, default=300)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensorboard", action="store_true",
                   help="同时写 TensorBoard event(需安装 tensorboard,仅 rank0)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="只保留最近 N 个 iter_*.pt(0=全部保留)")
    return p.parse_args()


def main():
    args = parse_args()
    device = D.setup()
    rank, world = D.get_rank(), D.get_world_size()
    # 各 rank 用不同种子,避免所有进程生成相同对局。
    seed = args.seed + rank
    np.random.seed(seed)
    torch.manual_seed(seed)

    if D.is_main_process():
        print(f"[DDP] world_size={world} | device={device}")

    net = XiangqiNet(channels=args.channels, num_blocks=args.blocks).to(device)
    trainer = Trainer(net, device, lr=args.lr)
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(args.resume, net, trainer, device)
        if D.is_main_process():
            print(f"[恢复] 从 {args.resume} 第 {start_iter} 轮继续")

    # DDP 包装:仅在真正分布式时启用;单进程直接用裸 net。
    if D.is_distributed():
        ddp_ids = [D.get_local_rank()] if device.type == "cuda" else None
        ddp_net = DDP(net, device_ids=ddp_ids)
        trainer.net = ddp_net  # 训练走 DDP 包装(梯度 all-reduce)
        train_module = ddp_net
    else:
        train_module = net

    # 自我对弈用未包装的 net(只推理,不需要 DDP),与训练共享同一份权重。
    evaluator = Evaluator(net, device=device)

    if D.is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    buffer_path = os.path.join(args.checkpoint_dir, f"buffer_rank{rank}.npz")
    if args.resume and os.path.exists(buffer_path):
        n = buffer.load(buffer_path)
        if D.is_main_process():
            print(f"[恢复] rank{rank} 加载 buffer {n} 条")

    sp_config = SelfPlayConfig(
        num_simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        max_moves=args.max_moves,
        batch_size=args.mcts_batch,
    )
    rng = np.random.default_rng(seed)
    # 仅 rank0 落盘指标,避免多进程争写同一 CSV。
    logger = MetricsLogger(args.checkpoint_dir,
                           tensorboard=args.tensorboard) if D.is_main_process() else None

    D.barrier()
    for it in range(start_iter, start_iter + args.iterations):
        t0 = time.time()

        # --- 自我对弈(每 rank 各自生成)---
        net.eval()
        results = {"red_win": 0, "black_win": 0, "draw": 0}
        total_moves = 0
        for _ in range(args.games_per_iter):
            samples, result, n_moves = play_game(evaluator, sp_config, rng)
            buffer.add_many(samples)
            key = result if result in results else "draw"
            results[key] += 1
            total_moves += n_moves
        sp_time = time.time() - t0

        # --- 训练(DDP 自动 all-reduce 梯度)---
        t1 = time.time()
        losses = _train_epoch(train_module, trainer, buffer,
                              args.batch_size, args.train_steps, device)
        train_time = time.time() - t1

        if D.is_main_process():
            avg_moves = total_moves / max(1, args.games_per_iter)
            loss_str = (f"total={losses['total']:.3f} policy={losses['policy']:.3f} "
                        f"value={losses['value']:.3f}") if losses else "buffer不足,跳过"
            print(f"[轮 {it}] rank0 对弈 {args.games_per_iter} 局 "
                  f"(红{results['red_win']}/黑{results['black_win']}/和{results['draw']}, "
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

        # --- checkpoint(仅 rank 0)---
        if D.is_main_process():
            ckpt = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
            save_checkpoint(ckpt, net, trainer, it + 1)
            save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"),
                            net, trainer, it + 1)
            prune_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
        buffer.save(buffer_path)  # 各 rank 存各自 buffer
        D.barrier()  # 确保 rank0 存完再进入下一轮

    D.cleanup()
    if D.is_main_process():
        logger.close()
        print("[完成] 训练结束")


def _train_epoch(train_module, trainer, buffer, batch_size, steps, device):
    """与 Trainer.train_epoch 等价,但前向走 DDP 包装的 module。"""
    from .train import loss_fn
    if len(buffer) < batch_size:
        return None
    agg = {"total": 0.0, "policy": 0.0, "value": 0.0}
    train_module.train()
    for _ in range(steps):
        states, pis, zs = buffer.sample(batch_size)
        x = torch.from_numpy(states).to(device)
        target_pi = torch.from_numpy(pis).to(device)
        target_z = torch.from_numpy(zs).to(device)
        logits, value = train_module(x)
        total, p_loss, v_loss = loss_fn(logits, value, target_pi, target_z)
        trainer.optimizer.zero_grad()
        total.backward()
        trainer.optimizer.step()
        agg["total"] += total.item()
        agg["policy"] += p_loss.item()
        agg["value"] += v_loss.item()
    return {k: v / steps for k, v in agg.items()}


if __name__ == "__main__":
    main()
