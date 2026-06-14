"""并行 AlphaZero 训练编排。

用多进程自我对弈(ParallelSelfPlay)持续产出对局,主进程消费样本并训练,
每轮训练后发布新权重给 workers。相比串行 pipeline,自我对弈吞吐随 worker
数量近线性提升。

用法:
  python -m xiangqi.pipeline_parallel --iterations 200 --games-per-iter 100 \
      --workers 8 --simulations 400 --channels 128 --blocks 10 \
      --batch-size 512 --worker-device cpu --train-device cuda

多卡建议:训练放 GPU(--train-device cuda),worker 推理放 CPU 避免争显存;
若 worker 也要用 GPU,可给不同 worker 指定不同卡(后续可扩展为按 rank 分配)。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .network import XiangqiNet
from .evaluator import pick_device
from .selfplay import SelfPlayConfig
from .train import ReplayBuffer, Trainer
from .parallel_selfplay import ParallelSelfPlay
from .pipeline import save_checkpoint, load_checkpoint, prune_checkpoints
from .metrics import MetricsLogger


def parse_args():
    p = argparse.ArgumentParser(description="并行 AlphaZero 中国象棋训练")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--games-per-iter", type=int, default=100)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--mcts-batch", type=int, default=8,
                   help="MCTS 批量推理的叶节点数(virtual loss),非训练批量")
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--buffer-size", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--temperature-moves", type=int, default=30)
    p.add_argument("--max-moves", type=int, default=300)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--train-device", type=str, default=None)
    p.add_argument("--worker-device", type=str, default="cpu")
    p.add_argument("--queue-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    # 最低 buffer 样本数,达到后才开始训练(让 buffer 先暖起来)。
    p.add_argument("--min-buffer", type=int, default=2000)
    p.add_argument("--tensorboard", action="store_true",
                   help="同时写 TensorBoard event(需安装 tensorboard)")
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="只保留最近 N 个 iter_*.pt(0=全部保留)")
    # 批量推理服务:中央 GPU 服务攒批前向,worker 走 RemoteEvaluator(CPU 搜索)。
    p.add_argument("--inference-server", action="store_true",
                   help="启用批量推理服务(集中 GPU 攒大 batch,提升 GPU 利用率)")
    p.add_argument("--server-device", type=str, default="cuda",
                   help="推理服务进程的设备(默认 cuda)")
    p.add_argument("--num-gpus", type=int, default=1,
                   help="推理服务卡数(每卡一个服务进程,worker 按 i%%G 路由)")
    p.add_argument("--max-infer-batch", type=int, default=256,
                   help="推理服务单次最大 batch")
    p.add_argument("--infer-timeout-ms", type=float, default=5.0,
                   help="攒批超时(ms),凑不满也发车防死锁")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.train_device)
    print(f"[训练设备] {device} | [worker 设备] {args.worker_device} "
          f"| [worker 数] {args.workers}")

    net = XiangqiNet(channels=args.channels, num_blocks=args.blocks)
    trainer = Trainer(net, device, lr=args.lr)
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        start_iter = load_checkpoint(args.resume, net, trainer, device)
        print(f"[恢复] 从 {args.resume} 第 {start_iter} 轮继续")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    buffer = ReplayBuffer(capacity=args.buffer_size)
    buffer_path = os.path.join(args.checkpoint_dir, "buffer.npz")
    # 续训时恢复 replay buffer,避免重新暖机、丢弃已生成的对局数据。
    if args.resume and os.path.exists(buffer_path):
        n = buffer.load(buffer_path)
        print(f"[恢复] 加载 replay buffer {n} 条样本")
    sp_config = SelfPlayConfig(
        num_simulations=args.simulations,
        temperature_moves=args.temperature_moves,
        max_moves=args.max_moves,
        batch_size=args.mcts_batch,
    )
    net_config = {"channels": args.channels, "blocks": args.blocks}
    weights_path = os.path.join(args.checkpoint_dir, "worker_weights.pt")

    parallel = ParallelSelfPlay(
        net_config=net_config,
        sp_config=sp_config,
        weights_path=weights_path,
        num_workers=args.workers,
        worker_device=args.worker_device,
        queue_size=args.queue_size,
        seed=args.seed,
        use_inference_server=args.inference_server,
        server_device=args.server_device,
        max_infer_batch=args.max_infer_batch,
        infer_timeout_ms=args.infer_timeout_ms,
        num_gpus=args.num_gpus,
    )
    # 先发布初始权重再启动 worker,确保 worker 一上来就用当前网络。
    parallel.publish_weights(net)
    parallel.start()
    print("[启动] worker 已开始自我对弈")
    logger = MetricsLogger(args.checkpoint_dir, tensorboard=args.tensorboard)

    try:
        for it in range(start_iter, start_iter + args.iterations):
            t0 = time.time()

            # --- 收集本轮对局样本 ---
            samples, results, total_moves = parallel.collect_games(args.games_per_iter)
            buffer.add_many(samples)
            sp_time = time.time() - t0

            # --- 训练 ---
            t1 = time.time()
            if len(buffer) >= args.min_buffer:
                losses = trainer.train_epoch(buffer, args.batch_size, args.train_steps)
            else:
                losses = None
            train_time = time.time() - t1

            # --- 发布新权重给 workers ---
            if losses is not None:
                parallel.publish_weights(net)

            avg_moves = total_moves / max(1, args.games_per_iter)
            loss_str = (f"total={losses['total']:.3f} policy={losses['policy']:.3f} "
                        f"value={losses['value']:.3f}") if losses else \
                       f"buffer={len(buffer)}<{args.min_buffer},暖机中"
            print(f"[轮 {it}] 收集 {args.games_per_iter} 局 "
                  f"(红{results['red_win']}/黑{results['black_win']}/和{results['draw']}, "
                  f"均步{avg_moves:.0f}) | buffer={len(buffer)} | {loss_str} | "
                  f"收集{sp_time:.0f}s 训练{train_time:.0f}s")

            logger.log(it + 1, {
                "loss_total": losses["total"] if losses else "",
                "loss_policy": losses["policy"] if losses else "",
                "loss_value": losses["value"] if losses else "",
                "red_win": results["red_win"],
                "black_win": results["black_win"],
                "draw": results["draw"],
                "avg_moves": round(avg_moves, 1),
                "buffer": len(buffer),
                "collect_sec": round(sp_time, 1),
                "train_sec": round(train_time, 1),
            })

            # --- checkpoint ---
            ckpt_path = os.path.join(args.checkpoint_dir, f"iter_{it+1:04d}.pt")
            save_checkpoint(ckpt_path, net, trainer, it + 1)
            save_checkpoint(os.path.join(args.checkpoint_dir, "latest.pt"),
                            net, trainer, it + 1)
            buffer.save(buffer_path)  # 持久化 buffer,支持断点续训
            prune_checkpoints(args.checkpoint_dir, args.keep_checkpoints)
    finally:
        print("[停止] 正在关闭 worker...")
        parallel.stop()
        logger.close()

    print("[完成] 训练结束")


if __name__ == "__main__":
    main()
