"""并行预编码:把棋谱一次性编码成磁盘分片,供预训练多 epoch 快速复用。

单进程编码 14 万局要 ~7 小时且每 epoch 重编码一遍,浪费 31/32 核。本模块用
多进程把编码分摊到所有核(快几十倍),结果存成若干 .npz 分片到磁盘;预训练
直接 mmap/加载分片做多轮训练,不再重复编码。

用法:
  python -m xiangqi.encode_dataset --games data/all_games.json \\
      --notation iccs --out-dir data/encoded --workers 30
"""

from __future__ import annotations

import os
import numpy as np

from .pretrain import game_to_samples, _z_for


def _encode_games(args):
    """worker:编码一批棋谱,返回 (states, idx, zs) 或 None。"""
    games_chunk, notation = args
    states, idxs, zs = [], [], []
    for game in games_chunk:
        result = game.get("result")
        samples, _ = game_to_samples(game.get("moves", []), result,
                                     notation=notation)
        for state, idx, to_move in samples:
            states.append(state)
            idxs.append(idx)
            zs.append(_z_for(result, to_move))
    if not states:
        return None
    return (np.stack(states),
            np.array(idxs, dtype=np.int64),
            np.array(zs, dtype=np.float32))


def encode_to_shards(games, notation, out_dir, workers=None,
                     games_per_task=128, shard_samples=200_000):
    """多进程编码全部棋谱,按样本数切分写成 shard_XXX.npz。

    返回 (总样本数, 分片数)。
    """
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    workers = workers or (os.cpu_count() or 4)

    # 切成小任务派给进程池。
    tasks = [(games[i:i + games_per_task], notation)
             for i in range(0, len(games), games_per_task)]

    buf_states, buf_idx, buf_z = [], [], []
    buffered = 0
    shard_id = 0
    total = 0

    def flush():
        nonlocal shard_id, buf_states, buf_idx, buf_z, buffered
        if buffered == 0:
            return
        path = os.path.join(out_dir, f"shard_{shard_id:04d}.npz")
        np.savez(path,
                 states=np.concatenate(buf_states),
                 idx=np.concatenate(buf_idx),
                 zs=np.concatenate(buf_z))
        shard_id += 1
        buf_states, buf_idx, buf_z = [], [], []
        buffered = 0

    with mp.get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(_encode_games, tasks, chunksize=1):
            if res is None:
                continue
            s, i, z = res
            buf_states.append(s)
            buf_idx.append(i)
            buf_z.append(z)
            buffered += len(s)
            total += len(s)
            if buffered >= shard_samples:
                flush()
                print(f"  已编码 {total} 样本,写出 {shard_id} 个分片", flush=True)
    flush()
    return total, shard_id


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="并行预编码棋谱为磁盘分片")
    p.add_argument("--games", required=True)
    p.add_argument("--notation", choices=["iccs", "chinese"], default="iccs")
    p.add_argument("--out-dir", default="data/encoded")
    p.add_argument("--workers", type=int, default=0, help="0=用全部核")
    p.add_argument("--games-per-task", type=int, default=128)
    p.add_argument("--shard-samples", type=int, default=200_000)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main():
    import json
    import time
    args = parse_args()
    with open(args.games, "r", encoding="utf-8") as f:
        games = json.load(f)
    if args.limit:
        games = games[:args.limit]
    workers = args.workers or (os.cpu_count() or 4)
    print(f"[编码] {len(games)} 局,{workers} 进程并行 -> {args.out_dir}")
    t = time.time()
    total, shards = encode_to_shards(
        games, args.notation, args.out_dir, workers=workers,
        games_per_task=args.games_per_task, shard_samples=args.shard_samples)
    print(f"[编码] 完成:{total} 样本,{shards} 个分片,耗时 {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
