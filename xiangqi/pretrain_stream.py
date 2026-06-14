"""大规模棋谱流式监督预训练。

14 万局棋谱约 1160 万样本,全量编码成张量需数百 GB 内存,不可行。本模块
分块处理:每次取一批棋谱 -> 即时编码成小批张量 -> 训练若干步 -> 释放 ->
取下一批。一遍扫完所有棋谱算一个 epoch。

与 pretrain.pretrain 的区别:那个接收已编码好的内存数组(适合小数据);
本模块接收原始棋谱列表 + 记法,边解析边训练(适合大数据)。
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F

from .pretrain import game_to_samples, _z_for


def _encode_chunk(games_chunk, notation):
    """把一批棋谱编码成 (states, idx, zs) numpy 数组。"""
    states, idxs, zs = [], [], []
    for game in games_chunk:
        result = game.get("result")
        samples, _ = game_to_samples(game.get("moves", []), result, notation=notation)
        for state, idx, to_move in samples:
            states.append(state)
            idxs.append(idx)
            zs.append(_z_for(result, to_move))
    if not states:
        return None
    return (np.stack(states),
            np.array(idxs, dtype=np.int64),
            np.array(zs, dtype=np.float32))


def pretrain_streaming(net, games, device, *, notation="iccs", epochs=1,
                       chunk_games=512, batch_size=256, lr=1e-3,
                       value_weight=1.0, log_every=20, rng=None, shuffle=True):
    """流式预训练。games 为原始棋谱列表 [{"moves":[...],"result":...}, ...]。

    notation: "iccs" 或 "chinese"。
    每个 epoch 打乱棋谱顺序,按 chunk_games 局为单位编码并训练。
    log_every 控制每多少个 chunk 打印一次进度。
    """
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = rng or np.random.default_rng(0)
    n_games = len(games)

    history = []
    for epoch in range(epochs):
        order = np.arange(n_games)
        if shuffle:
            rng.shuffle(order)

        agg = {"total": 0.0, "policy": 0.0, "value": 0.0}
        steps = 0
        samples_seen = 0
        for ci, start in enumerate(range(0, n_games, chunk_games)):
            sel = order[start:start + chunk_games]
            chunk = [games[i] for i in sel]
            enc = _encode_chunk(chunk, notation)
            if enc is None:
                continue
            states, idxs, zs = enc
            x = torch.from_numpy(states).to(device)
            ti = torch.from_numpy(idxs).to(device)
            tz = torch.from_numpy(zs).to(device)

            # 在这批样本内做若干 minibatch 训练步。
            m = len(states)
            perm = torch.randperm(m)
            for bstart in range(0, m, batch_size):
                bsel = perm[bstart:bstart + batch_size]
                logits, value = net(x[bsel])
                ploss = F.cross_entropy(logits, ti[bsel])
                vloss = F.mse_loss(value, tz[bsel])
                loss = ploss + value_weight * vloss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                agg["total"] += loss.item()
                agg["policy"] += ploss.item()
                agg["value"] += vloss.item()
                steps += 1
            samples_seen += m

            if log_every and (ci + 1) % log_every == 0:
                a = {k: v / max(1, steps) for k, v in agg.items()}
                print(f"[预训练 e{epoch+1} chunk {ci+1}] "
                      f"已见样本 {samples_seen} | "
                      f"total={a['total']:.3f} policy={a['policy']:.3f} "
                      f"value={a['value']:.3f}")

        avg = {k: v / max(1, steps) for k, v in agg.items()}
        history.append(avg)
        print(f"[预训练 epoch {epoch+1}/{epochs} 完成] "
              f"样本 {samples_seen} | total={avg['total']:.3f} "
              f"policy={avg['policy']:.3f} value={avg['value']:.3f}")
    return history


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="大规模棋谱流式监督预训练")
    p.add_argument("--games", default=None,
                   help="棋谱 JSON(流式模式);与 --shards 二选一")
    p.add_argument("--shards", default=None,
                   help="预编码分片目录(分片模式,推荐);与 --games 二选一")
    p.add_argument("--out", default="checkpoints/pretrained.pt")
    p.add_argument("--notation", choices=["iccs", "chinese"], default="iccs")
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--chunk-games", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=0,
                   help="只用前 N 局(0=全部),用于快速试跑")
    return p.parse_args()


def main():
    import json
    import os
    from .network import XiangqiNet
    from .evaluator import pick_device
    from .pretrain import save_pretrained

    args = parse_args()
    device = pick_device(args.device)
    net = XiangqiNet(channels=args.channels, num_blocks=args.blocks)

    if args.shards:
        # 分片模式:从预编码 .npz 训练(推荐,快)。
        print(f"[预训练] 分片模式 {args.shards},设备 {device}")
        pretrain_from_shards(net, args.shards, device, epochs=args.epochs,
                             batch_size=args.batch_size, lr=args.lr)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        save_pretrained(args.out, net)
        print(f"[预训练] 已保存 {args.out};正式训练用 --resume {args.out} 接续")
        return

    # 流式模式:从棋谱 JSON 边解析边训练。
    with open(args.games, "r", encoding="utf-8") as f:
        games = json.load(f)
    if args.limit:
        games = games[:args.limit]
    print(f"[预训练] 载入 {len(games)} 局,记法 {args.notation},设备 {device}")

    pretrain_streaming(net, games, device, notation=args.notation,
                       epochs=args.epochs, chunk_games=args.chunk_games,
                       batch_size=args.batch_size, lr=args.lr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_pretrained(args.out, net)
    print(f"[预训练] 已保存 {args.out};正式训练用 --resume {args.out} 接续")


def pretrain_from_shards(net, shard_dir, device, *, epochs=2, batch_size=1024,
                         lr=1e-3, value_weight=1.0, log_every=50, rng=None):
    """从预编码的 .npz 分片训练(已并行编码到磁盘,训练只管喂 GPU)。

    每个 epoch 遍历所有分片;分片内打乱做 minibatch。分片逐个加载,内存可控。
    """
    import glob
    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    rng = rng or np.random.default_rng(0)
    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"{shard_dir} 下无 shard_*.npz")

    history = []
    for epoch in range(epochs):
        order = list(range(len(shards)))
        rng.shuffle(order)
        agg = {"total": 0.0, "policy": 0.0, "value": 0.0}
        steps = 0
        seen = 0
        for si in order:
            data = np.load(shards[si])
            states, idx, zs = data["states"], data["idx"], data["zs"]
            x = torch.from_numpy(states).to(device)
            ti = torch.from_numpy(idx).to(device)
            tz = torch.from_numpy(zs).to(device)
            m = len(states)
            perm = torch.randperm(m)
            for bstart in range(0, m, batch_size):
                bsel = perm[bstart:bstart + batch_size]
                logits, value = net(x[bsel])
                ploss = F.cross_entropy(logits, ti[bsel])
                vloss = F.mse_loss(value, tz[bsel])
                loss = ploss + value_weight * vloss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                agg["total"] += loss.item()
                agg["policy"] += ploss.item()
                agg["value"] += vloss.item()
                steps += 1
            seen += m
            if log_every and steps % log_every < (m // batch_size + 1):
                a = {k: v / max(1, steps) for k, v in agg.items()}
                print(f"[预训练 e{epoch+1}] 分片 {si} 已见 {seen} 样本 | "
                      f"total={a['total']:.3f} policy={a['policy']:.3f} "
                      f"value={a['value']:.3f}", flush=True)
        avg = {k: v / max(1, steps) for k, v in agg.items()}
        history.append(avg)
        print(f"[预训练 epoch {epoch+1}/{epochs} 完成] 样本 {seen} | "
              f"total={avg['total']:.3f} policy={avg['policy']:.3f} "
              f"value={avg['value']:.3f}", flush=True)
    return history


if __name__ == "__main__":
    main()
