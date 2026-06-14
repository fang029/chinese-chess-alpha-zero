"""分布式训练辅助(DDP 数据并行)。

封装进程组初始化、rank/world_size 探测与清理,使训练脚本既能在单进程
(world_size=1,无需 torchrun)下运行,也能用 torchrun 拉起多进程多卡:

    torchrun --nproc_per_node=4 -m xiangqi.pipeline_ddp [args...]

每个 rank 持有一份网络副本,各自做自我对弈生成数据并训练;反向传播时 DDP
自动 all-reduce 梯度,等价于放大 batch 的数据并行。只有 rank 0 负责落盘
checkpoint 与打印日志,避免多进程争写。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """是否在 torchrun 等分布式启动器下运行(存在 WORLD_SIZE 且 > 1)。"""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """rank 0 负责落盘与日志。单进程时恒为 True。"""
    return get_rank() == 0


def setup(backend: str | None = None):
    """初始化进程组。单进程(world_size=1)直接返回,不建组。

    backend 默认按设备选择:有 CUDA 用 nccl,否则 gloo(CPU/调试)。
    返回本 rank 应使用的 torch.device。
    """
    if not is_distributed():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def cleanup():
    """销毁进程组(单进程为 no-op)。"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    """同步所有 rank(单进程为 no-op)。"""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
