"""自我对弈的多进程并行。

架构(生产者-消费者):
  - 多个 worker 进程各持网络副本,持续自我对弈,把样本推入 sample_queue。
  - 主进程从队列消费样本、训练,把新权重写入文件并递增共享版本号。
  - worker 每局开始前比对版本号,发现更新则从文件重载权重,实现异步权重同步。

worker 与训练解耦:worker 不间断产出对局,训练在主进程独立进行。这是
AlphaZero 提升吞吐的关键(自我对弈是瓶颈)。

设备约定:worker 默认在 CPU 上推理,避免与主进程的 GPU 训练争用显存;
在多卡云端可通过 worker_device 指定各 worker 用某张卡,或后续升级为推理服务。
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .network import XiangqiNet
from .evaluator import Evaluator
from .selfplay import SelfPlayConfig, play_game
from .inference_server import InferenceServer, MultiGPUInferenceServer


def _server_worker_loop(worker_id, sp_config, request_queue, result_queue,
                        sample_queue, stop_event, seed):
    """推理服务模式的 worker:不持网络,用 RemoteEvaluator 把推理发给中央服务。

    搜索逻辑(MCTS)在本进程 CPU 跑,局面评估走远程服务的大 batch GPU。
    """
    from .inference_server import RemoteEvaluator
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    evaluator = RemoteEvaluator(worker_id, request_queue, result_queue)

    while not stop_event.is_set():
        try:
            samples, result, n_moves = play_game(evaluator, sp_config, rng)
        except Exception as e:  # 单局异常不拖垮 worker
            print(f"[worker {worker_id}] 对局异常: {e}")
            continue
        sample_queue.put((samples, result, n_moves))


def _worker_loop(worker_id: int,
                 net_config: dict,
                 sp_config: SelfPlayConfig,
                 weights_path: str,
                 shared_version,
                 sample_queue,
                 stop_event,
                 worker_device: str,
                 seed: int):
    """worker 进程主循环。持续对弈并推送样本,按版本号热重载权重。"""
    # 限制每个 worker 的线程数:多 worker 同跑时,torch/BLAS 默认各开多线程会
    # 超额订阅 CPU(N worker × M 线程 >> 核数)导致互相抢核、整体变慢。每 worker
    # 单线程,把并行度交给 worker 进程数本身。
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(worker_device)

    net = XiangqiNet(channels=net_config["channels"],
                     num_blocks=net_config["blocks"])
    evaluator = Evaluator(net, device=device)

    local_version = -1

    while not stop_event.is_set():
        # 权重热重载:版本号变化则从文件加载最新权重。
        current = shared_version.value
        if current > local_version:
            if os.path.exists(weights_path):
                try:
                    state = torch.load(weights_path, map_location=device)
                    net.load_state_dict(state)
                    local_version = current
                except (RuntimeError, EOFError, FileNotFoundError):
                    # 主进程正在写文件,稍后重试。
                    time.sleep(0.1)
                    continue

        try:
            samples, result, n_moves = play_game(evaluator, sp_config, rng)
        except Exception as e:  # 单局异常不应拖垮 worker
            print(f"[worker {worker_id}] 对局异常: {e}")
            continue

        # 把样本送回主进程;队列满时阻塞等待,形成自然背压。
        sample_queue.put((samples, result, n_moves))


class ParallelSelfPlay:
    """管理一组自我对弈 worker 进程的生命周期与样本收集。"""

    def __init__(self,
                 net_config: dict,
                 sp_config: SelfPlayConfig,
                 weights_path: str,
                 num_workers: int,
                 worker_device: str = "cpu",
                 queue_size: int = 64,
                 seed: int = 0,
                 use_inference_server: bool = False,
                 server_device: str = "cuda",
                 max_infer_batch: int = 256,
                 infer_timeout_ms: float = 5.0,
                 num_gpus: int = 1):
        self.net_config = net_config
        self.sp_config = sp_config
        self.weights_path = weights_path
        self.num_workers = num_workers
        self.worker_device = worker_device
        self.seed = seed
        self.use_inference_server = use_inference_server

        # spawn 上下文:CUDA 与 macOS 下必须用 spawn。
        self.ctx = mp.get_context("spawn")
        self.sample_queue = self.ctx.Queue(maxsize=queue_size)
        self.shared_version = self.ctx.Value("i", 0)
        self.stop_event = self.ctx.Event()
        self.workers: list = []

        # 推理服务模式:中央服务独占 GPU 攒批前向,worker 用 RemoteEvaluator。
        self.server = None
        if use_inference_server:
            # 多卡服务统一走 MultiGPUInferenceServer(num_gpus=1 即单卡特例)。
            self.server = MultiGPUInferenceServer(
                net_config, weights_path, num_workers, num_gpus,
                server_device_prefix=server_device, max_batch=max_infer_batch,
                timeout_ms=infer_timeout_ms, ctx=self.ctx)

    def publish_weights(self, net: XiangqiNet):
        """发布新权重。推理服务模式只通知服务进程重载;否则写文件供 worker 重载。

        先写临时文件再原子重命名,避免读到半写状态。
        """
        if self.server is not None:
            self.server.publish_weights(net)
            return
        tmp = self.weights_path + ".tmp"
        cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save(cpu_state, tmp)
        os.replace(tmp, self.weights_path)
        with self.shared_version.get_lock():
            self.shared_version.value += 1

    def start(self):
        if self.server is not None:
            # 先启动推理服务(加载网络),再起 worker。
            self.server.start()
            for i in range(self.num_workers):
                p = self.ctx.Process(
                    target=_server_worker_loop,
                    args=(i, self.sp_config, self.server.request_queue_for(i),
                          self.server.result_queues[i], self.sample_queue,
                          self.stop_event, self.seed + i + 1),
                    daemon=True,
                )
                p.start()
                self.workers.append(p)
            return
        for i in range(self.num_workers):
            p = self.ctx.Process(
                target=_worker_loop,
                args=(i, self.net_config, self.sp_config, self.weights_path,
                      self.shared_version, self.sample_queue, self.stop_event,
                      self.worker_device, self.seed + i + 1),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def collect_games(self, num_games: int):
        """阻塞收集 num_games 局的样本,返回 (all_samples, results, total_moves)。"""
        all_samples = []
        results = {"red_win": 0, "black_win": 0, "draw": 0}
        total_moves = 0
        collected = 0
        while collected < num_games:
            samples, result, n_moves = self.sample_queue.get()
            all_samples.extend(samples)
            key = result if result in results else "draw"
            results[key] += 1
            total_moves += n_moves
            collected += 1
        return all_samples, results, total_moves

    def stop(self):
        self.stop_event.set()
        # 清空队列,避免 worker 阻塞在 put 上无法退出。
        try:
            while True:
                self.sample_queue.get_nowait()
        except Exception:
            pass
        for p in self.workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.workers = []
        if self.server is not None:
            self.server.stop()
