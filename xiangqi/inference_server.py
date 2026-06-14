"""批量推理服务:集中式 GPU 推理,把多 worker 的碎请求攒成大 batch。

AlphaZero 自弈的 MCTS 每次只评估一个局面(batch=1),GPU 干这种碎活利用率极低。
本模块开一个独占 GPU 的推理服务进程:所有 worker 把待评估局面丢进共享请求队列,
服务进程攒够一批(到 max_batch 或 timeout)一次性前向,再按 worker 分发结果。
GPU 吃大 batch,利用率大幅提升。

设计要点:
- 服务进程保持纯粹:只做 encoded_tensor -> (logits, value) 前向,不碰掩码/softmax。
  掩码 softmax、policy dict 构建在 worker(CPU)侧做,可并行。
- RemoteEvaluator 接口与 evaluator.Evaluator 完全一致(evaluate / evaluate_batch),
  MCTS / selfplay 无需改动。
- 防死锁:批超时即发车(凑不满也算);worker 收结果带超时重试 + stop_event。
- 权重只在服务进程一份:训练后主进程写文件 + 递增版本号,服务进程热重载。
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from .network import XiangqiNet, masked_policy
from .state_encoder import encode_state
from . import encoding


# ---- 服务进程主循环 ----

def _server_loop(net_config, weights_path, server_device,
                 request_queue, result_queues, stop_event,
                 shared_version, max_batch, timeout_s, ready_event):
    """推理服务进程:攒批前向,按 worker_id 分发 (logits, values)。

    request_queue 元素:(worker_id, encoded float32 数组 [k,15,10,9])。
    result_queues[worker_id] 收 (logits [k,ACTION_SIZE], values [k]) numpy。
    """
    torch.set_num_threads(1)
    device = torch.device(server_device)
    net = XiangqiNet(channels=net_config["channels"],
                     num_blocks=net_config["blocks"]).to(device)
    net.eval()
    local_version = -1
    ready_event.set()

    while not stop_event.is_set():
        # 权重热重载(版本号变化时)。
        v = shared_version.value
        if v > local_version and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=device)
                net.load_state_dict(state)
                local_version = v
            except (RuntimeError, EOFError, FileNotFoundError):
                time.sleep(0.05)

        # 攒一批:阻塞取第一个,然后在 timeout 内尽量多取。
        batch = []  # [(worker_id, arr)]
        try:
            batch.append(request_queue.get(timeout=0.1))
        except Exception:
            continue
        deadline = time.monotonic() + timeout_s
        n = len(batch[0][1])
        while n < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = request_queue.get(timeout=remaining)
            except Exception:
                break
            batch.append(item)
            n += len(item[1])

        # 拼大 batch 前向。
        arrays = [arr for _, arr in batch]
        big = np.concatenate(arrays, axis=0)
        with torch.no_grad():
            x = torch.from_numpy(big).to(device)
            logits, values = net(x)
            logits = logits.cpu().numpy()
            values = values.cpu().numpy()

        # 按请求边界切回,分发到各 worker 的结果队列。
        off = 0
        for worker_id, arr in batch:
            k = len(arr)
            result_queues[worker_id].put(
                (logits[off:off + k], values[off:off + k]))
            off += k


class InferenceServer:
    """管理推理服务进程的生命周期。多卡时可起多个实例(每卡一个)。"""

    def __init__(self, net_config, weights_path, num_workers,
                 server_device="cuda", max_batch=256, timeout_ms=5.0,
                 ctx=None):
        self.net_config = net_config
        self.weights_path = weights_path
        self.num_workers = num_workers
        self.server_device = server_device
        self.max_batch = max_batch
        self.timeout_s = timeout_ms / 1000.0
        self.ctx = ctx or mp.get_context("spawn")
        self.request_queue = self.ctx.Queue()
        self.result_queues = [self.ctx.Queue() for _ in range(num_workers)]
        self.stop_event = self.ctx.Event()
        self.shared_version = self.ctx.Value("i", 0)
        self._ready = self.ctx.Event()
        self.proc = None

    def publish_weights(self, net: XiangqiNet):
        """写权重文件 + 递增版本号,通知服务进程热重载。"""
        tmp = self.weights_path + ".tmp"
        cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save(cpu_state, tmp)
        os.replace(tmp, self.weights_path)
        with self.shared_version.get_lock():
            self.shared_version.value += 1

    def start(self):
        self.proc = self.ctx.Process(
            target=_server_loop,
            args=(self.net_config, self.weights_path, self.server_device,
                  self.request_queue, self.result_queues, self.stop_event,
                  self.shared_version, self.max_batch, self.timeout_s,
                  self._ready),
            daemon=True,
        )
        self.proc.start()
        self._ready.wait(timeout=60)  # 等服务进程加载完网络

    def make_evaluator(self, worker_id):
        return RemoteEvaluator(worker_id, self.request_queue,
                               self.result_queues[worker_id])

    def stop(self):
        self.stop_event.set()
        if self.proc is not None:
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.terminate()
            self.proc = None


# ---- worker 侧远程代理 evaluator(接口同 evaluator.Evaluator)----

class RemoteEvaluator:
    """worker 侧代理:把推理请求发给中央服务,本地做掩码 softmax。

    接口与 evaluator.Evaluator 一致:evaluate(state) / evaluate_batch(states)。
    MCTS 调用方无感知。掩码 softmax 与 policy dict 构建在此(CPU)完成,
    使服务进程保持纯前向。
    """

    def __init__(self, worker_id, request_queue, result_queue,
                 recv_timeout=30.0):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.recv_timeout = recv_timeout

    def _infer(self, xs: np.ndarray):
        """送编码张量给服务,收回 (logits, values) numpy。"""
        self.request_queue.put((self.worker_id, xs))
        # 阻塞收自己的结果;带超时防止服务异常时永久挂起。
        return self.result_queue.get(timeout=self.recv_timeout)

    def _build(self, legals, logits):
        """对一批 logits 做掩码 softmax,构建 [(policy_dict, ...), ...] 的 policy 部分。"""
        masks = np.array([encoding.legal_action_mask(lm) for lm in legals],
                         dtype=np.float32)
        lt = torch.from_numpy(logits)
        mt = torch.from_numpy(masks)
        probs = masked_policy(lt, mt).numpy()
        out = []
        for i, legal in enumerate(legals):
            if not legal:
                out.append({})
                continue
            pol = {m: float(probs[i, encoding.move_to_index(m)]) for m in legal}
            total = sum(pol.values())
            if total > 0:
                pol = {m: p / total for m, p in pol.items()}
            else:
                u = 1.0 / len(legal)
                pol = {m: u for m in legal}
            out.append(pol)
        return out

    def evaluate(self, state):
        """返回 (policy_dict, value)。与 Evaluator.evaluate 一致。"""
        legal = state.legal_moves()
        if not legal:
            return {}, 0.0
        xs = encode_state(state)[None, ...]
        logits, values = self._infer(xs)
        pol = self._build([legal], logits)[0]
        return pol, float(values[0])

    def evaluate_batch(self, states):
        """批量评估,返回 [(policy_dict, value), ...]。与 Evaluator.evaluate_batch 一致。"""
        if not states:
            return []
        legals = [s.legal_moves() for s in states]
        xs = np.stack([encode_state(s) for s in states])
        logits, values = self._infer(xs)
        pols = self._build(legals, logits)
        results = []
        for i, legal in enumerate(legals):
            if not legal:
                results.append(({}, 0.0))
            else:
                results.append((pols[i], float(values[i])))
        return results


# ---- 多卡推理服务 ----

class MultiGPUInferenceServer:
    """多卡批量推理:起 G 个服务进程,每个独占一张卡,各有独立请求队列。

    worker i 按 i % G 路由到某个服务,均衡分配。结果队列仍按 worker_id 区分。
    单卡是 G=1 的特例(行为等同 InferenceServer)。多机扩展时,每台机器各跑
    一个本实例,样本经上层 sample_queue 汇总(见 docs/06)。
    """

    def __init__(self, net_config, weights_path, num_workers, num_gpus,
                 server_device_prefix="cuda", max_batch=256, timeout_ms=5.0,
                 ctx=None):
        self.net_config = net_config
        self.weights_path = weights_path
        self.num_workers = num_workers
        self.num_gpus = num_gpus
        self.max_batch = max_batch
        self.timeout_s = timeout_ms / 1000.0
        self.ctx = ctx or mp.get_context("spawn")

        # 每张卡一个请求队列;结果队列按 worker 区分(全局共享)。
        self.request_queues = [self.ctx.Queue() for _ in range(num_gpus)]
        self.result_queues = [self.ctx.Queue() for _ in range(num_workers)]
        self.stop_event = self.ctx.Event()
        self.shared_version = self.ctx.Value("i", 0)
        self._ready_events = [self.ctx.Event() for _ in range(num_gpus)]
        # 设备字符串:cuda:0..G-1;若 prefix 是 cpu(测试用),全部用 cpu。
        if server_device_prefix == "cpu":
            self.devices = ["cpu"] * num_gpus
        else:
            self.devices = [f"{server_device_prefix}:{g}" for g in range(num_gpus)]
        self.procs = []

    def publish_weights(self, net: XiangqiNet):
        """写权重文件 + 递增版本号,所有服务进程各自热重载。"""
        tmp = self.weights_path + ".tmp"
        cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        torch.save(cpu_state, tmp)
        os.replace(tmp, self.weights_path)
        with self.shared_version.get_lock():
            self.shared_version.value += 1

    def start(self):
        for g in range(self.num_gpus):
            p = self.ctx.Process(
                target=_server_loop,
                args=(self.net_config, self.weights_path, self.devices[g],
                      self.request_queues[g], self.result_queues,
                      self.stop_event, self.shared_version, self.max_batch,
                      self.timeout_s, self._ready_events[g]),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        for ev in self._ready_events:
            ev.wait(timeout=60)

    def request_queue_for(self, worker_id):
        """worker 按 id 轮询分配到某张卡的请求队列。"""
        return self.request_queues[worker_id % self.num_gpus]

    def make_evaluator(self, worker_id):
        return RemoteEvaluator(worker_id, self.request_queue_for(worker_id),
                               self.result_queues[worker_id])

    def stop(self):
        self.stop_event.set()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.procs = []
