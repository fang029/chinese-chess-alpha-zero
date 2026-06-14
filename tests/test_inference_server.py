"""批量推理服务测试:正确性(与本地 Evaluator 一致)、MCTS 不变量、并发、关闭。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from xiangqi.game import GameState
from xiangqi.network import XiangqiNet
from xiangqi.evaluator import Evaluator
from xiangqi.inference_server import InferenceServer
from xiangqi.mcts import MCTS, action_probabilities


def _make_server(tmp_path, num_workers=1, channels=16, blocks=2,
                 max_batch=16, timeout_ms=20.0):
    """造一个推理服务 + 已知权重,返回 (server, net)。"""
    net = XiangqiNet(channels=channels, num_blocks=blocks)
    weights = str(tmp_path / "w.pt")
    cfg = {"channels": channels, "blocks": blocks}
    server = InferenceServer(cfg, weights, num_workers,
                             server_device="cpu", max_batch=max_batch,
                             timeout_ms=timeout_ms)
    server.publish_weights(net)   # 先写权重文件
    server.start()
    return server, net


def test_remote_matches_local(tmp_path):
    """RemoteEvaluator 输出应与本地 Evaluator 在容差内一致。"""
    server, net = _make_server(tmp_path)
    try:
        local = Evaluator(net, device=torch.device("cpu"))
        remote = server.make_evaluator(0)

        g = GameState()
        states = [g.copy()]
        g2 = g.copy()
        g2.push(g2.legal_moves()[0])
        states.append(g2)

        lres = local.evaluate_batch(states)
        rres = remote.evaluate_batch(states)
        assert len(lres) == len(rres)
        for (lp, lv), (rp, rv) in zip(lres, rres):
            assert abs(lv - rv) < 1e-4, f"value 不一致 {lv} vs {rv}"
            assert set(lp.keys()) == set(rp.keys())
            for m in lp:
                assert abs(lp[m] - rp[m]) < 1e-4, f"policy 不一致 {m}"
    finally:
        server.stop()


def test_single_evaluate(tmp_path):
    server, net = _make_server(tmp_path)
    try:
        remote = server.make_evaluator(0)
        pol, val = remote.evaluate(GameState())
        assert abs(sum(pol.values()) - 1.0) < 1e-4
        assert -1.0 <= val <= 1.0
    finally:
        server.stop()


def test_mcts_with_remote_evaluator(tmp_path):
    """用 RemoteEvaluator 跑 MCTS,根访问数==模拟数,子节点 value 合法。"""
    server, net = _make_server(tmp_path, max_batch=8, timeout_ms=20.0)
    try:
        remote = server.make_evaluator(0)
        mcts = MCTS(remote, batch_size=8)
        root = mcts.run(GameState(), num_simulations=32, add_noise=False)
        assert root.visit_count == 32
        child_sum = sum(c.visit_count for c in root.children.values())
        assert child_sum == 32
        for c in root.children.values():
            if c.visit_count > 0:
                assert -1.0 <= c.value() <= 1.0
        pi = action_probabilities(root, temperature=1.0)
        assert abs(sum(pi.values()) - 1.0) < 1e-5
    finally:
        server.stop()


def test_concurrent_workers(tmp_path):
    """多 worker 并发请求,结果正确路由(各拿到自己的结果)、无死锁。"""
    import threading
    server, net = _make_server(tmp_path, num_workers=4, max_batch=32,
                               timeout_ms=20.0)
    try:
        errors = []

        def run(wid):
            try:
                ev = server.make_evaluator(wid)
                for _ in range(5):
                    pol, val = ev.evaluate(GameState())
                    assert abs(sum(pol.values()) - 1.0) < 1e-4
                    assert -1.0 <= val <= 1.0
            except Exception as e:  # noqa
                errors.append(e)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"并发出错: {errors}"
    finally:
        server.stop()


def test_clean_shutdown(tmp_path):
    server, _ = _make_server(tmp_path)
    server.stop()
    assert server.proc is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))


def test_multigpu_routing_and_correctness(tmp_path):
    """G=2 伪多卡(CPU):worker 按 i%G 路由到不同服务,结果与本地一致。"""
    from xiangqi.inference_server import MultiGPUInferenceServer
    net = XiangqiNet(channels=16, num_blocks=2)
    weights = str(tmp_path / "w.pt")
    cfg = {"channels": 16, "blocks": 2}
    server = MultiGPUInferenceServer(cfg, weights, num_workers=4, num_gpus=2,
                                     server_device_prefix="cpu",
                                     max_batch=16, timeout_ms=20.0)
    server.publish_weights(net)
    server.start()
    try:
        local = Evaluator(net, device=torch.device("cpu"))
        # worker 0,2 -> server0;worker 1,3 -> server1
        assert server.request_queue_for(0) is server.request_queue_for(2)
        assert server.request_queue_for(1) is server.request_queue_for(3)
        assert server.request_queue_for(0) is not server.request_queue_for(1)
        g = GameState()
        lres = local.evaluate_batch([g])
        for wid in range(4):
            ev = server.make_evaluator(wid)
            rp, rv = ev.evaluate(g.copy())
            lp, lv = lres[0]
            assert abs(lv - rv) < 1e-4, f"worker{wid} value 不一致"
            assert set(lp.keys()) == set(rp.keys())
    finally:
        server.stop()


def test_multigpu_concurrent(tmp_path):
    """G=2 多 worker 并发,结果正确路由、无死锁。"""
    import threading
    from xiangqi.inference_server import MultiGPUInferenceServer
    net = XiangqiNet(channels=16, num_blocks=2)
    weights = str(tmp_path / "w.pt")
    server = MultiGPUInferenceServer({"channels": 16, "blocks": 2}, weights,
                                     num_workers=6, num_gpus=2,
                                     server_device_prefix="cpu",
                                     max_batch=32, timeout_ms=20.0)
    server.publish_weights(net)
    server.start()
    try:
        errors = []

        def run(wid):
            try:
                ev = server.make_evaluator(wid)
                for _ in range(4):
                    pol, val = ev.evaluate(GameState())
                    assert abs(sum(pol.values()) - 1.0) < 1e-4
            except Exception as e:  # noqa
                errors.append(e)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors, f"并发出错: {errors}"
    finally:
        server.stop()
