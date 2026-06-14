"""并行自我对弈集成测试。

验证:权重原子发布、worker 启动产出样本、主进程收集、干净关闭。
用极小网络与模拟次数,保持快速。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiangqi.network import XiangqiNet
from xiangqi.selfplay import SelfPlayConfig
from xiangqi.parallel_selfplay import ParallelSelfPlay
from xiangqi.encoding import ACTION_SIZE
from xiangqi.state_encoder import INPUT_CHANNELS
from xiangqi.constants import NUM_ROWS, NUM_COLS


def test_publish_weights_atomic(tmp_path):
    net = XiangqiNet(channels=16, num_blocks=2)
    weights = str(tmp_path / "w.pt")
    sp = ParallelSelfPlay(
        net_config={"channels": 16, "blocks": 2},
        sp_config=SelfPlayConfig(num_simulations=4, max_moves=10),
        weights_path=weights,
        num_workers=1,
        worker_device="cpu",
    )
    assert sp.shared_version.value == 0
    sp.publish_weights(net)
    assert sp.shared_version.value == 1
    assert os.path.exists(weights)
    assert not os.path.exists(weights + ".tmp")  # 临时文件已重命名
    # 发布的权重可被重新加载
    loaded = torch.load(weights, map_location="cpu")
    net2 = XiangqiNet(channels=16, num_blocks=2)
    net2.load_state_dict(loaded)


def test_parallel_produces_samples(tmp_path):
    """启动 worker,收集若干局,验证样本结构正确,然后干净关闭。"""
    net = XiangqiNet(channels=16, num_blocks=2)
    weights = str(tmp_path / "w.pt")
    sp = ParallelSelfPlay(
        net_config={"channels": 16, "blocks": 2},
        sp_config=SelfPlayConfig(num_simulations=4, temperature_moves=2,
                                 max_moves=12),
        weights_path=weights,
        num_workers=2,
        worker_device="cpu",
        queue_size=16,
        seed=1,
    )
    sp.publish_weights(net)
    sp.start()
    try:
        samples, results, total_moves = sp.collect_games(3)
    finally:
        sp.stop()

    assert sum(results.values()) == 3
    assert len(samples) > 0
    # 样本结构: (state_tensor, pi_vec, z)
    state_t, pi_vec, z = samples[0]
    assert state_t.shape == (INPUT_CHANNELS, NUM_ROWS, NUM_COLS)
    assert pi_vec.shape == (ACTION_SIZE,)
    assert -1.0 <= float(z) <= 1.0
    # 所有 worker 已退出
    assert all(not p.is_alive() for p in sp.workers) or sp.workers == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
