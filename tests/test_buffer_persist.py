"""Replay buffer 持久化测试:save/load 往返保真,支持断点续训。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from xiangqi.train import ReplayBuffer
from xiangqi import encoding
from xiangqi.state_encoder import INPUT_CHANNELS
from xiangqi.constants import NUM_ROWS, NUM_COLS


def _fake_samples(n, rng):
    """造 n 条形状正确的样本 (state, pi, z)。"""
    samples = []
    for _ in range(n):
        state = rng.standard_normal(
            (INPUT_CHANNELS, NUM_ROWS, NUM_COLS)).astype(np.float32)
        pi = rng.random(encoding.ACTION_SIZE).astype(np.float32)
        pi /= pi.sum()
        z = np.float32(rng.choice([-1.0, 0.0, 1.0]))
        samples.append((state, pi, z))
    return samples


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=100)
    buf.add_many(_fake_samples(10, rng))

    path = str(tmp_path / "buffer.npz")
    buf.save(path)

    buf2 = ReplayBuffer(capacity=100)
    n = buf2.load(path)
    assert n == 10
    assert len(buf2) == 10
    # 内容逐条比对
    for (s1, p1, z1), (s2, p2, z2) in zip(buf.buffer, buf2.buffer):
        assert np.allclose(s1, s2)
        assert np.allclose(p1, p2)
        assert z1 == z2


def test_save_empty_buffer(tmp_path):
    buf = ReplayBuffer(capacity=10)
    path = str(tmp_path / "empty.npz")
    buf.save(path)  # 不应抛异常
    buf2 = ReplayBuffer(capacity=10)
    assert buf2.load(path) == 0


def test_loaded_buffer_is_sampleable(tmp_path):
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=50)
    buf.add_many(_fake_samples(8, rng))
    path = str(tmp_path / "b.npz")
    buf.save(path)

    buf2 = ReplayBuffer(capacity=50)
    buf2.load(path)
    states, pis, zs = buf2.sample(4)
    assert states.shape == (4, INPUT_CHANNELS, NUM_ROWS, NUM_COLS)
    assert pis.shape == (4, encoding.ACTION_SIZE)
    assert zs.shape == (4,)


def test_save_atomic_overwrites(tmp_path):
    """二次 save 覆盖旧文件,且不残留 .tmp。"""
    rng = np.random.default_rng(2)
    buf = ReplayBuffer(capacity=50)
    buf.add_many(_fake_samples(5, rng))
    path = str(tmp_path / "b.npz")
    buf.save(path)
    buf.add_many(_fake_samples(3, rng))
    buf.save(path)

    buf2 = ReplayBuffer(capacity=50)
    assert buf2.load(path) == 8
    assert not os.path.exists(path + ".tmp.npz")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
