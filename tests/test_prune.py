"""checkpoint 清理(prune_checkpoints)测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.pipeline import prune_checkpoints


def _touch(path):
    with open(path, "w") as f:
        f.write("x")


def _make_ckpts(d, n):
    for i in range(1, n + 1):
        _touch(os.path.join(d, f"iter_{i:04d}.pt"))


def test_keeps_most_recent_n(tmp_path):
    d = str(tmp_path)
    _make_ckpts(d, 10)
    prune_checkpoints(d, keep=3)
    remaining = sorted(f for f in os.listdir(d) if f.startswith("iter_"))
    assert remaining == ["iter_0008.pt", "iter_0009.pt", "iter_0010.pt"]


def test_sorts_by_iter_number_not_lexical(tmp_path):
    """iter_0009 与 iter_0010:数值序应保留 0010,而非字典序误删。"""
    d = str(tmp_path)
    _make_ckpts(d, 12)  # 跨越 9->10->...->12,字典序与数值序在此分叉
    prune_checkpoints(d, keep=2)
    remaining = sorted(f for f in os.listdir(d) if f.startswith("iter_"))
    assert remaining == ["iter_0011.pt", "iter_0012.pt"]


def test_keep_zero_is_noop(tmp_path):
    d = str(tmp_path)
    _make_ckpts(d, 5)
    prune_checkpoints(d, keep=0)
    assert len([f for f in os.listdir(d) if f.startswith("iter_")]) == 5


def test_preserves_non_iter_files(tmp_path):
    d = str(tmp_path)
    _make_ckpts(d, 5)
    _touch(os.path.join(d, "latest.pt"))
    _touch(os.path.join(d, "champion.pt"))
    _touch(os.path.join(d, "buffer.npz"))
    prune_checkpoints(d, keep=2)
    files = set(os.listdir(d))
    assert "latest.pt" in files
    assert "champion.pt" in files
    assert "buffer.npz" in files
    iters = sorted(f for f in files if f.startswith("iter_"))
    assert iters == ["iter_0004.pt", "iter_0005.pt"]


def test_keep_larger_than_count(tmp_path):
    d = str(tmp_path)
    _make_ckpts(d, 3)
    prune_checkpoints(d, keep=10)  # 不应删任何
    assert len([f for f in os.listdir(d) if f.startswith("iter_")]) == 3


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
