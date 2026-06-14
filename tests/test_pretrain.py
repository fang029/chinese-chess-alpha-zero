"""人类棋谱监督预训练(pretrain)测试:样本生成、价值标签、训练下降。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from xiangqi.pretrain import (game_to_samples, build_dataset, pretrain,
                              save_pretrained, _z_for)
from xiangqi.network import XiangqiNet
from xiangqi.state_encoder import INPUT_CHANNELS
from xiangqi.constants import RED, BLACK
from xiangqi import encoding


def test_game_to_samples_basic():
    moves = ["炮二平五", "马8进7", "马二进三"]
    samples, g = game_to_samples(moves, "red_win")
    assert len(samples) == 3
    # 每条样本:(state, action_index, to_move)
    state, idx, to_move = samples[0]
    assert state.shape == (INPUT_CHANNELS, 10, 9)
    assert 0 <= idx < encoding.ACTION_SIZE
    assert to_move == RED  # 第一步红走
    assert samples[1][2] == BLACK


def test_game_to_samples_stops_on_illegal():
    """遇到无法解析的着法即停止,只保留之前的样本。"""
    moves = ["炮二平五", "口口口口", "马二进三"]
    samples, _ = game_to_samples(moves, None)
    assert len(samples) == 1


def test_z_label_perspective():
    # 红胜:红走方视角 +1,黑走方视角 -1
    assert _z_for("red_win", RED) == 1.0
    assert _z_for("red_win", BLACK) == -1.0
    assert _z_for("black_win", RED) == -1.0
    assert _z_for("draw", RED) == 0.0


def test_build_dataset_shapes():
    games = [
        {"moves": ["炮二平五", "马8进7"], "result": "red_win"},
        {"moves": ["炮八平五"], "result": "draw"},
    ]
    states, idx, zs = build_dataset(games)
    assert states.shape == (3, INPUT_CHANNELS, 10, 9)
    assert idx.shape == (3,)
    assert zs.shape == (3,)
    # 红胜局:红走方 z=+1
    assert zs[0] == 1.0


def test_build_dataset_empty():
    states, idx, zs = build_dataset([{"moves": ["口口口口"], "result": None}])
    assert len(states) == 0


def test_pretrain_reduces_loss():
    games = [
        {"moves": ["炮二平五", "马8进7", "马二进三", "车9平8"], "result": "red_win"},
        {"moves": ["炮八平五", "马2进3", "马八进七"], "result": "draw"},
    ]
    states, idx, zs = build_dataset(games)
    net = XiangqiNet(channels=16, num_blocks=2)
    hist = pretrain(net, states, idx, zs, torch.device("cpu"),
                    epochs=5, batch_size=4, log_every=0)
    assert hist[-1]["total"] < hist[0]["total"]


def test_save_pretrained_loadable(tmp_path):
    games = [{"moves": ["炮二平五"], "result": "red_win"}]
    states, idx, zs = build_dataset(games)
    net = XiangqiNet(channels=16, num_blocks=2)
    pretrain(net, states, idx, zs, torch.device("cpu"),
             epochs=1, batch_size=4, log_every=0)
    path = str(tmp_path / "pre.pt")
    save_pretrained(path, net)
    ckpt = torch.load(path, map_location="cpu")
    assert ckpt["iteration"] == 0
    assert ckpt["channels"] == 16
    assert "model" in ckpt
    # 能加载回同构网络
    net2 = XiangqiNet(channels=16, num_blocks=2)
    net2.load_state_dict(ckpt["model"])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
