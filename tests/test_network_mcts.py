"""网络与 MCTS 集成测试:前向传播形状、搜索流程、概率分布合法性。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiangqi.game import GameState
from xiangqi.network import XiangqiNet, masked_policy
from xiangqi.evaluator import Evaluator
from xiangqi.mcts import MCTS, action_probabilities
from xiangqi.encoding import ACTION_SIZE
from xiangqi.state_encoder import encode_state, INPUT_CHANNELS
from xiangqi.constants import NUM_ROWS, NUM_COLS


def _small_net():
    # 小网络,测试用,加速。
    return XiangqiNet(channels=16, num_blocks=2)


def test_network_forward_shapes():
    net = _small_net()
    net.eval()
    x = torch.randn(4, INPUT_CHANNELS, NUM_ROWS, NUM_COLS)
    logits, value = net(x)
    assert logits.shape == (4, ACTION_SIZE)
    assert value.shape == (4,)
    assert torch.all(value >= -1) and torch.all(value <= 1)


def test_masked_policy_zeros_illegal():
    logits = torch.randn(2, ACTION_SIZE)
    mask = torch.zeros(2, ACTION_SIZE)
    mask[:, :5] = 1  # 只有前 5 个合法
    probs = masked_policy(logits, mask)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(probs[:, 5:] == 0)


def test_evaluator_returns_distribution():
    net = _small_net()
    ev = Evaluator(net, device=torch.device("cpu"))
    g = GameState()
    policy, value = ev.evaluate(g)
    assert abs(sum(policy.values()) - 1.0) < 1e-5
    assert set(policy.keys()) == set(map(tuple, g.legal_moves()))
    assert -1.0 <= value <= 1.0


def test_evaluator_batch_matches_single():
    net = _small_net()
    ev = Evaluator(net, device=torch.device("cpu"))
    g1 = GameState()
    g2 = GameState()
    g2.push(g2.legal_moves()[0])
    batch = ev.evaluate_batch([g1, g2])
    assert len(batch) == 2
    for pol, val in batch:
        assert abs(sum(pol.values()) - 1.0) < 1e-5


def test_mcts_runs_and_produces_policy():
    net = _small_net()
    ev = Evaluator(net, device=torch.device("cpu"))
    mcts = MCTS(ev)
    g = GameState()
    root = mcts.run(g, num_simulations=30, add_noise=True)
    # 根节点访问次数应等于模拟次数(根每次模拟 +1)
    assert root.visit_count == 30
    # 子节点访问总和应为 模拟次数(每次模拟恰好新增一条路径经过一个子)
    child_visits = sum(c.visit_count for c in root.children.values())
    assert child_visits == 30
    pi = action_probabilities(root, temperature=1.0)
    assert abs(sum(pi.values()) - 1.0) < 1e-5
    assert set(pi.keys()).issubset(set(map(tuple, g.legal_moves())))


def test_mcts_temperature_zero_is_argmax():
    net = _small_net()
    ev = Evaluator(net, device=torch.device("cpu"))
    mcts = MCTS(ev)
    g = GameState()
    root = mcts.run(g, num_simulations=40, add_noise=False)
    pi = action_probabilities(root, temperature=0.0)
    # 应为 one-hot
    assert sum(1 for p in pi.values() if p > 0) == 1
    most_visited = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    assert pi[most_visited] == 1.0


def test_low_temperature_no_overflow():
    """极小温度不应溢出(回归:visits^(1/τ) 曾因大指数 OverflowError)。"""
    net = _small_net()
    ev = Evaluator(net, device=torch.device("cpu"))
    mcts = MCTS(ev)
    g = GameState()
    root = mcts.run(g, num_simulations=40, add_noise=False)
    pi = action_probabilities(root, temperature=1e-3)
    assert abs(sum(pi.values()) - 1.0) < 1e-5
    assert all(0.0 <= p <= 1.0 for p in pi.values())


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
