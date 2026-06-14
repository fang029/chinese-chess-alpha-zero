"""MCTS 批量推理(virtual loss)与树复用测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiangqi.game import GameState
from xiangqi.network import XiangqiNet
from xiangqi.evaluator import Evaluator
from xiangqi.mcts import MCTS, action_probabilities, advance_root


def _mcts(batch_size):
    net = XiangqiNet(channels=16, num_blocks=2)
    ev = Evaluator(net, device=torch.device("cpu"))
    return MCTS(ev, batch_size=batch_size)


def test_visit_count_invariant_with_virtual_loss():
    """无论 batch_size,根访问数 == 模拟数,子节点访问和 == 模拟数。

    这验证 virtual loss 在回传时被完全撤销,无残留计数。
    """
    for bs in (1, 4, 16):
        mcts = _mcts(bs)
        g = GameState()
        root = mcts.run(g, num_simulations=32, add_noise=False)
        assert root.visit_count == 32
        child_visits = sum(c.visit_count for c in root.children.values())
        assert child_visits == 32, f"batch_size={bs} 访问数不变量破坏"


def test_virtual_loss_no_residual_value():
    """搜索后,每个被访问子节点的 value_sum 不应含 virtual loss 残留。

    检验方式:value (= value_sum/visit_count) 必须落在 [-1, 1] 内。
    若 virtual loss 未撤销干净,value_sum 会被多扣 VL,导致越界。
    """
    mcts = _mcts(8)
    g = GameState()
    root = mcts.run(g, num_simulations=48, add_noise=False)
    for child in root.children.values():
        if child.visit_count > 0:
            assert -1.0 <= child.value() <= 1.0


def test_tree_reuse_preserves_subtree():
    """advance_root 返回的子节点保留其子树统计。"""
    mcts = _mcts(8)
    g = GameState()
    root = mcts.run(g, num_simulations=40, add_noise=False)
    # 选访问最多的走法
    best_move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    child = root.children[best_move]
    child_visits_before = child.visit_count

    new_root = advance_root(root, best_move)
    assert new_root is child
    assert new_root.visit_count == child_visits_before
    # 复用子树继续搜索:已扩展则不需重新评估根
    assert new_root.is_expanded or not new_root.children


def test_tree_reuse_continues_search():
    """用复用的子树继续搜索,访问数在原基础上累加。"""
    mcts = _mcts(8)
    g = GameState()
    root = mcts.run(g, num_simulations=40, add_noise=False)
    best_move = max(root.children.items(), key=lambda kv: kv[1].visit_count)[0]
    g.push(best_move)
    reused = advance_root(root, best_move)
    before = reused.visit_count

    new_root = mcts.run(g, num_simulations=20, add_noise=False, root=reused)
    assert new_root is reused
    # 继续搜索后访问数增加
    assert new_root.visit_count == before + 20


def test_advance_root_unknown_move():
    mcts = _mcts(4)
    g = GameState()
    root = mcts.run(g, num_simulations=10, add_noise=False)
    # 一个不在子节点中的伪走法
    assert advance_root(root, ((9, 9), (9, 9))) is None


def test_batch_produces_valid_policy():
    """大 batch 下仍产出合法归一化的策略分布。"""
    mcts = _mcts(16)
    g = GameState()
    root = mcts.run(g, num_simulations=40, add_noise=True)
    pi = action_probabilities(root, temperature=1.0)
    assert abs(sum(pi.values()) - 1.0) < 1e-5
    assert set(pi.keys()).issubset(set(map(tuple, g.legal_moves())))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
