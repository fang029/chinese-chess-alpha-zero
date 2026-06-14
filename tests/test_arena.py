"""评估门控(arena)测试:对弈流程、统计正确性、晋级判定。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from xiangqi.network import XiangqiNet
from xiangqi.evaluator import Evaluator
from xiangqi.arena import ArenaConfig, compare, play_match_game
from xiangqi.mcts import MCTS


def _eval():
    net = XiangqiNet(channels=16, num_blocks=2)
    return Evaluator(net, device=torch.device("cpu"))


def test_play_match_game_returns_result():
    ev = _eval()
    m_red = MCTS(ev)
    m_black = MCTS(ev)
    result = play_match_game(m_red, m_black, simulations=10, max_moves=20)
    assert result in ("red_win", "black_win", "draw")


def test_compare_stats_consistent():
    ev_a = _eval()
    ev_b = _eval()
    config = ArenaConfig(num_games=4, num_simulations=8, max_moves=20)
    res = compare(ev_a, ev_b, config)
    # 局数守恒
    assert res["wins"] + res["losses"] + res["draws"] == res["games"]
    assert res["games"] == 4
    # 得分率定义:胜1和0.5负0
    expected = (res["wins"] + 0.5 * res["draws"]) / res["games"]
    assert abs(res["score"] - expected) < 1e-9
    assert 0.0 <= res["score"] <= 1.0


def test_promote_flag_follows_threshold():
    ev_a = _eval()
    ev_b = _eval()
    # 阈值设 0 必晋级,设 >1 必不晋级,验证 promote 跟随 score。
    low = compare(ev_a, ev_b, ArenaConfig(num_games=2, num_simulations=8,
                                          max_moves=15, win_threshold=0.0))
    assert low["promote"] is True

    high = compare(ev_a, ev_b, ArenaConfig(num_games=2, num_simulations=8,
                                           max_moves=15, win_threshold=1.01))
    assert high["promote"] is False


def test_compare_color_alternation_covers_both_sides():
    """偶数局应让 challenger 红黑各半,统计不偏向某一颜色。"""
    ev = _eval()
    # 同一网络自对,得分率应接近(非严格,因贪心+先手),只验证流程跑通且无异常。
    res = compare(ev, ev, ArenaConfig(num_games=2, num_simulations=8,
                                      max_moves=15))
    assert res["games"] == 2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
