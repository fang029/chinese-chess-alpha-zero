"""重复局面裁决测试:长将判负、无害重复判和。

竞技规则简化版:三次重复局面时,若某一方在重复周期内步步将军(长将),
该方判负;双方对等(都不将军或都将军)则按重复和棋。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.board import Board
from xiangqi.game import GameState, RED_WIN, BLACK_WIN, DRAW
from xiangqi.constants import RED, BLACK, GENERAL, CHARIOT


def _setup_perpetual():
    """红车持续将黑将,黑将在 col4/col5 间振荡;红先手已将军黑(黑先逃)。"""
    b = Board()
    b.grid = [[0] * 9 for _ in range(10)]
    b.set(0, 3, GENERAL)        # 红将放 col3,避开照面
    b.set(9, 4, -GENERAL)       # 黑将
    b.set(5, 4, CHARIOT)        # 红车同列将军
    return GameState(board=b, to_move=BLACK)


def test_perpetual_check_loses():
    """红方长将,应判红负(black_win)。"""
    g = _setup_perpetual()
    moves = [((9, 4), (9, 5)), ((5, 4), (5, 5)),
             ((9, 5), (9, 4)), ((5, 5), (5, 4))]
    result = "ongoing"
    for i in range(12):
        m = moves[i % 4]
        legal = set(map(tuple, g.legal_moves()))
        assert tuple(m) in legal, f"步{i} 走法 {m} 非法"
        g.push(m)
        result = g.result()
        if result != "ongoing":
            break
    assert result == BLACK_WIN, f"长将应判红负,实得 {result}"


def test_benign_repetition_draws():
    """双方都不将军的重复,应判和。"""
    b = Board()
    b.grid = [[0] * 9 for _ in range(10)]
    b.set(0, 4, GENERAL)
    b.set(9, 4, -GENERAL)
    b.set(2, 0, CHARIOT)
    b.set(7, 8, -CHARIOT)
    g = GameState(board=b, to_move=RED)
    moves = [((2, 0), (2, 1)), ((7, 8), (7, 7)),
             ((2, 1), (2, 0)), ((7, 7), (7, 8))]
    result = "ongoing"
    for i in range(12):
        g.push(moves[i % 4])
        result = g.result()
        if result != "ongoing":
            break
    assert result == DRAW, f"无害重复应判和,实得 {result}"


def test_perpetual_detection_survives_copy():
    """copy() 应保留重复历史,裁决在副本上仍正确。"""
    g = _setup_perpetual()
    moves = [((9, 4), (9, 5)), ((5, 4), (5, 5)),
             ((9, 5), (9, 4)), ((5, 5), (5, 4))]
    for i in range(5):  # 推进若干步但未触发
        g.push(moves[i % 4])
    g2 = g.copy()
    # 在副本上继续直到触发裁决
    result = g2.result()
    i = 5
    while result == "ongoing" and i < 12:
        g2.push(moves[i % 4])
        result = g2.result()
        i += 1
    assert result == BLACK_WIN


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
