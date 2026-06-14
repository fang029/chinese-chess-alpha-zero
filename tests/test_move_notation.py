"""中文记谱解析(move_notation)测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.board import Board
from xiangqi.game import GameState
from xiangqi.move_notation import parse_move
from xiangqi.constants import (RED, BLACK, GENERAL, CANNON, HORSE, CHARIOT,
                               ELEPHANT, ADVISOR, SOLDIER)


def _legal(g, m):
    return tuple(m) in set(map(tuple, g.legal_moves()))


def test_standard_opening():
    """开局三步:炮二平五 / 马8进7 / 马二进三,均应合法。"""
    g = GameState()
    m = parse_move(g.board, "炮二平五", RED)
    assert m == ((2, 7), (2, 4)) and _legal(g, m)
    g.push(m)
    m = parse_move(g.board, "马8进7", BLACK)
    assert m == ((9, 7), (7, 6)) and _legal(g, m)
    g.push(m)
    m = parse_move(g.board, "马二进三", RED)
    assert m == ((0, 7), (2, 6)) and _legal(g, m)


def test_cannon_horizontal_and_advance():
    g = GameState()
    # 炮二退一:红炮 (2,7) 退(row减)到 (1,7)
    m = parse_move(g.board, "炮二退一", RED)
    assert m == ((1, 7), (1, 7)) or m == ((2, 7), (1, 7))
    assert m == ((2, 7), (1, 7)) and _legal(g, m)


def test_front_back_prefix():
    """同纵线两炮,前炮/后炮 分别取 row 大/小者(红方)。"""
    b = Board()
    b.grid = [[0] * 9 for _ in range(10)]
    b.set(0, 4, GENERAL)
    b.set(9, 4, -GENERAL)
    b.set(2, 4, CANNON)
    b.set(4, 4, CANNON)
    g = GameState(board=b, to_move=RED)
    m1 = parse_move(g.board, "前炮进一", RED)
    assert m1 == ((4, 4), (5, 4)) and _legal(g, m1)
    m2 = parse_move(g.board, "后炮平六", RED)
    assert m2 == ((2, 4), (2, 3)) and _legal(g, m2)


def test_black_perspective_files():
    """黑方纵线用阿拉伯数字,自黑方视角自右向左(col = f-1)。"""
    g = GameState()
    g.push(parse_move(g.board, "炮二平五", RED))
    # 黑炮2平5:col1 -> col4,row 不变(row7)
    m = parse_move(g.board, "炮2平5", BLACK)
    assert m == ((7, 1), (7, 4)) and _legal(g, m)


def test_chariot_advance_and_horizontal():
    b = Board()
    b.grid = [[0] * 9 for _ in range(10)]
    b.set(0, 3, GENERAL)
    b.set(9, 4, -GENERAL)
    b.set(0, 0, CHARIOT)
    g = GameState(board=b, to_move=RED)
    m = parse_move(g.board, "车九进一", RED)
    assert m == ((0, 0), (1, 0)) and _legal(g, m)
    m = parse_move(g.board, "车九平八", RED)
    assert m == ((0, 0), (0, 1)) and _legal(g, m)


def test_horse_diagonal_target_file():
    """马进退按目标纵线定位,row 位移由蹩腿方向推出。"""
    g = GameState()
    # 马二进三:(0,7)->(2,6),dc=1 故 dr=2
    m = parse_move(g.board, "马二进三", RED)
    assert m == ((0, 7), (2, 6)) and _legal(g, m)


def test_invalid_raises():
    g = GameState()
    import pytest
    with pytest.raises(ValueError):
        parse_move(g.board, "口口口口", RED)
    with pytest.raises(ValueError):
        parse_move(g.board, "马", RED)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
