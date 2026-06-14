"""规则引擎单元测试。覆盖各棋子走法、将军检测、白脸将、合法走法过滤。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.board import Board
from xiangqi.constants import (
    RED, BLACK, GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
)
from xiangqi import movegen


def moves_from(board, side, frm):
    """返回从指定起点出发的所有伪合法落点集合。"""
    return {to for (f, to) in movegen.pseudo_legal_moves(board, side) if f == frm}


def test_initial_move_count():
    """开局红方应有 44 个合法走法(象棋公认值)。"""
    b = Board()
    assert len(movegen.legal_moves(b, RED)) == 44
    assert len(movegen.legal_moves(b, BLACK)) == 44


def test_chariot_open_lines():
    b = Board.empty()
    b.set(4, 4, CHARIOT)  # 红车居中,空盘
    tos = moves_from(b, RED, (4, 4))
    # 横向 8 格 + 纵向 9 格 = 17
    assert len(tos) == 17


def test_chariot_blocked_and_capture():
    b = Board.empty()
    b.set(4, 4, CHARIOT)
    b.set(4, 6, SOLDIER)        # 己方兵,挡路且不可吃
    b.set(4, 1, -SOLDIER)       # 敌方卒,可吃
    tos = moves_from(b, RED, (4, 4))
    assert (4, 5) in tos        # 可走到己方兵前一格
    assert (4, 6) not in tos    # 不可吃己方
    assert (4, 7) not in tos    # 被挡,不可越过
    assert (4, 1) in tos        # 可吃敌方
    assert (4, 0) not in tos    # 不可越过敌方


def test_cannon_jump_capture():
    b = Board.empty()
    b.set(4, 4, CANNON)
    b.set(4, 6, SOLDIER)        # 炮架(己方)
    b.set(4, 8, -SOLDIER)       # 隔炮架的敌方,可吃
    tos = moves_from(b, RED, (4, 4))
    assert (4, 5) in tos        # 炮架前空格可移动
    assert (4, 6) not in tos    # 不可吃炮架(且是己方)
    assert (4, 7) not in tos    # 炮架后第一格但无子,不可停
    assert (4, 8) in tos        # 隔一子吃


def test_horse_leg_block():
    b = Board.empty()
    b.set(4, 4, HORSE)
    tos_free = moves_from(b, RED, (4, 4))
    assert len(tos_free) == 8   # 空盘马有 8 个落点
    b.set(5, 4, SOLDIER)        # 蹩住上方马腿
    tos_blocked = moves_from(b, RED, (4, 4))
    assert (6, 5) not in tos_blocked
    assert (6, 3) not in tos_blocked
    assert len(tos_blocked) == 6


def test_elephant_eye_and_river():
    b = Board.empty()
    b.set(2, 2, ELEPHANT)       # 红相
    tos = moves_from(b, RED, (2, 2))
    assert (4, 4) in tos
    assert (4, 0) in tos
    assert (0, 0) in tos
    assert (0, 4) in tos
    b.set(3, 3, SOLDIER)        # 塞象眼
    tos2 = moves_from(b, RED, (2, 2))
    assert (4, 4) not in tos2
    # 不可过河:红相走到 row>4 的点不存在
    b2 = Board.empty()
    b2.set(4, 2, ELEPHANT)
    tos3 = moves_from(b2, RED, (4, 2))
    assert all(to[0] <= 4 for to in tos3)


def test_advisor_in_palace():
    b = Board.empty()
    b.set(1, 4, ADVISOR)        # 红仕居九宫中心
    tos = moves_from(b, RED, (1, 4))
    assert tos == {(0, 3), (0, 5), (2, 3), (2, 5)}


def test_general_in_palace():
    b = Board.empty()
    b.set(0, 4, GENERAL)
    tos = moves_from(b, RED, (0, 4))
    assert tos == {(0, 3), (0, 5), (1, 4)}


def test_soldier_before_and_after_river():
    b = Board.empty()
    b.set(3, 4, SOLDIER)        # 未过河红兵
    assert moves_from(b, RED, (3, 4)) == {(4, 4)}
    b.set(5, 4, SOLDIER)        # 已过河红兵
    assert moves_from(b, RED, (5, 4)) == {(6, 4), (5, 3), (5, 5)}


def test_generals_face_is_illegal():
    b = Board.empty()
    b.set(0, 4, GENERAL)
    b.set(9, 4, -GENERAL)
    assert movegen.generals_face(b) is True
    b.set(4, 4, SOLDIER)        # 中间塞子,解除白脸
    assert movegen.generals_face(b) is False


def test_in_check_detection():
    b = Board.empty()
    b.set(0, 4, GENERAL)
    b.set(5, 4, -CHARIOT)       # 黑车正对红帅
    assert movegen.in_check(b, RED) is True
    b.set(2, 4, SOLDIER)        # 红兵挡住
    assert movegen.in_check(b, RED) is False


def test_legal_moves_filter_self_check():
    b = Board.empty()
    b.set(0, 4, GENERAL)
    b.set(1, 4, ADVISOR)        # 仕挡在帅前
    b.set(5, 4, -CHARIOT)       # 黑车将军路线上
    # 仕若离开 col 4 会暴露帅给黑车,应被过滤
    legal = movegen.legal_moves(b, RED)
    advisor_moves = {to for (f, to) in legal if f == (1, 4)}
    assert advisor_moves == set()  # 仕动则被将,无合法走法


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
