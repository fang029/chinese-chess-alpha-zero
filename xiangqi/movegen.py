"""走法生成。

分两层:
  1. 伪合法走法 (pseudo-legal): 仅依据棋子走法规则,不考虑走后是否被将军。
  2. 合法走法 (legal): 在伪合法基础上,排除走后己方帅/将被攻击的走法
     (含"白脸将"——双方帅将在同一列且中间无子)。

走法表示为 (from_rc, to_rc),即 ((fr, fc), (tr, tc))。
"""

from __future__ import annotations

from .board import Board
from .constants import (
    NUM_ROWS,
    NUM_COLS,
    RED,
    BLACK,
    GENERAL,
    ADVISOR,
    ELEPHANT,
    HORSE,
    CHARIOT,
    CANNON,
    SOLDIER,
    side_of,
    in_board,
    in_palace,
    own_half,
)

Move = tuple[tuple[int, int], tuple[int, int]]

# 正交四方向
_ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _add_if_target_ok(board: Board, moves: list, fr, fc, tr, tc, side) -> None:
    """若目标格在盘内且不是己方棋子,加入该走法。"""
    if not in_board(tr, tc):
        return
    target = board.get(tr, tc)
    if target == 0 or side_of(target) != side:
        moves.append(((fr, fc), (tr, tc)))


def _gen_general(board, moves, r, c, side):
    for dr, dc in _ORTHO:
        nr, nc = r + dr, c + dc
        if in_palace(nr, nc, side):
            _add_if_target_ok(board, moves, r, c, nr, nc, side)


def _gen_advisor(board, moves, r, c, side):
    for dr in (1, -1):
        for dc in (1, -1):
            nr, nc = r + dr, c + dc
            if in_palace(nr, nc, side):
                _add_if_target_ok(board, moves, r, c, nr, nc, side)


def _gen_elephant(board, moves, r, c, side):
    # 田字走法,象眼被占(塞象眼)则不可走;不可过河。
    for dr in (2, -2):
        for dc in (2, -2):
            nr, nc = r + dr, c + dc
            if not in_board(nr, nc):
                continue
            if not own_half(nr, side):  # 不可过河
                continue
            eye_r, eye_c = r + dr // 2, c + dc // 2
            if board.get(eye_r, eye_c) != 0:  # 塞象眼
                continue
            _add_if_target_ok(board, moves, r, c, nr, nc, side)


def _gen_horse(board, moves, r, c, side):
    # 马走日,先走一步直再走一步斜;别马腿则该方向不可走。
    # leg: 直行方向上的相邻格被占则蹩马腿。
    leg_moves = (
        # (leg_dr, leg_dc, [(dest_dr, dest_dc), ...])
        (1, 0, ((2, 1), (2, -1))),
        (-1, 0, ((-2, 1), (-2, -1))),
        (0, 1, ((1, 2), (-1, 2))),
        (0, -1, ((1, -2), (-1, -2))),
    )
    for leg_dr, leg_dc, dests in leg_moves:
        leg_r, leg_c = r + leg_dr, c + leg_dc
        if not in_board(leg_r, leg_c):
            continue  # 腿格出界,该方向两个落点也必然出界
        if board.get(leg_r, leg_c) != 0:
            continue  # 蹩马腿
        for dr, dc in dests:
            _add_if_target_ok(board, moves, r, c, r + dr, c + dc, side)


def _gen_chariot(board, moves, r, c, side):
    for dr, dc in _ORTHO:
        nr, nc = r + dr, c + dc
        while in_board(nr, nc):
            target = board.get(nr, nc)
            if target == 0:
                moves.append(((r, c), (nr, nc)))
            else:
                if side_of(target) != side:
                    moves.append(((r, c), (nr, nc)))
                break
            nr += dr
            nc += dc


def _gen_cannon(board, moves, r, c, side):
    for dr, dc in _ORTHO:
        nr, nc = r + dr, c + dc
        # 第一阶段:无子时可移动
        while in_board(nr, nc) and board.get(nr, nc) == 0:
            moves.append(((r, c), (nr, nc)))
            nr += dr
            nc += dc
        # 遇到炮架,跳过它继续找第一个棋子作为吃子目标
        nr += dr
        nc += dc
        while in_board(nr, nc):
            target = board.get(nr, nc)
            if target != 0:
                if side_of(target) != side:
                    moves.append(((r, c), (nr, nc)))
                break
            nr += dr
            nc += dc


def _gen_soldier(board, moves, r, c, side):
    forward = 1 if side == RED else -1
    # 前进
    _add_if_target_ok(board, moves, r, c, r + forward, c, side)
    # 过河后可左右
    if not own_half(r, side):
        _add_if_target_ok(board, moves, r, c, r, c + 1, side)
        _add_if_target_ok(board, moves, r, c, r, c - 1, side)


_GENERATORS = {
    GENERAL: _gen_general,
    ADVISOR: _gen_advisor,
    ELEPHANT: _gen_elephant,
    HORSE: _gen_horse,
    CHARIOT: _gen_chariot,
    CANNON: _gen_cannon,
    SOLDIER: _gen_soldier,
}


def pseudo_legal_moves(board: Board, side: int) -> list:
    """生成指定阵营所有伪合法走法。"""
    moves: list = []
    for r, c, piece in board.pieces_of(side):
        _GENERATORS[abs(piece)](board, moves, r, c, side)
    return moves


def generals_face(board: Board) -> bool:
    """白脸将:双方帅/将在同一列且中间无子,则为真(非法局面)。"""
    red_pos = board.find_general(RED)
    black_pos = board.find_general(BLACK)
    if red_pos is None or black_pos is None:
        return False
    rr, rc = red_pos
    br, bc = black_pos
    if rc != bc:
        return False
    lo, hi = (rr, br) if rr < br else (br, rr)
    for r in range(lo + 1, hi):
        if board.get(r, rc) != 0:
            return False
    return True


def is_attacked(board: Board, target_rc: tuple[int, int], by_side: int) -> bool:
    """判断 target_rc 是否被 by_side 阵营的某个棋子攻击。

    用于将军检测:目标通常是己方帅/将的位置。
    """
    tr, tc = target_rc
    for fr, fc, piece in board.pieces_of(by_side):
        pt = abs(piece)
        # 复用各棋子走法生成,看是否能走到目标格。
        sub: list = []
        _GENERATORS[pt](board, sub, fr, fc, by_side)
        for _, (dr, dc) in sub:
            if dr == tr and dc == tc:
                return True
    return False


def in_check(board: Board, side: int) -> bool:
    """判断 side 一方是否被将军(含白脸将)。"""
    if generals_face(board):
        return True
    gen_pos = board.find_general(side)
    if gen_pos is None:
        return True  # 帅/将已不在,视为已被将死
    return is_attacked(board, gen_pos, -side)


def legal_moves(board: Board, side: int) -> list:
    """生成合法走法:排除走后己方被将军或形成白脸将的走法。"""
    result: list = []
    for move in pseudo_legal_moves(board, side):
        from_rc, to_rc = move
        nxt = board.copy()
        nxt.move(from_rc, to_rc)
        if not in_check(nxt, side):
            result.append(move)
    return result
