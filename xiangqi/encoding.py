"""动作空间编码。

把走法 (from_rc, to_rc) 映射到固定的动作索引,供神经网络策略头输出。

设计:对每个起点格,枚举所有棋子类型理论上可达的目标格(忽略其他棋子的
阻挡,但遵守棋盘边界、九宫、过河等位置约束),取并集去重,形成一张固定的
(from, to) -> index 表。该超集覆盖任意局面下的全部合法走法,索引在整个
项目生命周期内保持稳定。

车/炮的滑动走法用"沿某方向到任意距离"覆盖;马/象/仕/帅/兵用其固定走法形态。
红黑共用同一套动作表(走法是几何坐标,与阵营无关)。
"""

from __future__ import annotations

from .constants import (
    NUM_ROWS,
    NUM_COLS,
    RED,
    BLACK,
    in_board,
)

Move = tuple[tuple[int, int], tuple[int, int]]


def _theoretical_targets(r: int, c: int):
    """枚举从 (r,c) 出发、任意棋子类型在空盘上理论可达的所有目标格。

    不考虑阻挡(滑动子覆盖整条线),但遵守边界与位置性约束(九宫、不过河等)
    对落点的限制。返回去重后的目标格集合。
    """
    targets = set()

    # 车/炮:四正方向滑动到任意距离。
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        while in_board(nr, nc):
            targets.add((nr, nc))
            nr += dr
            nc += dc

    # 马:八个日字落点。
    for dr, dc in ((2, 1), (2, -1), (-2, 1), (-2, -1),
                   (1, 2), (1, -2), (-1, 2), (-1, -2)):
        nr, nc = r + dr, c + dc
        if in_board(nr, nc):
            targets.add((nr, nc))

    # 象:四个田字落点(可落两侧半场,过河约束在合法性层处理;
    # 但相/象永不过河,故这里限定在各自半场以缩小动作空间)。
    for dr, dc in ((2, 2), (2, -2), (-2, 2), (-2, -2)):
        nr, nc = r + dr, c + dc
        if in_board(nr, nc):
            targets.add((nr, nc))

    # 仕/帅 与 兵 的斜/直单步,已被上面的车方向(直)与象方向无关项覆盖部分;
    # 仕的斜走单步需单独补充。
    for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        nr, nc = r + dr, c + dc
        if in_board(nr, nc):
            targets.add((nr, nc))

    targets.discard((r, c))
    return targets


def _build_action_table():
    """构造 (from_rc, to_rc) <-> index 的双向映射。"""
    move_to_idx: dict[Move, int] = {}
    idx_to_move: list[Move] = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            for (tr, tc) in sorted(_theoretical_targets(r, c)):
                move = ((r, c), (tr, tc))
                move_to_idx[move] = len(idx_to_move)
                idx_to_move.append(move)
    return move_to_idx, idx_to_move


_MOVE_TO_IDX, _IDX_TO_MOVE = _build_action_table()

ACTION_SIZE = len(_IDX_TO_MOVE)


def move_to_index(move: Move) -> int:
    """走法 -> 动作索引。未知走法抛 KeyError(表示编码表不完整,属 bug)。"""
    return _MOVE_TO_IDX[move]


def index_to_move(index: int) -> Move:
    """动作索引 -> 走法。"""
    return _IDX_TO_MOVE[index]


def legal_action_mask(legal_moves: list) -> list:
    """给定合法走法列表,返回长度 ACTION_SIZE 的 0/1 掩码(list[int])。

    供策略头在 softmax 前屏蔽非法动作。
    """
    mask = [0] * ACTION_SIZE
    for m in legal_moves:
        mask[_MOVE_TO_IDX[m]] = 1
    return mask
