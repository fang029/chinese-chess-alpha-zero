"""状态张量编码:把 GameState 编码为神经网络输入。

输入张量形状: (C, NUM_ROWS, NUM_COLS) = (15, 10, 9)
平面布局:
  0-6   : 当前走方的 7 种棋子 (帅仕相马车炮兵)
  7-13  : 对方的 7 种棋子
  14    : 轮走方标志(当前方为红则全 1,黑则全 0)

采用"当前走方视角":始终把轮到走的一方放在前 7 个平面,使网络具有
阵营对称性,无需分别学习红黑。价值 v 也从当前走方角度解释。
"""

from __future__ import annotations

import numpy as np

from .constants import NUM_ROWS, NUM_COLS, RED, PIECE_TYPES, PIECE_TYPE_TO_PLANE
from .game import GameState

NUM_PIECE_PLANES = len(PIECE_TYPES)        # 7
INPUT_CHANNELS = NUM_PIECE_PLANES * 2 + 1  # 15


def encode_state(state: GameState) -> np.ndarray:
    """返回 float32 张量 (INPUT_CHANNELS, NUM_ROWS, NUM_COLS),当前走方视角。"""
    planes = np.zeros((INPUT_CHANNELS, NUM_ROWS, NUM_COLS), dtype=np.float32)
    me = state.to_move
    grid = state.board.grid

    for r in range(NUM_ROWS):
        row = grid[r]
        for c in range(NUM_COLS):
            piece = row[c]
            if piece == 0:
                continue
            pt = abs(piece)
            plane = PIECE_TYPE_TO_PLANE[pt]
            if (piece > 0) == (me > 0):
                # 属于当前走方
                planes[plane, r, c] = 1.0
            else:
                planes[NUM_PIECE_PLANES + plane, r, c] = 1.0

    # 轮走方标志平面
    if me == RED:
        planes[INPUT_CHANNELS - 1, :, :] = 1.0

    return planes
