"""棋盘状态表示。

Board 仅负责存储棋子布局与基本的格子读写/显示,不含走子规则。
走子合法性与生成由 movegen 模块负责,以保持职责单一。
"""

from __future__ import annotations

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
    piece_char,
    piece_name,
    side_of,
    in_board,
)


def _initial_grid() -> list[list[int]]:
    """构造标准开局布局,返回 10x9 的二维列表。"""
    g = [[0] * NUM_COLS for _ in range(NUM_ROWS)]

    # 红方底线 (row 0)
    g[0] = [CHARIOT, HORSE, ELEPHANT, ADVISOR, GENERAL,
            ADVISOR, ELEPHANT, HORSE, CHARIOT]
    # 红方炮 (row 2)
    g[2][1] = CANNON
    g[2][7] = CANNON
    # 红方兵 (row 3)
    for c in range(0, NUM_COLS, 2):
        g[3][c] = SOLDIER

    # 黑方底线 (row 9),取负
    g[9] = [-CHARIOT, -HORSE, -ELEPHANT, -ADVISOR, -GENERAL,
            -ADVISOR, -ELEPHANT, -HORSE, -CHARIOT]
    # 黑方炮 (row 7)
    g[7][1] = -CANNON
    g[7][7] = -CANNON
    # 黑方卒 (row 6)
    for c in range(0, NUM_COLS, 2):
        g[6][c] = -SOLDIER

    return g


class Board:
    """存储棋盘格子状态。坐标 (row, col),内容见 constants 的棋子编码。"""

    __slots__ = ("grid",)

    def __init__(self, grid: list[list[int]] | None = None):
        if grid is None:
            self.grid = _initial_grid()
        else:
            self.grid = [row[:] for row in grid]

    @classmethod
    def empty(cls) -> "Board":
        """返回空棋盘。"""
        return cls([[0] * NUM_COLS for _ in range(NUM_ROWS)])

    def copy(self) -> "Board":
        return Board(self.grid)

    def get(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def set(self, row: int, col: int, piece: int) -> None:
        self.grid[row][col] = piece

    def move(self, from_rc: tuple[int, int], to_rc: tuple[int, int]) -> int:
        """执行一步移动,返回被吃掉的棋子(0 表示无)。不校验合法性。"""
        fr, fc = from_rc
        tr, tc = to_rc
        piece = self.grid[fr][fc]
        captured = self.grid[tr][tc]
        self.grid[tr][tc] = piece
        self.grid[fr][fc] = 0
        return captured

    def find_general(self, side: int) -> tuple[int, int] | None:
        """返回指定阵营帅/将的位置,找不到返回 None。"""
        target = GENERAL * side
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                if self.grid[r][c] == target:
                    return (r, c)
        return None

    def pieces_of(self, side: int):
        """生成指定阵营所有棋子的 (row, col, piece)。"""
        for r in range(NUM_ROWS):
            for c in range(NUM_COLS):
                p = self.grid[r][c]
                if p != 0 and side_of(p) == side:
                    yield (r, c, p)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Board) and self.grid == other.grid

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.grid))

    def to_string(self, use_chinese: bool = False) -> str:
        """返回多行字符串,row 9 在顶部、row 0 在底部,贴近实际棋盘视角。"""
        render = piece_name if use_chinese else piece_char
        lines = []
        for r in range(NUM_ROWS - 1, -1, -1):
            cells = " ".join(render(self.grid[r][c]) for c in range(NUM_COLS))
            lines.append(f"{r} {cells}")
            if r == 5:
                lines.append("  ----- 楚河  汉界 -----")
        col_header = "  " + " ".join(str(c) for c in range(NUM_COLS))
        lines.append(col_header)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_string()
