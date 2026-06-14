"""棋盘与棋子的基础常量定义。

坐标约定:
  - 棋盘 10 行 x 9 列,使用 (row, col)。
  - row 0 为红方底线,row 9 为黑方底线。
  - 楚河汉界位于 row 4 与 row 5 之间。
  - 红方九宫: row 0-2, col 3-5;黑方九宫: row 7-9, col 3-5。

棋子编码:正数为红方,负数为黑方,0 为空格。
  绝对值: 1=帅/将 2=仕/士 3=相/象 4=马 5=车 6=炮 7=兵/卒
"""

# 棋盘尺寸
NUM_ROWS = 10
NUM_COLS = 9

# 阵营
RED = 1
BLACK = -1

# 棋子类型(取绝对值)
GENERAL = 1   # 帅 / 将
ADVISOR = 2   # 仕 / 士
ELEPHANT = 3  # 相 / 象
HORSE = 4     # 马
CHARIOT = 5   # 车
CANNON = 6    # 炮
SOLDIER = 7   # 兵 / 卒

PIECE_TYPES = (GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER)

# 每个棋子类型在网络输入张量中的平面索引(0-based,红黑各一组)
# 由编码模块使用,集中定义以便单一来源。
PIECE_TYPE_TO_PLANE = {pt: i for i, pt in enumerate(PIECE_TYPES)}

# 用于显示的单字符(红大写,黑小写)
_PIECE_CHARS = {
    GENERAL: ("K", "k"),
    ADVISOR: ("A", "a"),
    ELEPHANT: ("E", "e"),
    HORSE: ("H", "h"),
    CHARIOT: ("R", "r"),
    CANNON: ("C", "c"),
    SOLDIER: ("P", "p"),
}

# 用于显示的中文棋子名(红, 黑)
_PIECE_NAMES = {
    GENERAL: ("帅", "将"),
    ADVISOR: ("仕", "士"),
    ELEPHANT: ("相", "象"),
    HORSE: ("马", "马"),
    CHARIOT: ("车", "车"),
    CANNON: ("炮", "炮"),
    SOLDIER: ("兵", "卒"),
}


def piece_char(piece: int) -> str:
    """返回棋子的单字符表示,空格为 '.'。"""
    if piece == 0:
        return "."
    pt = abs(piece)
    red_char, black_char = _PIECE_CHARS[pt]
    return red_char if piece > 0 else black_char


def piece_name(piece: int) -> str:
    """返回棋子的中文名,空格为 '·'。"""
    if piece == 0:
        return "·"
    pt = abs(piece)
    red_name, black_name = _PIECE_NAMES[pt]
    return red_name if piece > 0 else black_name


def side_of(piece: int) -> int:
    """返回棋子所属阵营 RED / BLACK,空格返回 0。"""
    if piece > 0:
        return RED
    if piece < 0:
        return BLACK
    return 0


def in_board(row: int, col: int) -> bool:
    """判断坐标是否在棋盘内。"""
    return 0 <= row < NUM_ROWS and 0 <= col < NUM_COLS


def in_palace(row: int, col: int, side: int) -> bool:
    """判断坐标是否在指定阵营的九宫内。"""
    if col < 3 or col > 5:
        return False
    if side == RED:
        return 0 <= row <= 2
    return 7 <= row <= 9


def own_half(row: int, side: int) -> bool:
    """判断 row 是否在指定阵营的己方半场(未过河)。"""
    if side == RED:
        return row <= 4
    return row >= 5
