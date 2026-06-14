"""中文记谱解析:把"炮二平五"这类着法转换为 ((fr,fc),(tr,tc)) 坐标走法。

记谱规则(标准中文纵线记谱):
  着法 = [棋子][纵线] [动作][目标]
  - 纵线:红方用汉字 一~九,从红方视角自右向左数;黑方用数字 1~9,从黑方
    视角自右向左数。本引擎坐标 col 0~8,故:
      红: 纵线 f -> col = 9 - f   (红一 = col8, 红九 = col0)
      黑: 纵线 f -> col = f - 1   (黑1  = col0, 黑9  = col8)
  - 动作:进(向前)/退(向后)/平(横走)。红前为 row 增,黑前为 row 减。
  - 直行子(车炮兵将)进退后数字为步数,平后数字为目标纵线。
  - 斜行子(马相仕)进退后数字为目标纵线(必变线)。
  - 同线两子用 前/后 前缀指定,如 前马、后炮。

局限:同线三子及以上(多兵用 前中后/数字)未实现;仅解析单子或同线两子。
"""

from __future__ import annotations

from .constants import (RED, BLACK, GENERAL, ADVISOR, ELEPHANT, HORSE,
                        CHARIOT, CANNON, SOLDIER, NUM_ROWS, NUM_COLS, side_of)

# 汉字数字 <-> 值
_CN_NUM = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}
# 阿拉伯数字(半角与全角)
_AR_NUM = {str(i): i for i in range(1, 10)}
_AR_NUM.update({chr(ord("１") + i): i + 1 for i in range(9)})

# 棋子名(红/黑写法都映射到类型)
_NAME_TO_TYPE = {
    "帅": GENERAL, "将": GENERAL,
    "仕": ADVISOR, "士": ADVISOR,
    "相": ELEPHANT, "象": ELEPHANT,
    "马": HORSE, "馬": HORSE,
    "车": CHARIOT, "車": CHARIOT,
    "炮": CANNON, "砲": CANNON, "炮兵": CANNON,
    "兵": SOLDIER, "卒": SOLDIER,
}

_STRAIGHT = {CHARIOT, CANNON, SOLDIER, GENERAL}   # 进退按步数
_DIAGONAL = {HORSE, ELEPHANT, ADVISOR}            # 进退按目标纵线

_FORWARD = {"进", "進"}
_BACKWARD = {"退"}
_HORIZONTAL = {"平"}
_FRONT = {"前"}
_BACK = {"后", "後"}


def _num_value(ch: str):
    if ch in _CN_NUM:
        return _CN_NUM[ch]
    if ch in _AR_NUM:
        return _AR_NUM[ch]
    return None


def _file_to_col(file_num: int, side: int) -> int:
    return (9 - file_num) if side == RED else (file_num - 1)


def _find_pieces(board, ptype: int, side: int):
    """返回该方该类型所有棋子坐标 [(r,c), ...]。"""
    target = ptype if side == RED else -ptype
    return [(r, c) for r in range(NUM_ROWS) for c in range(NUM_COLS)
            if board.get(r, c) == target]


def _resolve_source(board, ptype, side, second_char, first_char):
    """确定起子坐标。

    普通着法:first_char 是棋子名,second_char 是纵线号。
    前/后着法:first_char 是 前/后,second_char 是棋子名。
    返回 (from_rc, candidates_on_file) 或抛 ValueError。
    """
    if first_char in _FRONT or first_char in _BACK:
        # 前/后 + 棋子名:同纵线上多个同类子,按"前/后"取。
        cands = _find_pieces(board, ptype, side)
        # 按纵线分组找出有 >=2 子的纵线;通常题面只有一条这样的线。
        from collections import defaultdict
        by_col = defaultdict(list)
        for (r, c) in cands:
            by_col[c].append((r, c))
        multi = [col for col, lst in by_col.items() if len(lst) >= 2]
        if not multi:
            raise ValueError("前/后 着法但未找到同纵线两子")
        col = multi[0]
        group = sorted(by_col[col], key=lambda rc: rc[0])  # row 升序
        # 红方"前"= row 大(更靠黑方);黑方"前"= row 小。
        if side == RED:
            front, back = group[-1], group[0]
        else:
            front, back = group[0], group[-1]
        return front if first_char in _FRONT else back

    # 普通:棋子名 + 纵线
    file_num = _num_value(second_char)
    if file_num is None:
        raise ValueError(f"无法解析纵线: {second_char}")
    col = _file_to_col(file_num, side)
    cands = [rc for rc in _find_pieces(board, ptype, side) if rc[1] == col]
    if not cands:
        raise ValueError(f"纵线 {second_char} 上无 {ptype} 子")
    if len(cands) == 1:
        return cands[0]
    # 同纵线多子但未用前后:取最靠前者(罕见,容错)。
    cands.sort(key=lambda rc: rc[0])
    return cands[-1] if side == RED else cands[0]


def parse_move(board, text: str, side: int):
    """把一条中文着法解析为 ((fr,fc),(tr,tc))。失败抛 ValueError。"""
    text = text.strip()
    if len(text) < 4:
        raise ValueError(f"着法过短: {text}")
    c0, c1, action, c3 = text[0], text[1], text[2], text[3]

    # 棋子类型:普通着法在 c0,前后着法在 c1。
    if c0 in _FRONT or c0 in _BACK:
        ptype = _NAME_TO_TYPE.get(c1)
    else:
        ptype = _NAME_TO_TYPE.get(c0)
    if ptype is None:
        raise ValueError(f"未知棋子: {text}")

    fr, fc = _resolve_source(board, ptype, side, c1, c0)

    num = _num_value(c3)
    if action in _HORIZONTAL:
        # 平:横走到目标纵线,row 不变。
        if num is None:
            raise ValueError(f"平 着法缺目标纵线: {text}")
        tc = _file_to_col(num, side)
        return ((fr, fc), (fr, tc))

    forward = action in _FORWARD
    backward = action in _BACKWARD
    if not (forward or backward):
        raise ValueError(f"未知动作: {action}")

    # 红方"进"= row 增;黑方"进"= row 减。
    sign = 1 if side == RED else -1
    direction = sign if forward else -sign

    if ptype in _STRAIGHT:
        # 直行子:num 为步数,纵线不变。
        steps = num
        if steps is None:
            raise ValueError(f"进退缺步数: {text}")
        return ((fr, fc), (fr + direction * steps, fc))

    # 斜行子(马/相/仕):num 为目标纵线,row 位移由棋子固定走法+方向定。
    if num is None:
        raise ValueError(f"斜行子进退缺目标纵线: {text}")
    tc = _file_to_col(num, side)
    dc = abs(tc - fc)
    if ptype == HORSE:
        dr = 2 if dc == 1 else 1   # 马:一方向走 2,另一方向走 1
    elif ptype == ELEPHANT:
        dr = 2                     # 相:田字,row 变 2
    else:  # ADVISOR
        dr = 1                     # 仕:斜一步
    return ((fr, fc), (fr + direction * dr, tc))
