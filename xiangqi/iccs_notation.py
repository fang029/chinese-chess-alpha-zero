"""ICCS 坐标记谱解析:把 "C3-C4" 这类着法转为 ((fr,fc),(tr,tc))。

ICCS(Internet Chinese Chess Server)格式:
  着法 = 起点 + '-' + 终点,如 "H2-E2"、"B7-C5"。
  列用字母 A-I,数字用 0-9 表示行。

坐标系约定(与本引擎 (row, col) 对应):
  - 列字母 A..I -> col 0..8(A 为红方左手第一列)。
  - 行数字 0..9 -> row 0..9(0 为红方底线,9 为黑方底线)。
  即 ICCS "<字母><数字>" 直接映射为 (row=数字, col=字母序号)。

这比中文记谱简单可靠:直接给出起点终点坐标,无需按棋子类型推断方向。
解析失败(格式不符)抛 ValueError。
"""

from __future__ import annotations

import re

# 形如 H2-E2 / h2-e2,允许中间无连字符(H2E2)。
_ICCS_RE = re.compile(r"^([A-Ia-i])(\d)[-]?([A-Ia-i])(\d)$")


def _col(letter: str) -> int:
    return ord(letter.upper()) - ord("A")


def parse_iccs(text: str):
    """把一条 ICCS 着法解析为 ((fr,fc),(tr,tc))。失败抛 ValueError。"""
    m = _ICCS_RE.match(text.strip())
    if not m:
        raise ValueError(f"非 ICCS 着法: {text}")
    fc = _col(m.group(1))
    fr = int(m.group(2))
    tc = _col(m.group(3))
    tr = int(m.group(4))
    return ((fr, fc), (tr, tc))


def parse_iccs_pgn(text: str) -> list:
    """解析 ICCS 格式 PGN 文本为 [{"moves":[ICCS串...], "result":...}, ...]。

    与 pgn_to_json 类似,但着法 token 是 ICCS 串(如 C3-C4),非中文。
    用最近的 [Result] 标签作为随后着法块的结果。
    """
    games = []
    blocks = re.split(r"\n\s*\n", text.strip())
    pending_result = "draw"
    for blk in blocks:
        rm = re.search(r'\[Result\s+"([^"]+)"\]', blk)
        if rm:
            tag = rm.group(1).strip()
            pending_result = ("red_win" if tag == "1-0" else
                              "black_win" if tag == "0-1" else "draw")
        move_lines = [ln for ln in blk.splitlines()
                      if ln.strip() and not ln.lstrip().startswith("[")]
        if not move_lines:
            continue
        move_text = " ".join(move_lines)
        move_text = re.sub(r"\d+\s*\.", " ", move_text)       # 去回合号
        move_text = re.sub(r"(1-0|0-1|1/2-1/2|\*)", " ", move_text)
        tokens = move_text.split()
        moves = [t for t in tokens if _ICCS_RE.match(t)]
        if moves:
            games.append({"moves": moves, "result": pending_result})
            pending_result = "draw"
    return games
