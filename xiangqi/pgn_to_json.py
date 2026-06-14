"""把中文记谱 PGN 文件转成 pretrain.py 需要的 JSON 格式。

输入:标准中文记谱 PGN(每局含 [Event]/[Result] 等标签 + 着法行,
着法形如 "1. 炮二平五 马8进7 2. ...")。
输出:[{"moves": ["炮二平五", "马8进7", ...], "result": "red_win"/...}, ...]

用法:
  python -m xiangqi.pgn_to_json games.pgn -o records.json

Result 标签映射:1-0 红胜,0-1 黑胜,其余(1/2-1/2、*、无)记和/未知。
"""

from __future__ import annotations

import argparse
import json
import re


def _result_to_str(tag: str) -> str:
    tag = (tag or "").strip()
    if tag == "1-0":
        return "red_win"
    if tag == "0-1":
        return "black_win"
    return "draw"


def parse_pgn(text: str) -> list:
    """把中文记谱 PGN 文本解析为 [{"moves":[...], "result":...}, ...]。

    PGN 里 header 标签块与着法块可能被空行分开,故不能按空行块孤立处理。
    做法:顺序扫描所有块,用最近一次见到的 [Result] 标签作为随后着法块的结果。
    """
    games = []
    blocks = re.split(r"\n\s*\n", text.strip())
    pending_result = "draw"  # 最近 header 的结果,供随后的着法块使用
    for blk in blocks:
        result_match = re.search(r'\[Result\s+"([^"]+)"\]', blk)
        if result_match:
            pending_result = _result_to_str(result_match.group(1))

        # 着法行:不以 '[' 开头的非空行(着法可能跨多行)。
        move_lines = [ln for ln in blk.splitlines()
                      if ln.strip() and not ln.lstrip().startswith("[")]
        if not move_lines:
            continue
        move_text = " ".join(move_lines)
        # 去掉回合号 "1." "23." 与结果标记。
        move_text = re.sub(r"\d+\s*\.", " ", move_text)
        move_text = re.sub(r"(1-0|0-1|1/2-1/2|\*)", " ", move_text)
        tokens = move_text.split()
        # 着法是 4 个汉字/全角数字组成的串;过滤掉杂项 token。
        moves = [t for t in tokens if len(t) == 4]
        if moves:
            games.append({"moves": moves, "result": pending_result})
            pending_result = "draw"  # 用过即重置,避免串到下一局
    return games


def main():
    p = argparse.ArgumentParser(description="中文记谱 PGN -> pretrain JSON")
    p.add_argument("pgn", help="输入 PGN 文件(中文记谱)")
    p.add_argument("-o", "--out", default="records.json", help="输出 JSON")
    args = p.parse_args()

    with open(args.pgn, "r", encoding="utf-8") as f:
        text = f.read()
    games = parse_pgn(text)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(games, f, ensure_ascii=False)
    total_moves = sum(len(g["moves"]) for g in games)
    print(f"解析 {len(games)} 局,共 {total_moves} 步,写入 {args.out}")


if __name__ == "__main__":
    main()
