"""ICCS 记谱解析与 PGN 转换测试。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.game import GameState
from xiangqi.iccs_notation import parse_iccs, parse_iccs_pgn
from xiangqi.pgn_to_json import parse_pgn
from xiangqi.pretrain import build_dataset


def test_parse_iccs_basic():
    # H2-E2:列 H=7 行 2 -> 列 E=4 行 2,即炮二平五 (2,7)->(2,4)
    assert parse_iccs("H2-E2") == ((2, 7), (2, 4))
    # 无连字符、小写也支持
    assert parse_iccs("h2e2") == ((2, 7), (2, 4))


def test_parse_iccs_matches_chinese_opening():
    """ICCS 首手应与中文记谱炮二平五落到同一坐标且合法。"""
    g = GameState()
    move = parse_iccs("H2-E2")
    legal = set(map(tuple, g.legal_moves()))
    assert move in legal


def test_parse_iccs_invalid():
    import pytest
    with pytest.raises(ValueError):
        parse_iccs("炮二平五")
    with pytest.raises(ValueError):
        parse_iccs("Z9-Z9")


def test_iccs_full_game_legal():
    """一段真实 UCI/ICCS 开局应逐手合法落地。"""
    moves = ["h2e2", "h9g7", "h0g2", "i9h9", "c3c4", "g6g5"]
    g = GameState()
    for mv in moves:
        m = parse_iccs(mv)
        assert tuple(m) in set(map(tuple, g.legal_moves())), f"{mv} 非法"
        g.push(m)


def test_parse_iccs_pgn_with_result():
    text = (
        '[Event "T"]\n[Result "1-0"]\n\n'
        '1. H2-E2 H9-G7 2. H0-G2 I9-H9\n'
    )
    games = parse_iccs_pgn(text)
    assert len(games) == 1
    assert games[0]["result"] == "red_win"
    assert games[0]["moves"][:2] == ["H2-E2", "H9-G7"]


def test_build_dataset_iccs():
    games = [{"moves": ["h2e2", "h9g7", "h0g2"], "result": "red_win"}]
    states, idx, zs = build_dataset(games, notation="iccs")
    assert len(states) == 3
    assert zs[0] == 1.0  # 红胜,首手红方视角 +1


def test_pgn_to_json_chinese_result_mapping():
    text = (
        '[Event "A"]\n[Result "1-0"]\n\n1. 炮二平五 马8进7\n\n'
        '[Event "B"]\n[Result "0-1"]\n\n1. 炮八平五 马2进3\n'
    )
    games = parse_pgn(text)
    assert len(games) == 2
    assert games[0]["result"] == "red_win"
    assert games[1]["result"] == "black_win"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
