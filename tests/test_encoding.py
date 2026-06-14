"""编码层测试:动作表完整性、双向映射、掩码、状态张量。"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.game import GameState
from xiangqi.constants import RED, BLACK
from xiangqi import encoding
from xiangqi import movegen
from xiangqi.state_encoder import encode_state, INPUT_CHANNELS
from xiangqi.constants import NUM_ROWS, NUM_COLS


def test_action_table_bijective():
    """index <-> move 双向一致。"""
    for i in range(encoding.ACTION_SIZE):
        m = encoding.index_to_move(i)
        assert encoding.move_to_index(m) == i


def test_action_size_reasonable():
    """动作空间大小应在合理区间(数千)。"""
    assert 1500 < encoding.ACTION_SIZE < 3000


def test_action_table_covers_initial_moves():
    """开局所有合法走法都能被编码。"""
    g = GameState()
    for m in g.legal_moves():
        idx = encoding.move_to_index(m)  # 不抛 KeyError 即通过
        assert 0 <= idx < encoding.ACTION_SIZE


def test_action_table_covers_random_games():
    """随机对弈中出现的每个合法走法都必须在动作表内。

    这是动作表完整性的关键验证:若有遗漏会抛 KeyError。
    """
    rng = random.Random(7)
    for _ in range(20):
        g = GameState()
        for _ in range(120):
            if g.is_terminal():
                break
            moves = g.legal_moves()
            for m in moves:
                encoding.move_to_index(m)  # 全部可编码
            g.push(rng.choice(moves))


def test_legal_mask():
    g = GameState()
    moves = g.legal_moves()
    mask = encoding.legal_action_mask(moves)
    assert len(mask) == encoding.ACTION_SIZE
    assert sum(mask) == len(moves)
    for m in moves:
        assert mask[encoding.move_to_index(m)] == 1


def test_encode_state_shape():
    g = GameState()
    t = encode_state(g)
    assert t.shape == (INPUT_CHANNELS, NUM_ROWS, NUM_COLS)
    assert t.dtype.name == "float32"


def test_encode_state_perspective():
    """当前走方视角:红走时标志平面全 1,黑走时全 0;
    且前 7 平面始终是当前方棋子。"""
    g = GameState()  # 红先
    t_red = encode_state(g)
    assert t_red[INPUT_CHANNELS - 1].min() == 1.0
    # 红帅在 row0 col4,应出现在当前方(前7)的将平面(索引0)
    assert t_red[0, 0, 4] == 1.0
    assert t_red[7, 0, 4] == 0.0

    g.push(g.legal_moves()[0])  # 轮到黑
    t_black = encode_state(g)
    assert t_black[INPUT_CHANNELS - 1].max() == 0.0
    # 此时当前方是黑,黑将在 row9 col4 应出现在前7的将平面
    assert t_black[0, 9, 4] == 1.0
    # 红帅(对方)应在后7平面
    assert t_black[7, 0, 4] == 1.0


def test_piece_counts_in_encoding():
    """开局每方应有 16 个棋子,编码后前7与后7平面各 16 个 1。"""
    import numpy as np
    g = GameState()
    t = encode_state(g)
    assert np.sum(t[0:7]) == 16
    assert np.sum(t[7:14]) == 16


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
