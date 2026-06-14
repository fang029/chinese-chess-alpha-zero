"""对局状态管理。

GameState 封装一局棋的完整状态:棋盘、轮走方、历史记录、终局判定。
这是上层(自我对弈、MCTS、UI)与规则引擎交互的主要接口。
"""

from __future__ import annotations

from .board import Board
from .constants import RED, BLACK
from . import movegen

Move = tuple[tuple[int, int], tuple[int, int]]

# 终局结果
ONGOING = "ongoing"
RED_WIN = "red_win"
BLACK_WIN = "black_win"
DRAW = "draw"

# 连续无吃子达到该回合数判和(单位:半步 ply)。60 个回合 = 120 半步。
_NO_CAPTURE_PLY_LIMIT = 120


class GameState:
    """一局中国象棋的可变状态。"""

    def __init__(self, board: Board | None = None, to_move: int = RED):
        self.board = board if board is not None else Board()
        self.to_move = to_move
        # 历史:每项为 (move, captured_piece, gave_check, mover_side),用于 undo
        # 与重复局面裁决(长将检测需知道每步是否将军及由谁走出)。
        self._history: list[tuple] = []
        # 无吃子半步计数,用于和棋判定。
        self._no_capture = 0
        # 局面出现次数,用于重复局面检测(键含轮走方)。
        self._position_counts: dict = {}
        # 局面键的有序序列,用于定位重复周期(长将裁决)。
        self._position_history: list = []
        self._record_position()

    def _position_key(self):
        return (hash(self.board), self.to_move)

    def _record_position(self):
        key = self._position_key()
        self._position_counts[key] = self._position_counts.get(key, 0) + 1
        self._position_history.append(key)

    def legal_moves(self) -> list:
        """当前轮走方的所有合法走法。"""
        return movegen.legal_moves(self.board, self.to_move)

    def push(self, move: Move) -> None:
        """执行一步走子并切换轮走方。不校验合法性(调用方应传入合法走法)。"""
        from_rc, to_rc = move
        mover = self.to_move
        captured = self.board.move(from_rc, to_rc)
        # 走子后对手是否被将军(用于长将检测)。
        gave_check = movegen.in_check(self.board, -mover)
        self._history.append((move, captured, gave_check, mover))
        if captured != 0:
            self._no_capture = 0
        else:
            self._no_capture += 1
        self.to_move = -self.to_move
        self._record_position()

    def pop(self) -> Move:
        """悔一步,恢复到上一状态,返回被撤销的走法。"""
        # 先撤销当前局面计数
        key = self._position_key()
        self._position_counts[key] -= 1
        if self._position_counts[key] == 0:
            del self._position_counts[key]
        self._position_history.pop()

        move, captured, _gave_check, _mover = self._history.pop()
        from_rc, to_rc = move
        self.to_move = -self.to_move
        # 把棋子移回原位,并恢复被吃子
        self.board.move(to_rc, from_rc)
        self.board.set(to_rc[0], to_rc[1], captured)

        # 重算无吃子计数:简单起见从历史回放更稳妥,但代价高;
        # 这里采用增量恢复——记录足够信息较复杂,故重新扫描历史。
        self._recompute_no_capture()
        return move

    def _recompute_no_capture(self):
        count = 0
        for entry in self._history:
            captured = entry[1]
            count = 0 if captured != 0 else count + 1
        self._no_capture = count

    def is_repetition(self, times: int = 3) -> bool:
        """当前局面是否已出现 times 次(默认三次同局)。"""
        return self._position_counts.get(self._position_key(), 0) >= times

    def _perpetual_check_loser(self):
        """三次重复局面时判断是否为长将,返回判负的一方;非长将返回 None。

        竞技规则:一方连续将军、靠重复局面逼和则判负(长将)。判定方式:取
        当前重复局面最近一个重复周期内的走子,看是否某一方在其每一步都将军,
        而另一方没有连续将军——该连续将军方判负。双方都长将或都不长将则按
        重复和棋处理(长捉等更复杂情形不在此裁决,保持判和)。
        """
        key = self._position_key()
        # 找到上一次出现同一局面的位置,界定一个重复周期。
        # _position_history[-1] 是当前局面,向前找相同 key。
        last = len(self._position_history) - 1
        prev = None
        for i in range(last - 1, -1, -1):
            if self._position_history[i] == key:
                prev = i
                break
        if prev is None:
            return None

        # 周期内的走子对应 history[prev:last](每步 push 后追加一个局面)。
        # history 与 position_history 错位 1:position_history[0] 是初始局面,
        # history[k] 产生 position_history[k+1]。故周期走子为 history[prev:last]。
        cycle = self._history[prev:last]
        if not cycle:
            return None

        # 按走子方分组,统计各方在周期内是否“步步将军”。
        checks = {RED: [], BLACK: []}
        for entry in cycle:
            gave_check, mover = entry[2], entry[3]
            checks[mover].append(gave_check)

        red_perp = bool(checks[RED]) and all(checks[RED])
        black_perp = bool(checks[BLACK]) and all(checks[BLACK])

        if red_perp and not black_perp:
            return RED      # 红长将,红负
        if black_perp and not red_perp:
            return BLACK    # 黑长将,黑负
        return None         # 双方对等,按和棋

    def result(self) -> str:
        """返回终局结果。需要在轮到某方走棋时调用。

        将死/困毙: 当前方无合法走法 -> 对方胜。
        无吃子超限或三次重复: 和棋(简化处理)。
        """
        if not self.legal_moves():
            # 无棋可走:被将死或困毙,均判当前方负。
            return BLACK_WIN if self.to_move == RED else RED_WIN
        if self._no_capture >= _NO_CAPTURE_PLY_LIMIT:
            return DRAW
        if self.is_repetition(3):
            # 长将判负:连续将军逼和的一方判负;否则按重复和棋。
            loser = self._perpetual_check_loser()
            if loser == RED:
                return BLACK_WIN
            if loser == BLACK:
                return RED_WIN
            return DRAW
        return ONGOING

    def is_terminal(self) -> bool:
        return self.result() != ONGOING

    def copy(self) -> "GameState":
        new = GameState.__new__(GameState)
        new.board = self.board.copy()
        new.to_move = self.to_move
        new._history = list(self._history)
        new._no_capture = self._no_capture
        new._position_counts = dict(self._position_counts)
        new._position_history = list(self._position_history)
        return new

    def __repr__(self) -> str:
        side = "红" if self.to_move == RED else "黑"
        return f"{self.board.to_string(use_chinese=True)}\n轮到: {side}方"
