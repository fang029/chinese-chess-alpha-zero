"""评估门控(arena):让两个网络对弈,按胜率决定新网络是否晋级。

AlphaGo Zero 做法:新训练的网络(challenger)与当前最佳网络(champion)对弈
若干局,胜率超过阈值才替换 champion;否则保留旧网络继续自我对弈,避免一次
坏训练污染数据分布。AlphaZero 简化掉了这步(始终用最新网络),此处作为可选项
提供给希望更稳健训练的场景。

对弈用贪心走子(temperature=0,不加 Dirichlet 噪声),与 play.py 的 AIPlayer
一致,以反映网络的真实棋力。为消除先手优势,两个网络轮流执红。
"""

from __future__ import annotations

from .game import GameState
from .mcts import MCTS, action_probabilities
from .constants import RED, BLACK


class ArenaConfig:
    def __init__(self,
                 num_games: int = 20,
                 num_simulations: int = 200,
                 win_threshold: float = 0.55,
                 max_moves: int = 300,
                 c_puct: float = 1.5,
                 batch_size: int = 8):
        self.num_games = num_games
        self.num_simulations = num_simulations
        # 晋级阈值:challenger 的得分率(胜=1,和=0.5)需 >= 此值。
        self.win_threshold = win_threshold
        self.max_moves = max_moves
        self.c_puct = c_puct
        self.batch_size = batch_size  # MCTS 批量推理叶节点数


def _greedy_move(mcts: MCTS, state: GameState, simulations: int):
    """贪心选子:访问数最多的走法,不加噪声(评估真实棋力)。"""
    root = mcts.run(state, simulations, add_noise=False)
    pi = action_probabilities(root, temperature=0.0)
    return max(pi.items(), key=lambda kv: kv[1])[0]


def play_match_game(red_mcts: MCTS, black_mcts: MCTS,
                    simulations: int, max_moves: int) -> str:
    """红方用 red_mcts、黑方用 black_mcts 对弈一局,返回终局结果字符串。

    超过 max_moves 未分胜负判和(与自我对弈一致,避免无限局)。
    """
    state = GameState()
    move_count = 0
    while not state.is_terminal() and move_count < max_moves:
        mcts = red_mcts if state.to_move == RED else black_mcts
        move = _greedy_move(mcts, state, simulations)
        state.push(move)
        move_count += 1
    result = state.result()
    # 超步未分胜负:与自我对弈一致按和棋处理(result() 此时返回 "ongoing")。
    return result if result != "ongoing" else "draw"


def compare(challenger_eval, champion_eval, config: ArenaConfig | None = None):
    """让 challenger 与 champion 对弈 config.num_games 局,返回评估结果。

    两者轮流执红消除先手优势。返回 dict:
      wins/losses/draws: 以 challenger 视角统计
      score:             challenger 得分率(胜 1、和 0.5、负 0),范围 [0, 1]
      promote:           score 是否达到晋级阈值
    """
    config = config or ArenaConfig()
    challenger_mcts = MCTS(challenger_eval, c_puct=config.c_puct,
                           batch_size=config.batch_size)
    champion_mcts = MCTS(champion_eval, c_puct=config.c_puct,
                         batch_size=config.batch_size)

    wins = losses = draws = 0
    for g in range(config.num_games):
        challenger_is_red = (g % 2 == 0)
        if challenger_is_red:
            red_mcts, black_mcts = challenger_mcts, champion_mcts
        else:
            red_mcts, black_mcts = champion_mcts, challenger_mcts

        result = play_match_game(
            red_mcts, black_mcts, config.num_simulations, config.max_moves)

        if result == "draw":
            draws += 1
        else:
            red_won = (result == "red_win")
            challenger_won = (red_won == challenger_is_red)
            if challenger_won:
                wins += 1
            else:
                losses += 1

    played = wins + losses + draws
    score = (wins + 0.5 * draws) / played if played else 0.0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": played,
        "score": score,
        "promote": score >= config.win_threshold,
    }
