"""Web 对弈服务:用标准库 http.server 暴露 JSON API,前端为单页棋盘。

API:
  POST /api/new      {channels?, blocks?}            -> 新开一局,返回棋盘
  POST /api/legal    {from:[r,c]}                     -> 返回该子的合法落点
  POST /api/move     {from:[r,c], to:[r,c]}           -> 人类走子,返回新局面
  POST /api/ai                                       -> AI 走一步,返回其走法与局面
  GET  /                                               -> 返回棋盘页面

会话为单局全局状态(单用户本地对弈场景,简单够用)。模型可选:有 checkpoint
则 AI 用 MCTS+网络,否则回退到随机合法走法(便于无模型时先试 UI)。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from .game import GameState
from .constants import RED, BLACK, piece_char, NUM_ROWS, NUM_COLS
from .evaluator import Evaluator, pick_device
from .play import load_model, AIPlayer


class _RandomAI:
    """无模型时的回退 AI:随机选合法走法。"""

    def select_move(self, state: GameState):
        return random.choice(state.legal_moves())

    def advance(self, move):
        pass  # 随机 AI 无搜索树,无需复用


class GameServerState:
    """服务端持有的单局对局状态与 AI。"""

    def __init__(self, ai, ai_side):
        self.game = GameState()
        self.ai = ai
        self.ai_side = ai_side

    def board_payload(self):
        """把棋盘序列化为前端可渲染的结构。"""
        grid = self.game.board.grid
        cells = [[grid[r][c] for c in range(NUM_COLS)] for r in range(NUM_ROWS)]
        result = self.game.result()
        return {
            "grid": cells,
            "to_move": self.game.to_move,
            "result": result,
            "in_check": _in_check(self.game),
        }


def _in_check(game: GameState) -> bool:
    from . import movegen
    return movegen.in_check(game.board, game.to_move)


# 模块级单例状态,由 handler 共享。
_STATE: GameServerState | None = None
_AI_FACTORY = None  # 可调用,返回 (ai, ai_side)


def _reset_game():
    global _STATE
    ai, ai_side = _AI_FACTORY()
    _STATE = GameServerState(ai, ai_side)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def log_message(self, *args):
        pass  # 静默,避免刷屏

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send_html(_PAGE_HTML)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        global _STATE
        if self.path == "/api/new":
            _reset_game()
            self._send_json(_STATE.board_payload())
            return

        if _STATE is None:
            _reset_game()

        if self.path == "/api/legal":
            data = self._read_json()
            frm = tuple(data.get("from", []))
            legal = _STATE.game.legal_moves()
            tos = [list(to) for (f, to) in legal if f == frm]
            self._send_json({"targets": tos})
            return

        if self.path == "/api/move":
            data = self._read_json()
            frm = tuple(data.get("from", []))
            to = tuple(data.get("to", []))
            legal = set(map(tuple, _STATE.game.legal_moves()))
            if (frm, to) not in legal:
                self._send_json({"error": "illegal move"}, status=400)
                return
            _STATE.game.push((frm, to))
            _STATE.ai.advance((frm, to))  # 同步 AI 搜索树到当前局面
            self._send_json(_STATE.board_payload())
            return

        if self.path == "/api/ai":
            if _STATE.game.is_terminal():
                self._send_json(_STATE.board_payload())
                return
            move = _STATE.ai.select_move(_STATE.game)
            _STATE.game.push(move)
            _STATE.ai.advance(move)  # 同步 AI 搜索树到当前局面
            payload = _STATE.board_payload()
            payload["ai_move"] = [list(move[0]), list(move[1])]
            self._send_json(payload)
            return

        self._send_json({"error": "not found"}, status=404)


def _make_ai_factory(checkpoint, simulations, device, ai_side):
    """返回一个工厂函数,每次新局调用以构造 AI。"""
    if checkpoint and os.path.exists(checkpoint):
        net = load_model(checkpoint, device)
        evaluator = Evaluator(net, device=device)

        def factory():
            return AIPlayer(evaluator, simulations=simulations), ai_side
        print(f"[AI] 使用模型 {checkpoint},模拟次数 {simulations}")
    else:
        def factory():
            return _RandomAI(), ai_side
        print("[AI] 未提供模型,使用随机走子(可先体验 UI)")
    return factory


def parse_args():
    p = argparse.ArgumentParser(description="中国象棋 Web 对弈服务")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ai-side", choices=["red", "black"], default="black")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    global _AI_FACTORY
    args = parse_args()
    device = pick_device(args.device)
    ai_side = RED if args.ai_side == "red" else BLACK
    _AI_FACTORY = _make_ai_factory(args.checkpoint, args.simulations, device, ai_side)
    _reset_game()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[服务] http://{args.host}:{args.port}  (Ctrl-C 退出)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出]")
        server.shutdown()


# 页面 HTML 在 webui_page.py 中定义,避免单文件过长。
from .webui_page import PAGE_HTML as _PAGE_HTML  # noqa: E402


if __name__ == "__main__":
    main()
