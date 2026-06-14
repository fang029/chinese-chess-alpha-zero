"""Web 服务测试:用 TestClient 风格的真实 HTTP 请求验证 API 与页面。

不依赖浏览器;通过启动 ThreadingHTTPServer 在随机端口,用 urllib 请求。
"""

import os
import sys
import json
import threading
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from http.server import ThreadingHTTPServer

from xiangqi import webui
from xiangqi.webui_page import PAGE_HTML
from xiangqi.constants import RED, BLACK


def _post(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.read().decode()


def _start_server():
    # 用随机 AI(无 checkpoint),固定执黑。
    webui._AI_FACTORY = webui._make_ai_factory(None, 50, None, BLACK)
    webui._reset_game()
    server = ThreadingHTTPServer(("127.0.0.1", 0), webui.Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://127.0.0.1:{port}"


def test_page_served():
    server, base = _start_server()
    try:
        html = _get(base, "/")
        assert "<!DOCTYPE html>" in html
        assert "中国象棋" in html
        assert "canvas" in html
    finally:
        server.shutdown()


def test_page_html_balanced():
    """页面字符串拼接完整:script 标签成对,大括号平衡(廉价语法检查)。"""
    assert PAGE_HTML.count("<script>") == PAGE_HTML.count("</script>")
    assert PAGE_HTML.count("{") == PAGE_HTML.count("}")
    assert PAGE_HTML.strip().endswith("</html>")


def test_new_game_and_legal():
    server, base = _start_server()
    try:
        s = _post(base, "/api/new")
        assert s["to_move"] == RED
        assert s["result"] == "ongoing"
        assert len(s["grid"]) == 10 and len(s["grid"][0]) == 9
        # 红马合法落点
        legal = _post(base, "/api/legal", {"from": [0, 1]})
        targets = {tuple(t) for t in legal["targets"]}
        assert targets == {(2, 2), (2, 0)}
    finally:
        server.shutdown()


def test_move_and_ai_reply():
    server, base = _start_server()
    try:
        _post(base, "/api/new")
        s = _post(base, "/api/move", {"from": [2, 1], "to": [2, 4]})
        assert "error" not in s
        assert s["to_move"] == BLACK
        # AI 应招
        s2 = _post(base, "/api/ai")
        assert "ai_move" in s2
        assert s2["to_move"] == RED
    finally:
        server.shutdown()


def test_illegal_move_rejected():
    server, base = _start_server()
    try:
        _post(base, "/api/new")
        try:
            _post(base, "/api/move", {"from": [0, 0], "to": [5, 5]})
            assert False, "应抛 HTTP 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
