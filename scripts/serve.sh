#!/usr/bin/env bash
# 启动浏览器对弈界面。在【服务器】上执行(先激活好 Python 环境)。
# 启动后在本地浏览器访问 http://<SERVER_IP>:8000
# 前提:云端安全组已放行 TCP 8000 端口。
#
#   bash scripts/serve.sh                              # 用 latest.pt
#   CKPT=checkpoints/champion.pt bash scripts/serve.sh # 指定其他模型
set -e

cd "$(dirname "$0")/.."
PY="${PY:-python}"   # 可用 PY=/path/to/python 覆盖

CKPT="${CKPT:-checkpoints/latest.pt}"
SIMS="${SIMS:-400}"

if [ ! -f "$CKPT" ]; then
  echo "找不到模型 $CKPT;先训练或用 CKPT=路径 指定。无模型也可跑(随机走子)。"
  $PY -m xiangqi.webui --host 0.0.0.0 --port 8000
else
  echo "==> 用模型 $CKPT 启动,访问 http://<SERVER_IP>:8000"
  $PY -m xiangqi.webui --checkpoint "$CKPT" --simulations "$SIMS" \
    --host 0.0.0.0 --port 8000
fi
