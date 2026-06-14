#!/usr/bin/env bash
# 一键部署:把本地项目上传到远程服务器,并远程建环境、装依赖、自检。
# 在【本地 Mac】执行:  bash scripts/deploy.sh
#
# 前提:已能用 `ssh $REMOTE` 免密登录(~/.ssh/config 配好 Host myserver)。
set -e

REMOTE="${REMOTE:-myserver}"        # SSH 主机别名,默认 myserver
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-~/board}"

echo "==> [1/3] 上传项目到 $REMOTE:$REMOTE_DIR"
# 用 tar over ssh:只依赖两端都有的 tar,避免服务器未装 rsync 的问题。
# 排除虚拟环境、缓存、checkpoints、git。
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
tar czf - -C "$LOCAL_DIR" \
    --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='checkpoints' --exclude='.git' \
    . | ssh "$REMOTE" "tar xzf - -C $REMOTE_DIR"

echo "==> [2/3] 远程执行环境引导脚本"
ssh "$REMOTE" "bash $REMOTE_DIR/scripts/server_setup.sh"

echo "==> [3/3] 完成"
echo "登录服务器开始训练: ssh $REMOTE"
echo "然后: conda activate xiangqi && bash $REMOTE_DIR/scripts/train.sh"
