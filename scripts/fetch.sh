#!/usr/bin/env bash
# 把服务器上的训练成果拉回本地。在【本地 Mac】执行:
#   bash scripts/fetch.sh
#
# 释放云服务器前务必先跑这个,否则 checkpoints 只在服务器上,实例一释放就没了。
set -e

REMOTE="${REMOTE:-myserver}"
REMOTE_DIR="${REMOTE_DIR:-~/board}/checkpoints"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"

mkdir -p "$LOCAL_DIR"

echo "==> 从 $REMOTE 拉取训练成果到 $LOCAL_DIR"
# 关键文件(续训和对弈必需):
#   latest.pt   最新模型 + 优化器状态(续训用)
#   buffer.npz  replay buffer(续训免暖机)
#   metrics.csv 训练曲线
# 以及所有 iter_*.pt 历史存档、champion.pt(若开了门控)。
scp "$REMOTE:$REMOTE_DIR/latest.pt"   "$LOCAL_DIR/" 2>/dev/null || echo "  (无 latest.pt)"
scp "$REMOTE:$REMOTE_DIR/buffer.npz"  "$LOCAL_DIR/" 2>/dev/null || echo "  (无 buffer.npz)"
scp "$REMOTE:$REMOTE_DIR/metrics.csv" "$LOCAL_DIR/" 2>/dev/null || echo "  (无 metrics.csv)"
scp "$REMOTE:$REMOTE_DIR/champion.pt" "$LOCAL_DIR/" 2>/dev/null || true
# iter_*.pt 可能多个,用通配批量拉(没有则忽略)。
scp "$REMOTE:$REMOTE_DIR/iter_*.pt"   "$LOCAL_DIR/" 2>/dev/null || true

echo "==> 完成。本地 checkpoints:"
ls -lh "$LOCAL_DIR"
echo ""
echo "现在可以安全释放云服务器。下次续训:把 checkpoints 传回服务器后"
echo "  RESUME=1 screen -dmS train bash $REMOTE_DIR/scripts/train.sh"
