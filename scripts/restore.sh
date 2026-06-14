#!/usr/bin/env bash
# 把本地保存的 checkpoints 传回服务器,用于在新实例上继续训练。
# 在【本地 Mac】执行:  bash scripts/restore.sh
#
# 适用场景:之前 fetch.sh 拉回了成果、释放了旧实例;现在开了新实例、跑过
# deploy.sh(它会排除 checkpoints),需要把训练进度传回去再 RESUME 续训。
set -e

REMOTE="${REMOTE:-myserver}"
REMOTE_DIR="${REMOTE_DIR:-~/board}/checkpoints"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)/checkpoints}"

if [ ! -f "$LOCAL_DIR/latest.pt" ]; then
  echo "本地没有 $LOCAL_DIR/latest.pt,无可恢复的进度。"
  exit 1
fi

echo "==> 把本地 checkpoints 传回 $REMOTE:$REMOTE_DIR"
ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
scp "$LOCAL_DIR/latest.pt"   "$REMOTE:$REMOTE_DIR/"
scp "$LOCAL_DIR/buffer.npz"  "$REMOTE:$REMOTE_DIR/" 2>/dev/null || echo "  (本地无 buffer.npz,续训会重新暖机)"
scp "$LOCAL_DIR/metrics.csv" "$REMOTE:$REMOTE_DIR/" 2>/dev/null || true

echo "==> 完成。在服务器续训:"
echo "  ssh $REMOTE"
echo "  RESUME=1 screen -dmS train bash $REMOTE_DIR/scripts/train.sh"
