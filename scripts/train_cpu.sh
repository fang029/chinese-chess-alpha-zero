#!/usr/bin/env bash
# 纯 CPU 自我对弈训练(多核机器)。从预训练模型接续。
# 在【服务器】上执行,推荐 screen 后台:
#   screen -dmS train bash scripts/train_cpu.sh        # 从 pretrained.pt 起
#   RESUME=1 screen -dmS train bash scripts/train_cpu.sh  # 接 latest.pt 续训
#   screen -r train      # 回来看;Ctrl+A 再按 D 挂起
#
# 目标:能下、不乱走、基本棋力。默认网络 64×6(纯 CPU 快);无需 GPU。
# 示例按多核(数百核)纯 CPU 机器调好;worker 数按机器核数调整。
# 可用环境变量覆盖:PY、WORKERS、CHANNELS、BLOCKS、SIMS、CKPT。
set -e

cd "$(dirname "$0")/.."
PY="${PY:-python}"   # 可用 PY=/path/to/python 覆盖
# 限制 BLAS/OMP 线程,避免多进程超额订阅 CPU(worker 内已 set_num_threads(1))。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

WORKERS="${WORKERS:-$(( $(nproc 2>/dev/null || echo 16) - 8 ))}"  # 留 8 核给主进程/系统
CHANNELS="${CHANNELS:-64}"
BLOCKS="${BLOCKS:-6}"
SIMS="${SIMS:-400}"
# 起点权重:RESUME=1 接 latest.pt 续训;否则从指定预训练模型起。
PRE="${CKPT:-checkpoints/pretrained.pt}"

if [ "${RESUME:-0}" = "1" ] && [ -f checkpoints/latest.pt ]; then
  RESUME_ARG="--resume checkpoints/latest.pt"
  echo "==> 断点续训:接 checkpoints/latest.pt"
elif [ -f "$PRE" ]; then
  RESUME_ARG="--resume $PRE"
  echo "==> 从预训练模型起步:$PRE"
else
  RESUME_ARG=""
  echo "==> 警告:无预训练模型,将从随机权重开始(冷启动会停滞)"
fi

# 256 核纯 CPU:大量 worker 并行自弈,训练也走 CPU。
$PY -m xiangqi.pipeline_parallel \
    --iterations 2000 --games-per-iter 200 --workers "$WORKERS" \
    --simulations "$SIMS" --channels "$CHANNELS" --blocks "$BLOCKS" \
    --batch-size 1024 \
    --train-device cpu --worker-device cpu --max-moves 200 \
    --queue-size 512 --min-buffer 2000 --keep-checkpoints 5 \
    $RESUME_ARG \
    2>&1 | tee -a train.log
