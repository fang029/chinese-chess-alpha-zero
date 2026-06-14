#!/usr/bin/env bash
# 启动正式训练。在【服务器】上执行。
# 推荐用 screen 后台跑,断开 SSH 也不停:
#   screen -dmS train bash scripts/train.sh        # 全新训练
#   RESUME=1 screen -dmS train bash scripts/train.sh  # 断点续训
#   screen -r train      # 回来看进度;Ctrl+A 再按 D 挂起
#
# 示例配置按单卡 GPU(如 A10 24G)+ 约 32 核 CPU 调好,按需修改参数。
set -e

cd "$(dirname "$0")/.."
# 用 PY 环境变量指定 Python(默认 python);加 -u 关闭 stdout 缓冲,经 tee 实时见日志。
PY="${PY:-python} -u"
# 限制 BLAS/OMP 线程,避免多进程超额订阅 CPU(worker 内已 set_num_threads(1))。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RESUME_ARG=""
if [ "${RESUME:-0}" = "1" ] && [ -f checkpoints/latest.pt ]; then
  RESUME_ARG="--resume checkpoints/latest.pt"
  echo "==> 断点续训:接 checkpoints/latest.pt"
fi

# 32 核单卡:满规模 128×10 网络在 CPU 上自我对弈太慢(单局 10 分钟级),
# 反馈周期过长。改用较小网络 64×6 + 300 模拟,推理快 3-4 倍。
# 冷启动期随机网络几乎全和棋、每局都耗到 max-moves,故把 max-moves 降到 120
# 避免长局拖慢;min-buffer 调低让训练尽早开始、尽快脱离随机乱走。
# 28 worker 并行铺开对局(每 worker 单线程),GPU 专心训练。
# 注:并行管线不含评估门控(门控只在串行 pipeline 里)。
$PY -m xiangqi.pipeline_parallel \
    --iterations 1000 --games-per-iter 24 --workers 28 \
    --simulations 200 --channels 64 --blocks 6 --batch-size 512 \
    --train-device cuda --worker-device cpu --max-moves 120 \
    --queue-size 128 --min-buffer 1000 --keep-checkpoints 5 \
    $RESUME_ARG \
    2>&1 | tee -a train.log
