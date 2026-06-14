#!/usr/bin/env bash
# 服务器端环境引导:建 conda 环境并装依赖。由 deploy.sh 上传并远程执行,
# 也可登录后手动:bash scripts/server_setup.sh
#
# 部分云服务器自带的 Python 过旧。若无 conda 则自动装 miniconda;建 Python 3.10
# 环境 xiangqi,装 CPU 版 PyTorch(纯 CPU 自弈机,无需 CUDA)。
set -e

ENV_NAME="xiangqi"
PY_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_DIR="$HOME/miniconda3"
TUNA="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> [1/4] 准备 conda"
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]; then
  source "$CONDA_DIR/etc/profile.d/conda.sh"
else
  echo "    未找到 conda,安装 miniconda 到 $CONDA_DIR"
  MC=/tmp/miniconda.sh
  curl -fsSL -o "$MC" \
    https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    || curl -fsSL -o "$MC" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$MC" -b -p "$CONDA_DIR"
  source "$CONDA_DIR/etc/profile.d/conda.sh"
fi

echo "==> [2/4] 创建/复用 conda 环境 $ENV_NAME (Python $PY_VERSION)"
# 新版 conda 默认 channel 需接受 ToS;用 conda-forge 避开,且接受 ToS 兜底。
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "    环境已存在,跳过创建"
else
  conda create -y -n "$ENV_NAME" -c conda-forge python="$PY_VERSION"
fi
conda activate "$ENV_NAME"

echo "==> [3/4] 安装依赖(CPU 版 torch + numpy + pytest)"
pip install --upgrade pip -i "$TUNA"
# 纯 CPU 机:装 CPU 版 torch(从官方 CPU 索引,体积小、无 CUDA)。
pip install torch --index-url https://download.pytorch.org/whl/cpu \
  || pip install torch -i "$TUNA"
pip install numpy pytest -i "$TUNA"

echo "==> [4/4] 自检"
cd "$PROJECT_DIR"
PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
[ -x "$PY" ] || PY="$(conda info --base)/envs/$ENV_NAME/bin/python"
"$PY" --version
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda可用:', torch.cuda.is_available())"
"$PY" -m pytest tests/ -q

echo ""
echo "============================================"
echo " 环境就绪。env python 绝对路径:"
echo "   $PY"
echo " 启动自弈见 scripts/train_cpu.sh。"
echo "============================================"
