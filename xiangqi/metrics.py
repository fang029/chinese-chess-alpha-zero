"""训练指标记录:把每轮 loss、胜负、耗时落盘为 CSV,便于事后画曲线。

零依赖(标准库 csv),可选 TensorBoard:若安装了 torch.utils.tensorboard
且传入 tensorboard=True,则同时写 event 文件。CSV 始终写,作为最可靠的记录。

每轮调用 log(iteration, metrics_dict)。首次写入按 metrics 的键定表头;
后续轮次复用同一组键(新增键会被忽略并告警,避免错列)。
"""

from __future__ import annotations

import csv
import os


class MetricsLogger:
    def __init__(self, log_dir: str, filename: str = "metrics.csv",
                 tensorboard: bool = False):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, filename)
        self._fieldnames = None
        self._file = None
        self._writer = None
        # 续训时若已存在 CSV,沿用其表头并追加,不覆盖历史。
        if os.path.exists(self.csv_path):
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header:
                self._fieldnames = header
                self._file = open(self.csv_path, "a", newline="")
                self._writer = csv.DictWriter(self._file, fieldnames=header)

        self._tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("[指标] 未安装 tensorboard,仅写 CSV")

    def log(self, iteration: int, metrics: dict):
        """记录一轮指标。metrics 的值应为标量(int/float)。"""
        row = {"iteration": iteration}
        row.update(metrics)

        if self._writer is None:
            # 首轮:按当前键建表头并写入。
            self._fieldnames = list(row.keys())
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        # 只写已知列,缺失补空,多余告警一次。
        filtered = {k: row.get(k, "") for k in self._fieldnames}
        extra = set(row) - set(self._fieldnames)
        if extra:
            print(f"[指标] 忽略未在表头中的键: {sorted(extra)}")
        self._writer.writerow(filtered)
        self._file.flush()

        if self._tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, iteration)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
