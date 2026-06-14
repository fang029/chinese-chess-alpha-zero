"""训练指标记录(MetricsLogger)测试:CSV 写入、续训追加、缺列处理。"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xiangqi.metrics import MetricsLogger


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_writes_header_and_rows(tmp_path):
    logger = MetricsLogger(str(tmp_path))
    logger.log(1, {"loss_total": 7.5, "red_win": 2})
    logger.log(2, {"loss_total": 7.1, "red_win": 3})
    logger.close()

    rows = _read_csv(os.path.join(str(tmp_path), "metrics.csv"))
    assert len(rows) == 2
    assert rows[0]["iteration"] == "1"
    assert rows[0]["loss_total"] == "7.5"
    assert rows[1]["red_win"] == "3"


def test_resume_appends_not_overwrites(tmp_path):
    logger = MetricsLogger(str(tmp_path))
    logger.log(1, {"loss_total": 7.5, "red_win": 2})
    logger.close()

    # 模拟续训:新建 logger 指向同目录,应追加而非覆盖。
    logger2 = MetricsLogger(str(tmp_path))
    logger2.log(2, {"loss_total": 7.0, "red_win": 4})
    logger2.close()

    rows = _read_csv(os.path.join(str(tmp_path), "metrics.csv"))
    assert len(rows) == 2
    assert rows[0]["iteration"] == "1"
    assert rows[1]["iteration"] == "2"


def test_empty_loss_values_ok(tmp_path):
    """暖机轮 loss 为空字符串,应正常写入。"""
    logger = MetricsLogger(str(tmp_path))
    logger.log(1, {"loss_total": "", "buffer": 100})
    logger.close()
    rows = _read_csv(os.path.join(str(tmp_path), "metrics.csv"))
    assert rows[0]["loss_total"] == ""
    assert rows[0]["buffer"] == "100"


def test_context_manager(tmp_path):
    with MetricsLogger(str(tmp_path)) as logger:
        logger.log(1, {"x": 1})
    assert os.path.exists(os.path.join(str(tmp_path), "metrics.csv"))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
