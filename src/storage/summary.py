import json
import os
from datetime import datetime

SUMMARY_PATH = "/data/records/summary.jsonl"


def append_summary(label, prob_normal, valid_segs, total_segs):
    """
    每次块推理完成后调用，追加一行到 records/summary.jsonl。

    参数:
        label       - "Normal" / "Abnormal" / "noise"（全部低质量时）
        prob_normal - SQA 加权平均的 prob_normal，信号差时传 None
        valid_segs  - 通过 SQA 的窗口数
        total_segs  - 本块总窗口数
    """
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

    record = {
        "ts":          datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "label":       label if label is not None else "noise",
        "prob_normal": round(prob_normal, 4) if prob_normal is not None else None,
        "valid_segs":  valid_segs,
        "total_segs":  total_segs,
    }

    with open(SUMMARY_PATH, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
