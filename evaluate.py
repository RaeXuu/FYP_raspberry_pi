"""
量化模型准确率评估（Pi 端）
对测试集每条音频跑完整推理流水线，输出准确率、灵敏度、特异度、F1。

标签约定（与训练一致）：
    0 = Normal
    1 = Abnormal

运行：
    python evaluate.py --csv data/metadata_physionet.csv
    python evaluate.py --csv data/test_split.csv          # 推荐：只传测试集
    python evaluate.py --csv data/test_split.csv --verbose # 逐条打印结果
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml
import ai_edge_litert.interpreter as tflite

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size

# ──────────────────────────────────────────
# 配置（与 main_pi.py 保持一致）
# ──────────────────────────────────────────
SQA_THRESHOLD  = 0.6
DIAG_THRESHOLD = 0.5   # prob_normal > 此值 → Normal

SQA_MODEL  = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
DIAG_MODEL = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
CONFIG     = os.path.join(PROJECT_ROOT, "config.yaml")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_file(filepath, mel_cfg, q_interp, d_interp,
                 q_in, q_out, d_in, d_out):
    """
    对单条音频文件跑完整推理，返回 (pred_label, avg_prob_normal, valid_segs, total_segs, elapsed_ms)
    pred_label: 1=Normal, 0=Abnormal, None=全部低质量
    """
    t0 = time.perf_counter()

    y, sr = load_wav(filepath, target_sr=2000)
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)
    segments = segment_audio(y, sr)

    valid_results = []   # [(sqa_score, prob_normal), ...]

    for seg in segments:
        max_val = np.max(np.abs(seg))
        if max_val > 0:
            seg = seg / max_val

        mel    = logmel_fixed_size(y=seg, sr=sr, mel_cfg=mel_cfg,
                                   target_shape=(mel_cfg["n_mels"], 64))
        tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

        # SQA
        q_interp.set_tensor(q_in, tensor)
        q_interp.invoke()
        q_probs   = softmax(q_interp.get_tensor(q_out)[0])
        sqa_score = float(q_probs[1])

        if sqa_score < SQA_THRESHOLD:
            continue

        # 诊断
        d_interp.set_tensor(d_in, tensor)
        d_interp.invoke()
        d_probs     = softmax(d_interp.get_tensor(d_out)[0])
        prob_normal = float(d_probs[0])
        valid_results.append((sqa_score, prob_normal))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not valid_results:
        return None, None, 0, len(segments), elapsed_ms

    weights  = [r[0] for r in valid_results]
    probs    = [r[1] for r in valid_results]
    avg_prob = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
    pred     = 0 if avg_prob > DIAG_THRESHOLD else 1

    return pred, avg_prob, len(valid_results), len(segments), elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="测试集 CSV，需含 filepath 和 label 列（1=Normal, 0=Abnormal）")
    parser.add_argument("--verbose", action="store_true",
                        help="逐条打印预测结果")
    args = parser.parse_args()

    # ── 加载配置和模型 ──
    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    q_interp = tflite.Interpreter(model_path=SQA_MODEL)
    d_interp = tflite.Interpreter(model_path=DIAG_MODEL)
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()
    q_in  = q_interp.get_input_details()[0]["index"]
    q_out = q_interp.get_output_details()[0]["index"]
    d_in  = d_interp.get_input_details()[0]["index"]
    d_out = d_interp.get_output_details()[0]["index"]

    # ── 读取测试集 ──
    import csv
    rows = []
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total     = len(rows)
    skipped   = 0   # 全部低质量，无法判断
    elapsed_all = []

    # 混淆矩阵计数
    tp = tn = fp = fn = 0   # tp/tn/fp/fn 以 Abnormal(0) 为正类

    print(f"\n{'='*60}")
    print(f"测试集：{args.csv}  （{total} 条）")
    print(f"{'='*60}")

    for i, row in enumerate(rows):
        filepath = row["filepath"]
        gt_label = int(row["label"])   # 0=Normal, 1=Abnormal

        if not os.path.exists(filepath):
            print(f"  [{i+1:04d}] 文件不存在，跳过：{filepath}")
            skipped += 1
            continue

        pred, avg_prob, valid_segs, total_segs, elapsed_ms = predict_file(
            filepath, mel_cfg, q_interp, d_interp, q_in, q_out, d_in, d_out)

        elapsed_all.append(elapsed_ms)

        if pred is None:
            skipped += 1
            if args.verbose:
                print(f"  [{i+1:04d}] {os.path.basename(filepath):<20} "
                      f"GT={'Normal' if gt_label==1 else 'Abnormal':<8} "
                      f"Pred=SKIP (0/{total_segs} segs passed SQA)")
            continue

        pred_str = "Normal" if pred == 0 else "Abnormal"
        gt_str   = "Normal" if gt_label == 0 else "Abnormal"
        correct  = "✓" if pred == gt_label else "✗"

        # 以 Abnormal(1) 为正类统计混淆矩阵
        if gt_label == 1 and pred == 1:
            tp += 1
        elif gt_label == 0 and pred == 0:
            tn += 1
        elif gt_label == 0 and pred == 1:
            fp += 1
        else:
            fn += 1

        if args.verbose:
            print(f"  [{i+1:04d}] {os.path.basename(filepath):<20} "
                  f"GT={gt_str:<8} Pred={pred_str:<8} "
                  f"prob_N={avg_prob:.3f}  "
                  f"valid={valid_segs}/{total_segs}  {correct}")

    # ── 计算指标 ──
    evaluated = tp + tn + fp + fn
    if evaluated == 0:
        print("没有可评估的样本。")
        return

    accuracy    = (tp + tn) / evaluated
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Abnormal recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Normal recall
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)

    print(f"\n{'='*60}")
    print("评估结果")
    print(f"{'='*60}")
    print(f"  样本总数     {total}")
    print(f"  有效评估     {evaluated}  （跳过 {skipped} 条低质量/缺失）")
    print(f"")
    print(f"  Accuracy     {accuracy*100:.1f}%")
    print(f"  Sensitivity  {sensitivity*100:.1f}%   ← 异常检出率（Abnormal recall）")
    print(f"  Specificity  {specificity*100:.1f}%   ← 正常判对率（Normal recall）")
    print(f"  Precision    {precision*100:.1f}%")
    print(f"  F1 Score     {f1*100:.1f}%")
    print(f"")
    print(f"  混淆矩阵（正类 = Abnormal）")
    print(f"                   Pred Normal  Pred Abnormal")
    print(f"  True Normal       {tn:>8}       {fp:>8}")
    print(f"  True Abnormal     {fn:>8}       {tp:>8}")

    if elapsed_all:
        arr = np.array(elapsed_all)
        print(f"")
        print(f"  推理耗时（per file）")
        print(f"    mean={arr.mean():.0f}ms  "
              f"min={arr.min():.0f}ms  max={arr.max():.0f}ms")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
