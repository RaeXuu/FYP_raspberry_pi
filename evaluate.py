"""
量化模型准确率评估（Pi 端）
对测试集每条音频跑完整推理流水线，对比 FP32 vs INT8 准确率。

诊断模型评估：
    python evaluate.py --mode diag

SQA 模型评估：
    python evaluate.py --mode sqa

两者都跑：
    python evaluate.py --mode both
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import yaml
import ai_edge_litert.interpreter as tflite
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.load_wav import load_wav
from src.preprocess.filters import apply_bandpass
from src.preprocess.segment import segment_audio
from src.preprocess.mel import logmel_fixed_size

# ──────────────────────────────────────────
# 模型路径
# ──────────────────────────────────────────
DIAG_INT8  = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
DIAG_FP32  = os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite")
SQA_INT8   = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
SQA_FP32   = os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite")

# 测试集 / metadata
DIAG_SPLIT    = os.path.join(PROJECT_ROOT, "data", "test_split.csv")
SQA_SPLIT     = os.path.join(PROJECT_ROOT, "data", "test_split_sqa.csv")
DIAG_META     = os.path.join(PROJECT_ROOT, "data", "metadata_physionet.csv")
SQA_META      = os.path.join(PROJECT_ROOT, "data", "metadata_quality.csv")

CONFIG        = os.path.join(PROJECT_ROOT, "config.yaml")

SQA_THRESHOLD  = 0.6   # 仅诊断模式的 SQA 门槛（连续加权时不用）
DIAG_THRESHOLD = 0.5


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def load_interp(model_path):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    return interp, in_idx, out_idx


def build_lookup(meta_path, split_path):
    """从 metadata CSV 建立 fname → (filepath, label) 字典，再按 split CSV 过滤。"""
    meta = {}
    with open(meta_path, newline="") as f:
        for row in csv.DictReader(f):
            meta[row["fname"]] = (row["filepath"], int(row["label"]))

    fnames = set()
    with open(split_path, newline="") as f:
        for row in csv.DictReader(f):
            fnames.add(row["fname"])

    rows = []
    for fname in sorted(fnames):
        if fname in meta:
            fp, label = meta[fname]
            rows.append({"fname": fname, "filepath": fp, "label": label})
    return rows


def predict_diag(filepath, mel_cfg, sqa_interp, sqa_in, sqa_out,
                 diag_interp, diag_in, diag_out):
    """诊断模型推理：SQA 加权平均 → pred label。"""
    t0 = time.perf_counter()
    y, sr = load_wav(filepath, target_sr=2000)
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)
    segments = segment_audio(y, sr)

    valid_results = []
    for seg in segments:
        mx = np.max(np.abs(seg))
        if mx > 0:
            seg = seg / mx
        mel    = logmel_fixed_size(y=seg, sr=sr, mel_cfg=mel_cfg,
                                   target_shape=(mel_cfg["n_mels"], 64))
        tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

        sqa_interp.set_tensor(sqa_in, tensor)
        sqa_interp.invoke()
        sqa_score = float(softmax(sqa_interp.get_tensor(sqa_out)[0])[1])
        if sqa_score < SQA_THRESHOLD:
            continue

        diag_interp.set_tensor(diag_in, tensor)
        diag_interp.invoke()
        prob_normal = float(softmax(diag_interp.get_tensor(diag_out)[0])[0])
        valid_results.append((sqa_score, prob_normal))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not valid_results:
        return None, None, 0, len(segments), elapsed_ms

    weights  = [r[0] for r in valid_results]
    probs    = [r[1] for r in valid_results]
    avg_prob = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
    pred     = 0 if avg_prob > DIAG_THRESHOLD else 1
    return pred, avg_prob, len(valid_results), len(segments), elapsed_ms


def predict_sqa(filepath, mel_cfg, sqa_interp, sqa_in, sqa_out):
    """SQA 模型推理：多窗口投票 → pred label (1=Bad/positive, 0=Good)。"""
    t0 = time.perf_counter()
    y, sr = load_wav(filepath, target_sr=2000)
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)
    segments = segment_audio(y, sr)

    prob_bad_list = []
    for seg in segments:
        mx = np.max(np.abs(seg))
        if mx > 0:
            seg = seg / mx
        mel    = logmel_fixed_size(y=seg, sr=sr, mel_cfg=mel_cfg,
                                   target_shape=(mel_cfg["n_mels"], 64))
        tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

        sqa_interp.set_tensor(sqa_in, tensor)
        sqa_interp.invoke()
        # label 约定（reversed）：Bad=1（正类），Good=0
        prob_bad = float(softmax(sqa_interp.get_tensor(sqa_out)[0])[1])
        prob_bad_list.append(prob_bad)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not prob_bad_list:
        return None, None, elapsed_ms

    avg_prob_bad = float(np.mean(prob_bad_list))
    pred = 1 if avg_prob_bad > 0.5 else 0
    return pred, avg_prob_bad, elapsed_ms


def compute_metrics(tp, tn, fp, fn):
    evaluated = tp + tn + fp + fn
    if evaluated == 0:
        return {}
    acc  = (tp + tn) / evaluated
    se   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * se / (prec + se)) if (prec + se) > 0 else 0.0
    return {"acc": acc, "se": se, "sp": sp, "f1": f1,
            "mscore": (se + sp) / 2, "evaluated": evaluated,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def run_diag_eval(rows, mel_cfg, sqa_path, diag_path, label):
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}  "
          f"DIAG={os.path.basename(diag_path)}")
    sqa_interp,  sqa_in,  sqa_out  = load_interp(sqa_path)
    diag_interp, diag_in, diag_out = load_interp(diag_path)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        fp_file  = row["filepath"]
        gt_label = row["label"]   # 1=Abnormal, 0=Normal  (physionet convention)

        if not os.path.exists(fp_file):
            skipped += 1
            continue

        pred, _, valid_segs, total_segs, elapsed_ms = predict_diag(
            fp_file, mel_cfg, sqa_interp, sqa_in, sqa_out,
            diag_interp, diag_in, diag_out)
        elapsed_all.append(elapsed_ms)

        if pred is None:
            skipped += 1
            continue

        if gt_label == 1 and pred == 1:   tp += 1
        elif gt_label == 0 and pred == 0: tn += 1
        elif gt_label == 0 and pred == 1: fp += 1
        else:                             fn += 1

    m = compute_metrics(tp, tn, fp, fn)
    if not m:
        print("    无可评估样本")
        return None

    arr = np.array(elapsed_all) if elapsed_all else np.array([0])
    print(f"    Accuracy={m['acc']*100:.1f}%  M-Score={m['mscore']*100:.1f}%  "
          f"Se={m['se']*100:.1f}%  Sp={m['sp']*100:.1f}%  "
          f"(evaluated={m['evaluated']}, skipped={skipped})")
    print(f"    推理耗时 mean={arr.mean():.0f}ms  "
          f"min={arr.min():.0f}ms  max={arr.max():.0f}ms")
    return m


def run_sqa_eval(rows, mel_cfg, sqa_path, label):
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}")
    sqa_interp, sqa_in, sqa_out = load_interp(sqa_path)

    tp = tn = fp = fn = skipped = 0

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        fp_file  = row["filepath"]
        gt_label = row["label"]   # metadata_quality: 1=Good, 0=Bad
        # SQA 正类 = Bad(0 in metadata) → 反转：gt_sqa=1 if Bad
        gt_sqa = 1 if gt_label == 0 else 0

        if not os.path.exists(fp_file):
            skipped += 1
            continue

        pred, _, elapsed_ms = predict_sqa(fp_file, mel_cfg, sqa_interp, sqa_in, sqa_out)

        if pred is None:
            skipped += 1
            continue

        if gt_sqa == 1 and pred == 1:   tp += 1
        elif gt_sqa == 0 and pred == 0: tn += 1
        elif gt_sqa == 0 and pred == 1: fp += 1
        else:                           fn += 1

    m = compute_metrics(tp, tn, fp, fn)
    if not m:
        print("    无可评估样本")
        return None

    print(f"    Accuracy={m['acc']*100:.1f}%  M-Score={m['mscore']*100:.1f}%  "
          f"Se(Bad)={m['se']*100:.1f}%  Sp(Good)={m['sp']*100:.1f}%  "
          f"(evaluated={m['evaluated']}, skipped={skipped})")
    return m


def print_comparison(fp32_m, int8_m, mode="diag"):
    if fp32_m is None or int8_m is None:
        return
    print(f"\n{'='*60}")
    if mode == "diag":
        print("FP32 vs INT8 对比（诊断模型，Table 5.5 / Table 6.3）")
        print(f"{'='*60}")
        print(f"  {'Metric':<14} {'FP32':>10} {'INT8':>10} {'Change':>10}")
        print(f"  {'-'*44}")
        for key, label in [("mscore","M-Score"), ("se","Sensitivity"),
                            ("sp","Specificity"), ("acc","Accuracy")]:
            v32, v8 = fp32_m[key]*100, int8_m[key]*100
            print(f"  {label:<14} {v32:>9.1f}% {v8:>9.1f}% {v8-v32:>+9.1f}%")
    else:
        print("FP32 vs INT8 对比（SQA 模型）")
        print(f"{'='*60}")
        print(f"  {'Metric':<14} {'FP32':>10} {'INT8':>10} {'Change':>10}")
        print(f"  {'-'*44}")
        for key, label in [("mscore","M-Score"), ("se","Se(Bad)"),
                            ("sp","Sp(Good)"), ("acc","Accuracy")]:
            v32, v8 = fp32_m[key]*100, int8_m[key]*100
            print(f"  {label:<14} {v32:>9.1f}% {v8:>9.1f}% {v8-v32:>+9.1f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diag", "sqa", "both"], default="both",
                        help="评估模式：diag / sqa / both（默认 both）")
    args = parser.parse_args()

    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    if args.mode in ("diag", "both"):
        print(f"\n{'='*60}")
        print(f"诊断模型评估（test_split.csv，{os.path.basename(DIAG_SPLIT)}）")
        rows_diag = build_lookup(DIAG_META, DIAG_SPLIT)
        print(f"  测试录音数：{len(rows_diag)}")
        print(f"{'='*60}")
        m_diag_fp32 = run_diag_eval(rows_diag, mel_cfg, SQA_FP32,  DIAG_FP32,  "FP32")
        m_diag_int8 = run_diag_eval(rows_diag, mel_cfg, SQA_INT8,  DIAG_INT8,  "INT8")
        print_comparison(m_diag_fp32, m_diag_int8, mode="diag")

    if args.mode in ("sqa", "both"):
        print(f"\n{'='*60}")
        print(f"SQA 模型评估（test_split_sqa.csv，{os.path.basename(SQA_SPLIT)}）")
        rows_sqa = build_lookup(SQA_META, SQA_SPLIT)
        print(f"  测试录音数：{len(rows_sqa)}")
        print(f"{'='*60}")
        m_sqa_fp32 = run_sqa_eval(rows_sqa, mel_cfg, SQA_FP32, "FP32")
        m_sqa_int8 = run_sqa_eval(rows_sqa, mel_cfg, SQA_INT8, "INT8")
        print_comparison(m_sqa_fp32, m_sqa_int8, mode="sqa")


if __name__ == "__main__":
    main()
