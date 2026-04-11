"""
量化模型准确率评估（Pi 端）
对测试集每条音频跑完整推理流水线，对比 FP32 vs INT8 准确率。

模型输出索引约定
  SQA 模型（heart_quality_*.tflite）：
    训练时 label 0 = Good, label 1 = Bad（reversed convention）
    → index 0 = Good 概率, index 1 = Bad 概率

  诊断模型（heart_model_*.tflite）：
    训练时 label 0 = Normal, label 1 = Abnormal
    → index 0 = Normal 概率, index 1 = Abnormal 概率

使用方式
  python evaluate.py --mode sqa          # SQA 独立评估（test_split_sqa.csv）
  python evaluate.py --mode diag         # 诊断模型（无 SQA 门控，test_split.csv）
  python evaluate.py --mode both         # 耦合流水线（SQA 门控 + 加权，test_split.csv）
  python evaluate.py --mode sqa --verify # 额外打印若干样本输出，排查索引
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
# 路径 / 常量
# ──────────────────────────────────────────
DIAG_FP32 = os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite")
DIAG_INT8 = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
SQA_FP32  = os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite")
SQA_INT8  = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")

DIAG_SPLIT = os.path.join(PROJECT_ROOT, "data", "test_split.csv")
SQA_SPLIT  = os.path.join(PROJECT_ROOT, "data", "test_split_sqa.csv")
DIAG_META  = os.path.join(PROJECT_ROOT, "data", "metadata_physionet.csv")
SQA_META   = os.path.join(PROJECT_ROOT, "data", "metadata_quality.csv")

CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")

# SQA 门控阈值：Good 概率低于此值的窗口在耦合模式中被过滤
SQA_THRESHOLD  = 0.5
DIAG_THRESHOLD = 0.5

# 与 main_pi.py 对齐的分块参数
SAMPLE_RATE    = 2000
CHUNK_DURATION = 20
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_DURATION   # 40000 samples
SEG_SAMPLES    = int(SAMPLE_RATE * 2.0)          # 4000 samples（2s 滑窗）
HOP_SAMPLES    = int(SEG_SAMPLES * 0.5)          # 2000 samples（1s hop）

# SQA 模型输出索引（Bad=1 reversed convention）
SQA_IDX_GOOD = 0
SQA_IDX_BAD  = 1

# 诊断模型输出索引
DIAG_IDX_NORMAL   = 0
DIAG_IDX_ABNORMAL = 1


# ──────────────────────────────────────────
# 基础工具
# ──────────────────────────────────────────

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
    """metadata CSV → fname:(filepath, label)，按 split CSV 过滤并去重。"""
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
            filepath, label = meta[fname]
            rows.append({"fname": fname, "filepath": filepath, "label": label})
    return rows


def load_tensors(filepath, mel_cfg):
    """WAV → 带通 → 切片 → 归一化 → mel tensor 列表。"""
    y, sr = load_wav(filepath, target_sr=2000)
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)
    segments = segment_audio(y, sr)
    tensors = []
    for seg in segments:
        mx = np.max(np.abs(seg))
        if mx > 0:
            seg = seg / mx
        mel = logmel_fixed_size(y=seg, sr=sr, mel_cfg=mel_cfg,
                                target_shape=(mel_cfg["n_mels"], 64))
        tensors.append(mel[np.newaxis, np.newaxis, ...].astype(np.float32))
    return tensors


# ──────────────────────────────────────────
# 推理
# ──────────────────────────────────────────

def infer_sqa(tensors, interp, in_idx, out_idx):
    """返回 (avg_prob_bad, avg_prob_good)，输入为已预处理好的 tensor 列表。"""
    bad_probs = []
    for t in tensors:
        interp.set_tensor(in_idx, t)
        interp.invoke()
        sm = softmax(interp.get_tensor(out_idx)[0])
        bad_probs.append(sm[SQA_IDX_BAD])
    avg_bad = float(np.mean(bad_probs))
    return avg_bad, 1.0 - avg_bad


def infer_diag(tensors, interp, in_idx, out_idx):
    """返回 avg_prob_normal。"""
    normal_probs = []
    for t in tensors:
        interp.set_tensor(in_idx, t)
        interp.invoke()
        sm = softmax(interp.get_tensor(out_idx)[0])
        normal_probs.append(sm[DIAG_IDX_NORMAL])
    return float(np.mean(normal_probs))


def predict_sqa(filepath, mel_cfg, interp, in_idx, out_idx):
    """
    SQA 推理。
    返回 (pred, avg_prob_bad, elapsed_ms)
      pred=1 → Bad, pred=0 → Good
    """
    t0 = time.perf_counter()
    tensors = load_tensors(filepath, mel_cfg)
    if not tensors:
        return None, None, (time.perf_counter() - t0) * 1000

    avg_bad, _ = infer_sqa(tensors, interp, in_idx, out_idx)
    pred = 1 if avg_bad > 0.5 else 0
    return pred, avg_bad, (time.perf_counter() - t0) * 1000


def predict_diag_only(filepath, mel_cfg, interp, in_idx, out_idx):
    """
    诊断推理（无 SQA 门控）。
    返回 (pred, avg_prob_normal, n_segs, elapsed_ms)
      pred=1 → Abnormal, pred=0 → Normal
    """
    t0 = time.perf_counter()
    tensors = load_tensors(filepath, mel_cfg)
    if not tensors:
        return None, None, 0, (time.perf_counter() - t0) * 1000

    avg_normal = infer_diag(tensors, interp, in_idx, out_idx)
    pred = 0 if avg_normal > DIAG_THRESHOLD else 1
    return pred, avg_normal, len(tensors), (time.perf_counter() - t0) * 1000


def predict_diag_coupled(filepath, mel_cfg,
                         sqa_interp, sqa_in, sqa_out,
                         diag_interp, diag_in, diag_out):
    """
    诊断推理（SQA 门控 + 加权平均），与 main_pi.py run_inference 完全对齐：
    - 音频先切成 20s chunk
    - 每个 chunk 内做 2s / 50% overlap 滑窗
    - sqa_score = sm[1]，低于 SQA_THRESHOLD 的窗口跳过
    - 所有 chunk 的有效窗口汇总后加权平均得到文件级预测
    返回 (pred, avg_prob_normal, valid_wins, total_wins, elapsed_ms)
    """
    t0 = time.perf_counter()
    y, sr = load_wav(filepath, target_sr=SAMPLE_RATE)
    y = apply_bandpass(y, fs=sr, lowcut=25, highcut=400)

    valid      = []   # [(sqa_score, prob_normal), ...]
    total_wins = 0

    for chunk_start in range(0, len(y), CHUNK_SAMPLES):
        chunk = y[chunk_start : chunk_start + CHUNK_SAMPLES]
        if len(chunk) < SEG_SAMPLES:
            continue   # chunk 过短，跳过

        for win_start in range(0, len(chunk) - SEG_SAMPLES + 1, HOP_SAMPLES):
            window = chunk[win_start : win_start + SEG_SAMPLES]
            total_wins += 1

            mx = np.max(np.abs(window))
            if mx > 0:
                window = window / mx

            mel = logmel_fixed_size(y=window, sr=sr, mel_cfg=mel_cfg,
                                    target_shape=(mel_cfg["n_mels"], 64))
            t = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

            # SQA 门控
            sqa_interp.set_tensor(sqa_in, t)
            sqa_interp.invoke()
            sqa_score = float(softmax(sqa_interp.get_tensor(sqa_out)[0])[1])  # q_probs[1]
            if sqa_score < SQA_THRESHOLD:
                continue

            # 诊断
            diag_interp.set_tensor(diag_in, t)
            diag_interp.invoke()
            normal_prob = float(softmax(diag_interp.get_tensor(diag_out)[0])[DIAG_IDX_NORMAL])
            valid.append((sqa_score, normal_prob))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if not valid:
        return None, None, 0, total_wins, elapsed_ms

    weights  = [v[0] for v in valid]
    normals  = [v[1] for v in valid]
    avg_norm = sum(w * p for w, p in zip(weights, normals)) / sum(weights)
    pred     = 0 if avg_norm > DIAG_THRESHOLD else 1
    return pred, avg_norm, len(valid), total_wins, elapsed_ms


# ──────────────────────────────────────────
# 指标
# ──────────────────────────────────────────

def compute_metrics(tp, tn, fp, fn):
    n = tp + tn + fp + fn
    if n == 0:
        return {}
    se   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp   = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = (2 * prec * se / (prec + se)) if (prec + se) > 0 else 0.0
    return {
        "acc": (tp + tn) / n, "se": se, "sp": sp, "f1": f1,
        "mscore": (se + sp) / 2, "evaluated": n,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# ──────────────────────────────────────────
# SQA 验证打印（--verify 时调用）
# ──────────────────────────────────────────

def verify_sqa_outputs(rows, mel_cfg, sqa_path, n_samples=5):
    """打印若干 Bad / Good 样本的原始模型输出，方便核查索引约定。"""
    interp, in_idx, out_idx = load_interp(sqa_path)

    bad_rows  = [r for r in rows if r["label"] == 0][:n_samples]
    good_rows = [r for r in rows if r["label"] == 1][:n_samples]

    print(f"\n  [VERIFY] {os.path.basename(sqa_path)}")
    print(f"  {'fname':<12} {'meta_label':<12} {'idx0(Good?)':<14} {'idx1(Bad?)':<14} {'pred'}")
    print(f"  {'-'*65}")
    for r in bad_rows + good_rows:
        tensors = load_tensors(r["filepath"], mel_cfg)
        if not tensors:
            continue
        sm_list = []
        for t in tensors[:3]:           # 最多取前3个窗口
            interp.set_tensor(in_idx, t)
            interp.invoke()
            sm_list.append(softmax(interp.get_tensor(out_idx)[0]))
        avg = np.mean(sm_list, axis=0)
        pred = "Bad" if avg[SQA_IDX_BAD] > 0.5 else "Good"
        lbl  = "Bad(0)" if r["label"] == 0 else "Good(1)"
        print(f"  {r['fname']:<12} {lbl:<12} {avg[0]:.3f}          {avg[1]:.3f}          {pred}")


# ──────────────────────────────────────────
# 评估函数
# ──────────────────────────────────────────

def run_sqa_eval(rows, mel_cfg, sqa_path, label):
    """
    SQA 独立评估（切片级，与 evaluate_pc.py 一致）。
    metadata_quality: label 1=Good, 0=Bad → gt_sqa = 1-label（Bad→1, Good→0）
    每个文件拆成多个切片，每个切片独立预测，用 argmax 取类别。
    """
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}")
    interp, in_idx, out_idx = load_interp(sqa_path)

    tp = tn = fp = fn = skipped = 0

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_sqa   = 1 - row["label"]   # metadata 1=Good→0, 0=Bad→1

        if not os.path.exists(filepath):
            skipped += 1
            continue

        tensors = load_tensors(filepath, mel_cfg)
        if not tensors:
            skipped += 1
            continue

        for t in tensors:
            interp.set_tensor(in_idx, t)
            interp.invoke()
            pred = int(np.argmax(interp.get_tensor(out_idx)[0]))

            if   gt_sqa == 1 and pred == 1: tp += 1
            elif gt_sqa == 0 and pred == 0: tn += 1
            elif gt_sqa == 0 and pred == 1: fp += 1
            else:                           fn += 1

    m = compute_metrics(tp, tn, fp, fn)
    if not m:
        print("    无可评估样本")
        return None

    print(f"    TP={tp} TN={tn} FP={fp} FN={fn}  (skipped={skipped})")
    print(f"    Accuracy={m['acc']*100:.1f}%  M-Score={m['mscore']*100:.1f}%  "
          f"Se(Bad)={m['se']*100:.1f}%  Sp(Good)={m['sp']*100:.1f}%  "
          f"(evaluated={m['evaluated']} 切片)")
    return m


def run_diag_only_eval(rows, mel_cfg, diag_path, label):
    """
    诊断模型评估（解耦，无 SQA 门控）。
    切片级评估，与 evaluate_pc.py 对齐：
    每个切片独立 argmax → 切片级 TP/TN/FP/FN，文件 GT label 作为每条切片的标签。
    """
    print(f"\n  [{label}]  DIAG={os.path.basename(diag_path)}")
    interp, in_idx, out_idx = load_interp(diag_path)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_label = row["label"]   # physionet: 1=Abnormal, 0=Normal

        if not os.path.exists(filepath):
            skipped += 1
            continue

        tensors = load_tensors(filepath, mel_cfg)
        if not tensors:
            skipped += 1
            continue

        for t in tensors:
            t0 = time.perf_counter()
            interp.set_tensor(in_idx, t)
            interp.invoke()
            pred = int(np.argmax(interp.get_tensor(out_idx)[0]))
            elapsed_all.append((time.perf_counter() - t0) * 1000)

            if   gt_label == 1 and pred == 1: tp += 1
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
          f"(evaluated={m['evaluated']} 切片, skipped={skipped} 文件)")
    print(f"    推理耗时 mean={arr.mean():.2f}ms  "
          f"min={arr.min():.2f}ms  max={arr.max():.2f}ms")
    return m


def run_diag_coupled_eval(rows, mel_cfg, sqa_path, diag_path, label):
    """诊断模型评估（SQA 门控 + Good 概率加权平均）。"""
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}  "
          f"DIAG={os.path.basename(diag_path)}")
    sqa_interp,  sqa_in,  sqa_out  = load_interp(sqa_path)
    diag_interp, diag_in, diag_out = load_interp(diag_path)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_label = row["label"]

        if not os.path.exists(filepath):
            skipped += 1
            continue

        pred, _, valid_segs, total_segs, elapsed_ms = predict_diag_coupled(
            filepath, mel_cfg, sqa_interp, sqa_in, sqa_out,
            diag_interp, diag_in, diag_out)
        elapsed_all.append(elapsed_ms)

        if pred is None:
            skipped += 1
            continue

        if   gt_label == 1 and pred == 1: tp += 1
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


# ──────────────────────────────────────────
# 对比表格
# ──────────────────────────────────────────

def print_comparison(fp32_m, int8_m, title, metrics):
    if fp32_m is None or int8_m is None:
        return
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    print(f"  {'Metric':<14} {'FP32':>10} {'INT8':>10} {'Change':>10}")
    print(f"  {'-'*44}")
    for key, lbl in metrics:
        v32, v8 = fp32_m[key] * 100, int8_m[key] * 100
        print(f"  {lbl:<14} {v32:>9.1f}% {v8:>9.1f}% {v8-v32:>+9.1f}%")
    print(f"{'='*60}\n")


DIAG_METRICS = [("mscore","M-Score"), ("se","Sensitivity"),
                ("sp","Specificity"), ("acc","Accuracy")]
SQA_METRICS  = [("mscore","M-Score"), ("se","Se(Bad)"),
                ("sp","Sp(Good)"),    ("acc","Accuracy")]


# ──────────────────────────────────────────
# 入口
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diag", "sqa", "both"], default="both")
    parser.add_argument("--verify", action="store_true",
                        help="打印若干样本的原始模型输出，用于核查索引约定")
    args = parser.parse_args()

    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    # ── SQA 评估 ────────────────────────────
    if args.mode == "sqa":
        rows_sqa = build_lookup(SQA_META, SQA_SPLIT)
        bad_n  = sum(1 for r in rows_sqa if r["label"] == 0)
        good_n = sum(1 for r in rows_sqa if r["label"] == 1)

        print(f"\n{'='*60}")
        print("SQA 模型评估（test_split_sqa.csv）")
        print(f"  测试录音数：{len(rows_sqa)}  Bad(label=0)={bad_n}  Good(label=1)={good_n}")
        print(f"  索引约定：SQA_IDX_BAD={SQA_IDX_BAD}  SQA_IDX_GOOD={SQA_IDX_GOOD}")
        print(f"{'='*60}")

        if args.verify:
            verify_sqa_outputs(rows_sqa, mel_cfg, SQA_FP32)

        m_sqa_fp32 = run_sqa_eval(rows_sqa, mel_cfg, SQA_FP32, "FP32")
        m_sqa_int8 = run_sqa_eval(rows_sqa, mel_cfg, SQA_INT8, "INT8")
        print_comparison(m_sqa_fp32, m_sqa_int8,
                         "FP32 vs INT8 对比（SQA 模型）", SQA_METRICS)

    # ── 诊断模型（解耦）──────────────────────
    if args.mode == "diag":
        rows_diag = build_lookup(DIAG_META, DIAG_SPLIT)
        print(f"\n{'='*60}")
        print("诊断模型评估（解耦，无 SQA 门控，test_split.csv）")
        print(f"  测试录音数：{len(rows_diag)}")
        print(f"{'='*60}")
        m_diag_fp32 = run_diag_only_eval(rows_diag, mel_cfg, DIAG_FP32, "FP32")
        m_diag_int8 = run_diag_only_eval(rows_diag, mel_cfg, DIAG_INT8, "INT8")
        print_comparison(m_diag_fp32, m_diag_int8,
                         "FP32 vs INT8 对比（诊断模型，解耦）", DIAG_METRICS)

    # ── 诊断模型（耦合）──────────────────────
    if args.mode == "both":
        rows_diag = build_lookup(DIAG_META, DIAG_SPLIT)
        print(f"\n{'='*60}")
        print("诊断模型评估（耦合：SQA 门控 + 加权平均，test_split.csv）")
        print(f"  测试录音数：{len(rows_diag)}")
        print(f"  SQA_THRESHOLD={SQA_THRESHOLD}（sm[1] 分数，低于此值的窗口被过滤，与 main_pi.py 对齐）")
        print(f"{'='*60}")
        m_coupled_fp32 = run_diag_coupled_eval(
            rows_diag, mel_cfg, SQA_FP32, DIAG_FP32, "FP32")
        m_coupled_int8 = run_diag_coupled_eval(
            rows_diag, mel_cfg, SQA_INT8, DIAG_INT8, "INT8")
        print_comparison(m_coupled_fp32, m_coupled_int8,
                         "FP32 vs INT8 对比（诊断模型，耦合 SQA 门控）", DIAG_METRICS)


if __name__ == "__main__":
    main()
