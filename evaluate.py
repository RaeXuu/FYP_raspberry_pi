"""
量化模型准确率评估（Pi 端）
对测试集每条音频跑完整推理流水线，对比 FP32 vs INT8 动态 vs INT8 全整型。

模型输出索引约定
  SQA 模型（heart_quality_*.tflite）：
    训练时 label 0 = Good, label 1 = Bad（reversed convention）
    → index 0 = Good 概率, index 1 = Bad 概率

  诊断模型（heart_model_*.tflite）：
    训练时 label 0 = Normal, label 1 = Abnormal
    → index 0 = Normal 概率, index 1 = Abnormal 概率

使用方式
  python evaluate_tflite_on_pi.py --mode sqa          # SQA 独立评估
  python evaluate_tflite_on_pi.py --mode diag         # 诊断模型（无 SQA 门控）
  python evaluate_tflite_on_pi.py --mode both         # 耦合流水线（SQA 门控 + 加权）
  python evaluate_tflite_on_pi.py --mode sqa --verify # 额外打印若干样本输出，排查索引
  python evaluate_tflite_on_pi.py --mode all          # 全部模式依次执行
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
DIAG_FP32     = os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite")
DIAG_INT8     = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
DIAG_INT8FULL = os.path.join(PROJECT_ROOT, "heart_model_int8full.tflite")
SQA_FP32      = os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite")
SQA_INT8      = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
SQA_INT8FULL  = os.path.join(PROJECT_ROOT, "heart_quality_int8full.tflite")

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
    """
    加载 TFLite 模型，返回 (interpreter, in_idx, out_idx,
    is_int8_in, in_scale, in_zp, is_int8_out, out_scale, out_zp)
    """
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()

    in_info  = interp.get_input_details()[0]
    out_info = interp.get_output_details()[0]
    in_idx   = in_info["index"]
    out_idx  = out_info["index"]

    in_dtype  = in_info["dtype"]
    out_dtype = out_info["dtype"]
    is_int8_in  = in_dtype in (np.int8, np.uint8)
    is_int8_out = out_dtype in (np.int8, np.uint8)

    in_scale, in_zp = (0.0, 0)
    out_scale, out_zp = (0.0, 0)
    if is_int8_in:
        in_scale, in_zp = in_info["quantization"]
    if is_int8_out:
        out_scale, out_zp = out_info["quantization"]

    model_tag = os.path.basename(model_path)
    in_type_str = "int8" if is_int8_in else "float32"
    out_type_str = "int8" if is_int8_out else "float32"
    print(f"  [load] {model_tag}  in={in_type_str}  out={out_type_str}  "
          f"in_scale={in_scale:.6f}  out_scale={out_scale:.6f}")

    return (interp, in_idx, out_idx,
            is_int8_in, in_scale, in_zp,
            is_int8_out, out_scale, out_zp)


def quantize_input(data, is_int8, scale, zp):
    """如果模型输入是 INT8，将 float32 数据量化为 INT8。"""
    if is_int8:
        q = data / scale + zp
        return np.clip(q, -128, 127).astype(np.int8)
    return data.astype(np.float32)


def dequantize_output(raw, is_int8, scale, zp):
    """如果模型输出是 INT8，将 INT8 数据反量化为 float32。"""
    if is_int8:
        return (raw.astype(np.float32) - zp) * scale
    return raw


def warmup_interp(interp, in_idx, out_idx,
                  is_int8_in, in_scale, in_zp,
                  is_int8_out, out_scale, out_zp,
                  n_warmup=10, input_shape=(1, 1, 64, 64)):
    """预热：用随机数据跑 n_warmup 次推理，消除冷启动和缓存抖动。"""
    for _ in range(n_warmup):
        dummy = np.random.randn(*input_shape).astype(np.float32)
        t_q = quantize_input(dummy, is_int8_in, in_scale, in_zp)
        interp.set_tensor(in_idx, t_q)
        interp.invoke()
        _ = interp.get_tensor(out_idx)  # 不反量化，仅预热


def format_timing(arr, unit="ms"):
    """格式化延迟统计：mean / median / p95 / min / max / std。"""
    a = np.array(arr) if len(arr) > 0 else np.array([0])
    return (f"mean={np.mean(a):.2f}{unit}  median={np.median(a):.2f}{unit}  "
            f"p95={np.percentile(a, 95):.2f}{unit}  "
            f"min={np.min(a):.2f}{unit}  max={np.max(a):.2f}{unit}  "
            f"std={np.std(a):.2f}{unit}")


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
# 推理（支持 INT8 输入/输出）
# ──────────────────────────────────────────

def infer_sqa(tensors, interp, in_idx, out_idx,
              is_int8_in, in_scale, in_zp,
              is_int8_out, out_scale, out_zp):
    """返回 (avg_prob_bad, avg_prob_good)。"""
    bad_probs = []
    for t in tensors:
        t_q = quantize_input(t, is_int8_in, in_scale, in_zp)
        interp.set_tensor(in_idx, t_q)
        interp.invoke()
        raw = interp.get_tensor(out_idx)
        out = dequantize_output(raw, is_int8_out, out_scale, out_zp)
        sm = softmax(out[0])
        bad_probs.append(sm[SQA_IDX_BAD])
    avg_bad = float(np.mean(bad_probs)) if bad_probs else 0.5
    return avg_bad, 1.0 - avg_bad


def infer_diag(tensors, interp, in_idx, out_idx,
               is_int8_in, in_scale, in_zp,
               is_int8_out, out_scale, out_zp):
    """返回 avg_prob_normal。"""
    normal_probs = []
    for t in tensors:
        t_q = quantize_input(t, is_int8_in, in_scale, in_zp)
        interp.set_tensor(in_idx, t_q)
        interp.invoke()
        raw = interp.get_tensor(out_idx)
        out = dequantize_output(raw, is_int8_out, out_scale, out_zp)
        sm = softmax(out[0])
        normal_probs.append(sm[DIAG_IDX_NORMAL])
    return float(np.mean(normal_probs)) if normal_probs else 0.5


def predict_sqa(filepath, mel_cfg, interp, in_idx, out_idx,
                is_int8_in, in_scale, in_zp,
                is_int8_out, out_scale, out_zp):
    """SQA 推理，返回 (pred, avg_prob_bad, elapsed_ms)。"""
    t0 = time.perf_counter()
    tensors = load_tensors(filepath, mel_cfg)
    if not tensors:
        return None, None, (time.perf_counter() - t0) * 1000

    avg_bad, _ = infer_sqa(tensors, interp, in_idx, out_idx,
                           is_int8_in, in_scale, in_zp,
                           is_int8_out, out_scale, out_zp)
    pred = 1 if avg_bad > 0.5 else 0
    return pred, avg_bad, (time.perf_counter() - t0) * 1000


def predict_diag_only(filepath, mel_cfg, interp, in_idx, out_idx,
                      is_int8_in, in_scale, in_zp,
                      is_int8_out, out_scale, out_zp):
    """诊断推理（无 SQA 门控），返回 (pred, avg_prob_normal, n_segs, elapsed_ms)。"""
    t0 = time.perf_counter()
    tensors = load_tensors(filepath, mel_cfg)
    if not tensors:
        return None, None, 0, (time.perf_counter() - t0) * 1000

    avg_normal = infer_diag(tensors, interp, in_idx, out_idx,
                            is_int8_in, in_scale, in_zp,
                            is_int8_out, out_scale, out_zp)
    pred = 0 if avg_normal > DIAG_THRESHOLD else 1
    return pred, avg_normal, len(tensors), (time.perf_counter() - t0) * 1000


def predict_diag_coupled(filepath, mel_cfg,
                         sqa_interp, sqa_in, sqa_out,
                         sqa_is_int8_in, sqa_in_scale, sqa_in_zp,
                         sqa_is_int8_out, sqa_out_scale, sqa_out_zp,
                         diag_interp, diag_in, diag_out,
                         diag_is_int8_in, diag_in_scale, diag_in_zp,
                         diag_is_int8_out, diag_out_scale, diag_out_zp):
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
            continue

        for win_start in range(0, len(chunk) - SEG_SAMPLES + 1, HOP_SAMPLES):
            window = chunk[win_start : win_start + SEG_SAMPLES]
            total_wins += 1

            mx = np.max(np.abs(window))
            if mx > 0:
                window = window / mx

            mel = logmel_fixed_size(y=window, sr=sr, mel_cfg=mel_cfg,
                                    target_shape=(mel_cfg["n_mels"], 64))
            t = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

            # SQA 门控（支持 INT8）
            t_q = quantize_input(t, sqa_is_int8_in, sqa_in_scale, sqa_in_zp)
            sqa_interp.set_tensor(sqa_in, t_q)
            sqa_interp.invoke()
            sqa_raw = sqa_interp.get_tensor(sqa_out)
            sqa_out_f = dequantize_output(sqa_raw, sqa_is_int8_out,
                                          sqa_out_scale, sqa_out_zp)
            sqa_score = float(softmax(sqa_out_f[0])[1])
            if sqa_score < SQA_THRESHOLD:
                continue

            # 诊断推理（支持 INT8）
            t_q2 = quantize_input(t, diag_is_int8_in, diag_in_scale, diag_in_zp)
            diag_interp.set_tensor(diag_in, t_q2)
            diag_interp.invoke()
            diag_raw = diag_interp.get_tensor(diag_out)
            diag_out_f = dequantize_output(diag_raw, diag_is_int8_out,
                                           diag_out_scale, diag_out_zp)
            normal_prob = float(softmax(diag_out_f[0])[DIAG_IDX_NORMAL])
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
    (interp, in_idx, out_idx,
     is_int8_in, in_scale, in_zp,
     is_int8_out, out_scale, out_zp) = load_interp(sqa_path)

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
        for t in tensors[:3]:
            t_q = quantize_input(t, is_int8_in, in_scale, in_zp)
            interp.set_tensor(in_idx, t_q)
            interp.invoke()
            raw = interp.get_tensor(out_idx)
            out = dequantize_output(raw, is_int8_out, out_scale, out_zp)
            sm_list.append(softmax(out[0]))
        avg = np.mean(sm_list, axis=0)
        pred = "Bad" if avg[SQA_IDX_BAD] > 0.5 else "Good"
        lbl  = "Bad(0)" if r["label"] == 0 else "Good(1)"
        print(f"  {r['fname']:<12} {lbl:<12} {avg[0]:.3f}          {avg[1]:.3f}          {pred}")


# ──────────────────────────────────────────
# 评估函数
# ──────────────────────────────────────────

def run_sqa_eval(rows, mel_cfg, sqa_path, label):
    """SQA 独立评估（切片级）。"""
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}")
    if not os.path.exists(sqa_path):
        print(f"    文件不存在，跳过")
        return None

    (interp, in_idx, out_idx,
     is_int8_in, in_scale, in_zp,
     is_int8_out, out_scale, out_zp) = load_interp(sqa_path)

    # 预热
    warmup_interp(interp, in_idx, out_idx,
                  is_int8_in, in_scale, in_zp,
                  is_int8_out, out_scale, out_zp,
                  n_warmup=10)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_sqa   = 1 - row["label"]

        if not os.path.exists(filepath):
            skipped += 1
            continue

        tensors = load_tensors(filepath, mel_cfg)
        if not tensors:
            skipped += 1
            continue

        for t in tensors:
            t0 = time.perf_counter()
            t_q = quantize_input(t, is_int8_in, in_scale, in_zp)
            interp.set_tensor(in_idx, t_q)
            interp.invoke()
            raw = interp.get_tensor(out_idx)
            out = dequantize_output(raw, is_int8_out, out_scale, out_zp)
            pred = int(np.argmax(out[0]))
            elapsed_all.append((time.perf_counter() - t0) * 1000)

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
    print(f"    推理耗时 {format_timing(elapsed_all, 'ms')}")
    return m


def run_diag_only_eval(rows, mel_cfg, diag_path, label):
    """诊断模型评估（解耦，无 SQA 门控，切片级）。"""
    print(f"\n  [{label}]  DIAG={os.path.basename(diag_path)}")
    if not os.path.exists(diag_path):
        print(f"    文件不存在，跳过")
        return None

    (interp, in_idx, out_idx,
     is_int8_in, in_scale, in_zp,
     is_int8_out, out_scale, out_zp) = load_interp(diag_path)

    # 预热
    warmup_interp(interp, in_idx, out_idx,
                  is_int8_in, in_scale, in_zp,
                  is_int8_out, out_scale, out_zp,
                  n_warmup=10)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_label = row["label"]

        if not os.path.exists(filepath):
            skipped += 1
            continue

        tensors = load_tensors(filepath, mel_cfg)
        if not tensors:
            skipped += 1
            continue

        for t in tensors:
            t0 = time.perf_counter()
            t_q = quantize_input(t, is_int8_in, in_scale, in_zp)
            interp.set_tensor(in_idx, t_q)
            interp.invoke()
            raw = interp.get_tensor(out_idx)
            out = dequantize_output(raw, is_int8_out, out_scale, out_zp)
            pred = int(np.argmax(out[0]))
            elapsed_all.append((time.perf_counter() - t0) * 1000)

            if   gt_label == 1 and pred == 1: tp += 1
            elif gt_label == 0 and pred == 0: tn += 1
            elif gt_label == 0 and pred == 1: fp += 1
            else:                             fn += 1

    m = compute_metrics(tp, tn, fp, fn)
    if not m:
        print("    无可评估样本")
        return None

    print(f"    Accuracy={m['acc']*100:.1f}%  M-Score={m['mscore']*100:.1f}%  "
          f"Se={m['se']*100:.1f}%  Sp={m['sp']*100:.1f}%  "
          f"(evaluated={m['evaluated']} 切片, skipped={skipped} 文件)")
    print(f"    推理耗时 {format_timing(elapsed_all, 'ms')}")
    return m


def run_diag_coupled_eval(rows, mel_cfg, sqa_path, diag_path, label):
    """诊断模型评估（SQA 门控 + Good 概率加权平均，文件级）。"""
    print(f"\n  [{label}]  SQA={os.path.basename(sqa_path)}  "
          f"DIAG={os.path.basename(diag_path)}")
    if not os.path.exists(sqa_path) or not os.path.exists(diag_path):
        print(f"    文件不存在，跳过")
        return None

    (sqa_interp, sqa_in, sqa_out,
     sqa_is_int8_in, sqa_in_scale, sqa_in_zp,
     sqa_is_int8_out, sqa_out_scale, sqa_out_zp) = load_interp(sqa_path)
    (diag_interp, diag_in, diag_out,
     diag_is_int8_in, diag_in_scale, diag_in_zp,
     diag_is_int8_out, diag_out_scale, diag_out_zp) = load_interp(diag_path)

    # 预热两个模型
    warmup_interp(sqa_interp, sqa_in, sqa_out,
                  sqa_is_int8_in, sqa_in_scale, sqa_in_zp,
                  sqa_is_int8_out, sqa_out_scale, sqa_out_zp,
                  n_warmup=5)
    warmup_interp(diag_interp, diag_in, diag_out,
                  diag_is_int8_in, diag_in_scale, diag_in_zp,
                  diag_is_int8_out, diag_out_scale, diag_out_zp,
                  n_warmup=5)

    tp = tn = fp = fn = skipped = 0
    elapsed_all = []
    valid_counts = []
    total_counts = []

    for row in tqdm(rows, desc=f"    {label}", unit="file", leave=True):
        filepath = row["filepath"]
        gt_label = row["label"]

        if not os.path.exists(filepath):
            skipped += 1
            continue

        pred, _, valid_segs, total_segs, elapsed_ms = predict_diag_coupled(
            filepath, mel_cfg,
            sqa_interp, sqa_in, sqa_out,
            sqa_is_int8_in, sqa_in_scale, sqa_in_zp,
            sqa_is_int8_out, sqa_out_scale, sqa_out_zp,
            diag_interp, diag_in, diag_out,
            diag_is_int8_in, diag_in_scale, diag_in_zp,
            diag_is_int8_out, diag_out_scale, diag_out_zp)
        elapsed_all.append(elapsed_ms)
        valid_counts.append(valid_segs)
        total_counts.append(total_segs)

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

    avg_valid = np.mean(valid_counts) if valid_counts else 0
    avg_total = np.mean(total_counts) if total_counts else 0
    print(f"    Accuracy={m['acc']*100:.1f}%  M-Score={m['mscore']*100:.1f}%  "
          f"Se={m['se']*100:.1f}%  Sp={m['sp']*100:.1f}%  "
          f"(n={m['evaluated']}, skipped={skipped})")
    print(f"    有效窗口/总窗口: {avg_valid:.0f}/{avg_total:.0f} (平均)")
    print(f"    文件级推理耗时 {format_timing(elapsed_all, 'ms')}")
    return m


# ──────────────────────────────────────────
# 对比表格
# ──────────────────────────────────────────

def print_comparison_3way(fp32_m, int8_m, int8full_m, title, metrics):
    """三栏对比：FP32 vs INT8 动态 vs INT8 全整型"""
    if fp32_m is None or int8_m is None:
        return
    print(f"\n{'='*72}")
    print(title)
    print(f"{'='*72}")
    print(f"  {'Metric':<14} {'FP32':>10} {'INT8动态':>10} {'INT8全整型':>10}")
    print(f"  {'-'*56}")
    for key, lbl in metrics:
        v32    = fp32_m[key] * 100
        v8     = int8_m[key] * 100
        if int8full_m:
            v8full = int8full_m[key] * 100
            print(f"  {lbl:<14} {v32:>9.1f}% {v8:>9.1f}% {v8full:>9.1f}%")
        else:
            print(f"  {lbl:<14} {v32:>9.1f}% {v8:>9.1f}% {'N/A':>10}")
    print(f"{'='*72}\n")


DIAG_METRICS = [("mscore","M-Score"), ("se","Sensitivity"),
                ("sp","Specificity"), ("acc","Accuracy")]
SQA_METRICS  = [("mscore","M-Score"), ("se","Se(Bad)"),
                ("sp","Sp(Good)"),    ("acc","Accuracy")]


# ──────────────────────────────────────────
# 入口
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diag", "sqa", "both", "all"], default="both")
    parser.add_argument("--verify", action="store_true",
                        help="打印若干样本的原始模型输出，用于核查索引约定")
    args = parser.parse_args()

    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    # ── SQA 评估 ────────────────────────────
    if args.mode in ("sqa", "all"):
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

        m_sqa_fp32     = run_sqa_eval(rows_sqa, mel_cfg, SQA_FP32,     "FP32")
        m_sqa_int8     = run_sqa_eval(rows_sqa, mel_cfg, SQA_INT8,     "INT8动态")
        m_sqa_int8full = run_sqa_eval(rows_sqa, mel_cfg, SQA_INT8FULL, "INT8全整型")
        print_comparison_3way(m_sqa_fp32, m_sqa_int8, m_sqa_int8full,
                              "FP32 vs INT8动态 vs INT8全整型（SQA 模型）", SQA_METRICS)

    # ── 诊断模型（解耦）──────────────────────
    if args.mode in ("diag", "all"):
        rows_diag = build_lookup(DIAG_META, DIAG_SPLIT)
        print(f"\n{'='*60}")
        print("诊断模型评估（解耦，无 SQA 门控，test_split.csv）")
        print(f"  测试录音数：{len(rows_diag)}")
        print(f"{'='*60}")
        m_diag_fp32     = run_diag_only_eval(rows_diag, mel_cfg, DIAG_FP32,     "FP32")
        m_diag_int8     = run_diag_only_eval(rows_diag, mel_cfg, DIAG_INT8,     "INT8动态")
        m_diag_int8full = run_diag_only_eval(rows_diag, mel_cfg, DIAG_INT8FULL, "INT8全整型")
        print_comparison_3way(m_diag_fp32, m_diag_int8, m_diag_int8full,
                              "FP32 vs INT8动态 vs INT8全整型（诊断模型，解耦）", DIAG_METRICS)

    # ── 诊断模型（耦合）──────────────────────
    if args.mode in ("both", "all"):
        rows_diag = build_lookup(DIAG_META, DIAG_SPLIT)
        print(f"\n{'='*60}")
        print("诊断模型评估（耦合：SQA 门控 + 加权平均，test_split.csv）")
        print(f"  测试录音数：{len(rows_diag)}")
        print(f"  SQA_THRESHOLD={SQA_THRESHOLD}（sm[1] 分数，低于此值的窗口被过滤）")
        print(f"{'='*60}")
        m_coupled_fp32     = run_diag_coupled_eval(rows_diag, mel_cfg, SQA_FP32,     DIAG_FP32,     "FP32")
        m_coupled_int8     = run_diag_coupled_eval(rows_diag, mel_cfg, SQA_INT8,     DIAG_INT8,     "INT8动态")
        m_coupled_int8full = run_diag_coupled_eval(rows_diag, mel_cfg, SQA_INT8FULL, DIAG_INT8FULL, "INT8全整型")
        print_comparison_3way(m_coupled_fp32, m_coupled_int8, m_coupled_int8full,
                              "FP32 vs INT8动态 vs INT8全整型（诊断模型，耦合 SQA 门控）", DIAG_METRICS)


if __name__ == "__main__":
    main()