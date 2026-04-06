"""
推理性能基准测试
用随机噪声模拟一个完整 chunk，测量各阶段耗时和资源占用。
模型训练完替换 .tflite 文件后直接重跑即可。

运行：
    python benchmark.py
    python benchmark.py --chunks 5   # 测多个 chunk 取均值
    python benchmark.py --wav path/to/file.wav  # 用真实音频
"""

import argparse
import os
import sys
import time
import wave

import numpy as np
import psutil
import yaml
import ai_edge_litert.interpreter as tflite

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.filters import apply_bandpass
from src.preprocess.mel import logmel_fixed_size

# ──────────────────────────────────────────
# 参数（与 main_pi.py 保持一致）
# ──────────────────────────────────────────
SAMPLE_RATE    = 2000
SEG_DURATION   = 2.0
OVERLAP        = 0.5
CHUNK_DURATION = 20
SQA_THRESHOLD  = 0.6

SEG_SAMPLES   = int(SAMPLE_RATE * SEG_DURATION)
HOP_SAMPLES   = int(SEG_SAMPLES * (1 - OVERLAP))
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION

SQA_MODEL  = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
DIAG_MODEL = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
CONFIG     = os.path.join(PROJECT_ROOT, "config.yaml")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def load_audio_from_wav(path):
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # 截断或补零到 CHUNK_SAMPLES
    if len(audio) >= CHUNK_SAMPLES:
        return audio[:CHUNK_SAMPLES]
    return np.pad(audio, (0, CHUNK_SAMPLES - len(audio)))


def run_benchmark(audio, mel_cfg, q_interp, d_interp, q_in, q_out, d_in, d_out):
    """
    对一个 chunk 跑完整推理，返回各阶段计时结果。
    """
    total_windows = int((CHUNK_SAMPLES - SEG_SAMPLES) / HOP_SAMPLES) + 1

    t_preprocess_filter = []   # 带通滤波（整块一次）
    t_mel        = []          # mel 计算（per window）
    t_sqa        = []          # SQA invoke（per window）
    t_diag       = []          # 诊断 invoke（per valid window）

    # ── 带通滤波（整块）──
    t0 = time.perf_counter()
    filtered = apply_bandpass(audio, fs=SAMPLE_RATE, lowcut=25, highcut=400)
    t_filter_total = (time.perf_counter() - t0) * 1000  # ms

    valid_count = 0

    for win_idx, start in enumerate(range(0, CHUNK_SAMPLES - SEG_SAMPLES + 1, HOP_SAMPLES)):
        window = filtered[start: start + SEG_SAMPLES]
        max_val = np.max(np.abs(window))
        if max_val > 0:
            window = window / max_val

        # ── mel ──
        t0 = time.perf_counter()
        mel    = logmel_fixed_size(y=window, sr=SAMPLE_RATE, mel_cfg=mel_cfg,
                                   target_shape=(mel_cfg["n_mels"], 64))
        tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)
        t_mel.append((time.perf_counter() - t0) * 1000)

        # ── SQA ──
        t0 = time.perf_counter()
        q_interp.set_tensor(q_in, tensor)
        q_interp.invoke()
        q_probs   = softmax(q_interp.get_tensor(q_out)[0])
        t_sqa.append((time.perf_counter() - t0) * 1000)

        sqa_score = float(q_probs[1])
        if sqa_score < SQA_THRESHOLD:
            continue

        # ── 诊断 ──
        t0 = time.perf_counter()
        d_interp.set_tensor(d_in, tensor)
        d_interp.invoke()
        _ = softmax(d_interp.get_tensor(d_out)[0])
        t_diag.append((time.perf_counter() - t0) * 1000)
        valid_count += 1

    return {
        "total_windows" : total_windows,
        "valid_windows" : valid_count,
        "t_filter_ms"   : t_filter_total,
        "t_mel_ms"      : t_mel,
        "t_sqa_ms"      : t_sqa,
        "t_diag_ms"     : t_diag,
    }


def print_stats(label, values_ms):
    if not values_ms:
        print(f"  {label:<20} 无数据")
        return
    arr = np.array(values_ms)
    print(f"  {label:<20} mean={arr.mean():.1f}ms  "
          f"min={arr.min():.1f}ms  max={arr.max():.1f}ms  "
          f"std={arr.std():.1f}ms  (n={len(arr)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=3,
                        help="测试 chunk 数量（默认 3）")
    parser.add_argument("--wav", type=str, default=None,
                        help="使用真实 WAV 文件（否则用随机噪声）")
    args = parser.parse_args()

    # ── 加载配置 ──
    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    # ── 加载模型 ──
    print(f"\n{'='*55}")
    print("模型文件")
    print(f"{'='*55}")
    for path, label in [(SQA_MODEL, "SQA"), (DIAG_MODEL, "诊断")]:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {label:<6} {os.path.basename(path):<35} {size_kb:.1f} KB")

    q_interp = tflite.Interpreter(model_path=SQA_MODEL)
    d_interp = tflite.Interpreter(model_path=DIAG_MODEL)
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()
    q_in  = q_interp.get_input_details()[0]["index"]
    q_out = q_interp.get_output_details()[0]["index"]
    d_in  = d_interp.get_input_details()[0]["index"]
    d_out = d_interp.get_output_details()[0]["index"]

    # ── 准备音频 ──
    if args.wav:
        print(f"\n使用 WAV 文件：{args.wav}")
        base_audio = load_audio_from_wav(args.wav)
    else:
        print(f"\n使用随机噪声（{CHUNK_DURATION}s × {args.chunks} chunks）")
        rng = np.random.default_rng(42)
        base_audio = rng.uniform(-0.1, 0.1, CHUNK_SAMPLES).astype(np.float32)

    # ── 系统基线 ──
    proc = psutil.Process()
    cpu_before  = psutil.cpu_percent(interval=1)
    mem_before  = proc.memory_info().rss / 1024 / 1024

    # ── 跑 benchmark ──
    all_filter, all_mel, all_sqa, all_diag, all_chunk = [], [], [], [], []

    print(f"\n{'='*55}")
    print(f"推理延迟（{args.chunks} chunks）")
    print(f"{'='*55}")

    for i in range(args.chunks):
        # 每个 chunk 略微加点噪声，避免完全相同
        rng2 = np.random.default_rng(i)
        audio = base_audio + rng2.uniform(-0.01, 0.01, CHUNK_SAMPLES).astype(np.float32)

        t_chunk_start = time.perf_counter()
        result = run_benchmark(audio, mel_cfg, q_interp, d_interp,
                               q_in, q_out, d_in, d_out)
        t_chunk_total = (time.perf_counter() - t_chunk_start) * 1000

        all_filter.append(result["t_filter_ms"])
        all_mel.extend(result["t_mel_ms"])
        all_sqa.extend(result["t_sqa_ms"])
        all_diag.extend(result["t_diag_ms"])
        all_chunk.append(t_chunk_total)

        print(f"  Chunk {i+1:02d}  总耗时={t_chunk_total/1000:.2f}s  "
              f"窗口={result['total_windows']}  有效={result['valid_windows']}")

    # ── 汇总 ──
    print(f"\n{'='*55}")
    print("各阶段平均耗时")
    print(f"{'='*55}")
    filter_arr = np.array(all_filter)
    print(f"  {'带通滤波(整块)':<20} mean={filter_arr.mean():.1f}ms  "
          f"min={filter_arr.min():.1f}ms  max={filter_arr.max():.1f}ms")
    print_stats("Mel 频谱(per win)",  all_mel)
    print_stats("SQA invoke(per win)", all_sqa)
    print_stats("诊断 invoke(per win)", all_diag)

    chunk_arr = np.array(all_chunk)
    print(f"\n  {'Chunk 总耗时':<20} mean={chunk_arr.mean()/1000:.2f}s  "
          f"min={chunk_arr.min()/1000:.2f}s  max={chunk_arr.max()/1000:.2f}s")
    print(f"  采集时长                {CHUNK_DURATION}s  "
          f"→ 实时性：{'✓ 可跟上' if chunk_arr.mean() < CHUNK_DURATION * 1000 else '✗ 跟不上'}")

    # ── 资源占用 ──
    cpu_after = psutil.cpu_percent(interval=1)
    mem_after = proc.memory_info().rss / 1024 / 1024
    print(f"\n{'='*55}")
    print("系统资源占用")
    print(f"{'='*55}")
    print(f"  CPU（推理前）  {cpu_before:.1f}%")
    print(f"  CPU（推理后）  {cpu_after:.1f}%")
    print(f"  内存（推理前） {mem_before:.1f} MB")
    print(f"  内存（推理后） {mem_after:.1f} MB  （Δ {mem_after - mem_before:+.1f} MB）")
    temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            temp = temps["cpu_thermal"][0].current
        elif "coretemp" in temps:
            temp = temps["coretemp"][0].current
    except Exception:
        pass
    if temp:
        print(f"  CPU 温度       {temp:.1f}°C")

    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()
