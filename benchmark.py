"""
推理性能基准测试（FP32 vs INT8）
对单个 2s 窗口重复推理 N 次，取中位数延迟。
资源占用在全流程结束后统计。

运行：
    python benchmark.py                          # 随机噪声，100次
    python benchmark.py --wav a.wav b.wav        # 真实音频（取第一条）
    python benchmark.py --wav a.wav --runs 200   # 自定义重复次数
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

SAMPLE_RATE  = 2000
SEG_SAMPLES  = 4000   # 2s

DIAG_INT8 = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
DIAG_FP32 = os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite")
SQA_INT8  = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
SQA_FP32  = os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite")
CONFIG    = os.path.join(PROJECT_ROOT, "config.yaml")


def load_interp(model_path):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_idx  = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    return interp, in_idx, out_idx


def load_wav_segment(path):
    with wave.open(path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if len(audio) >= SEG_SAMPLES:
        return audio[:SEG_SAMPLES]
    return np.pad(audio, (0, SEG_SAMPLES - len(audio)))


def make_segment(wav_paths):
    """返回一个 2s float32 片段（真实音频或随机噪声）。"""
    if wav_paths:
        path = wav_paths[0]
        print(f"使用 WAV 文件：{path}")
        return load_wav_segment(path)
    rng = np.random.default_rng(42)
    print("使用随机噪声")
    return rng.uniform(-0.1, 0.1, SEG_SAMPLES).astype(np.float32)


def bench_stage(interp, in_idx, out_idx, tensor, runs):
    """对单个模型重复推理 runs 次，返回延迟列表（ms）。"""
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, tensor)
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def stats(vals, label, unit="ms"):
    arr = np.array(vals)
    med = np.median(arr)
    print(f"  {label:<28} median={med:.2f}{unit}  "
          f"mean={arr.mean():.2f}  min={arr.min():.2f}  "
          f"max={arr.max():.2f}  (n={len(arr)})")
    return med


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", nargs="*", default=None,
                        help="真实 WAV 文件路径（可多条，取第一条作为测试片段）")
    parser.add_argument("--runs", type=int, default=100,
                        help="每阶段重复推理次数（默认 100）")
    args = parser.parse_args()

    with open(CONFIG) as f:
        mel_cfg = yaml.safe_load(f)["mel"]

    segment = make_segment(args.wav)

    # ── 带通滤波 ──
    t_filter = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        filtered = apply_bandpass(segment, fs=SAMPLE_RATE, lowcut=25, highcut=400)
        t_filter.append((time.perf_counter() - t0) * 1000)
    # 用最后一次的 filtered 作为后续输入
    filtered = apply_bandpass(segment, fs=SAMPLE_RATE, lowcut=25, highcut=400)
    mx = np.max(np.abs(filtered))
    if mx > 0:
        filtered = filtered / mx

    # ── Mel 频谱 ──
    t_mel = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        mel = logmel_fixed_size(y=filtered, sr=SAMPLE_RATE, mel_cfg=mel_cfg,
                                target_shape=(mel_cfg["n_mels"], 64))
        t_mel.append((time.perf_counter() - t0) * 1000)
    mel = logmel_fixed_size(y=filtered, sr=SAMPLE_RATE, mel_cfg=mel_cfg,
                            target_shape=(mel_cfg["n_mels"], 64))
    tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

    # ── 加载四个模型 ──
    sqa_fp32,  sqa_fp32_in,  sqa_fp32_out  = load_interp(SQA_FP32)
    sqa_int8,  sqa_int8_in,  sqa_int8_out  = load_interp(SQA_INT8)
    diag_fp32, diag_fp32_in, diag_fp32_out = load_interp(DIAG_FP32)
    diag_int8, diag_int8_in, diag_int8_out = load_interp(DIAG_INT8)

    t_sqa_fp32  = bench_stage(sqa_fp32,  sqa_fp32_in,  sqa_fp32_out,  tensor, args.runs)
    t_sqa_int8  = bench_stage(sqa_int8,  sqa_int8_in,  sqa_int8_out,  tensor, args.runs)
    t_diag_fp32 = bench_stage(diag_fp32, diag_fp32_in, diag_fp32_out, tensor, args.runs)
    t_diag_int8 = bench_stage(diag_int8, diag_int8_in, diag_int8_out, tensor, args.runs)

    # ── 模型文件大小 ──
    sizes = {
        "SQA  FP32" : os.path.getsize(SQA_FP32)  / 1024,
        "SQA  INT8" : os.path.getsize(SQA_INT8)  / 1024,
        "Diag FP32" : os.path.getsize(DIAG_FP32) / 1024,
        "Diag INT8" : os.path.getsize(DIAG_INT8) / 1024,
    }

    # ── 资源占用 ──
    proc = psutil.Process()
    cpu_pct = psutil.cpu_percent(interval=1)
    mem_mb  = proc.memory_info().rss / 1024 / 1024
    temp = None
    try:
        temps = psutil.sensors_temperatures()
        if "cpu_thermal" in temps:
            temp = temps["cpu_thermal"][0].current
    except Exception:
        pass

    # ══════════════ 输出 ══════════════
    W = 60
    print(f"\n{'='*W}")
    print(f"模型文件大小")
    print(f"{'='*W}")
    for name, kb in sizes.items():
        print(f"  {name:<12} {kb:.1f} KB")

    print(f"\n{'='*W}")
    print(f"各阶段延迟（median of {args.runs} runs，单个 2s 窗口）")
    print(f"{'='*W}")
    med_filter    = stats(t_filter,    "Bandpass filter")
    med_mel       = stats(t_mel,       "Log-Mel spectrogram")
    med_sqa_fp32  = stats(t_sqa_fp32,  "SQA model     FP32")
    med_sqa_int8  = stats(t_sqa_int8,  "SQA model     INT8")
    med_diag_fp32 = stats(t_diag_fp32, "Diag model    FP32")
    med_diag_int8 = stats(t_diag_int8, "Diag model    INT8")

    print(f"\n{'='*W}")
    print("FP32 vs INT8 对比（Table 6.1）")
    print(f"{'='*W}")
    print(f"  {'Stage':<28} {'FP32 (ms)':>10} {'INT8 (ms)':>10} {'Speedup':>10}")
    print(f"  {'-'*58}")
    print(f"  {'Bandpass filter':<28} {'—':>10} {'—':>10} {'—':>10}")
    print(f"  {'Log-Mel spectrogram':<28} {'—':>10} {'—':>10} {'—':>10}")
    for stage, fp32, int8 in [("SQA model",  med_sqa_fp32,  med_sqa_int8),
                               ("Diag model", med_diag_fp32, med_diag_int8)]:
        speedup = fp32 / int8 if int8 > 0 else float("nan")
        print(f"  {stage:<28} {fp32:>9.2f}ms {int8:>9.2f}ms {speedup:>9.2f}x")

    tot_fp32 = med_filter + med_mel + med_sqa_fp32 + med_diag_fp32
    tot_int8 = med_filter + med_mel + med_sqa_int8 + med_diag_int8
    print(f"  {'Total per segment':<28} {tot_fp32:>9.2f}ms {tot_int8:>9.2f}ms")
    print(f"\n  实时性约束：< 2000ms/segment")
    print(f"  FP32 总延迟：{tot_fp32:.1f}ms  {'✓' if tot_fp32 < 2000 else '✗'}")
    print(f"  INT8 总延迟：{tot_int8:.1f}ms  {'✓' if tot_int8 < 2000 else '✗'}")

    print(f"\n{'='*W}")
    print("系统资源占用（Table 6.2）")
    print(f"{'='*W}")
    print(f"  Peak CPU utilisation   {cpu_pct:.1f}%")
    print(f"  Memory (RSS)           {mem_mb:.1f} MB")
    if temp:
        print(f"  CPU temperature        {temp:.1f} °C")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
