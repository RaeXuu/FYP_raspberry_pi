"""
TFLite 模型性能基准测试（树莓派端）
测试 6 个模型的推理延迟：FP32 / INT8 动态 / INT8 全整型 × SQA / Diagnosis

使用方式
  python benchmark_on_pi.py                  # 默认 100 轮，10 轮预热
  python benchmark_on_pi.py --runs 200       # 自定义轮数
  python benchmark_on_pi.py --warmup 20      # 自定义预热轮数
  python benchmark_on_pi.py --model diag     # 只测诊断模型
  python benchmark_on_pi.py --model sqa      # 只测 SQA 模型
"""

import argparse
import os
import sys
import time

import numpy as np

# ── TFLite 解释器：优先 ai_edge_litert，回退 tflite_runtime ──
try:
    import ai_edge_litert.interpreter as tflite
    TFLITE_BACKEND = "ai_edge_litert"
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        TFLITE_BACKEND = "tflite_runtime"
    except ImportError:
        raise ImportError("无法导入 ai_edge_litert 或 tflite_runtime，请确认已安装其中之一")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ──────────────────────────────────────────
# 模型路径
# ──────────────────────────────────────────
DIAG_FP32     = os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite")
DIAG_INT8     = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
DIAG_INT8FULL = os.path.join(PROJECT_ROOT, "heart_model_int8full.tflite")
SQA_FP32      = os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite")
SQA_INT8      = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
SQA_INT8FULL  = os.path.join(PROJECT_ROOT, "heart_quality_int8full.tflite")

MODEL_PATHS = {
    "sqa_fp32":     (SQA_FP32,      "SQA FP32"),
    "sqa_int8":     (SQA_INT8,      "SQA INT8动态"),
    "sqa_int8full": (SQA_INT8FULL,  "SQA INT8全整型"),
    "diag_fp32":    (DIAG_FP32,     "Diag FP32"),
    "diag_int8":    (DIAG_INT8,     "Diag INT8动态"),
    "diag_int8full":(DIAG_INT8FULL, "Diag INT8全整型"),
}

# 输入形状 (batch, channel, n_mels, time)
INPUT_SHAPE = (1, 1, 64, 64)


# ──────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────

def load_interp(model_path):
    """
    加载 TFLite 模型。
    返回 (interpreter, in_idx, out_idx,
           is_int8_in, in_scale, in_zp,
           is_int8_out, out_scale, out_zp)
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
    in_type_str  = "int8" if is_int8_in else "float32"
    out_type_str = "int8" if is_int8_out else "float32"
    print(f"  [load] {model_tag:<28s}  in={in_type_str:<7s}  out={out_type_str:<7s}  "
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


def warmup_interp(interp, in_idx, out_idx,
                  is_int8_in, in_scale, in_zp,
                  is_int8_out, out_scale, out_zp,
                  n_warmup=10):
    """预热：用随机数据跑 n_warmup 次推理，消除冷启动和缓存抖动。"""
    for _ in range(n_warmup):
        dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        t_q = quantize_input(dummy, is_int8_in, in_scale, in_zp)
        interp.set_tensor(in_idx, t_q)
        interp.invoke()
        _ = interp.get_tensor(out_idx)  # 触发输出内存分配


def bench_model(interp, in_idx, out_idx,
                is_int8_in, in_scale, in_zp,
                is_int8_out, out_scale, out_zp,
                n_runs=100):
    """
    用固定随机张量跑 n_runs 次推理，返回延迟列表 (ms)。
    每次推理包括 set_tensor + invoke + get_tensor，
    对 INT8 输出模型额外做一次反量化（计入耗时，模拟实际推理流水线）。
    """
    latencies = []
    dummy = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    t_q = quantize_input(dummy, is_int8_in, in_scale, in_zp)

    for _ in range(n_runs):
        t0 = time.perf_counter()
        interp.set_tensor(in_idx, t_q)
        interp.invoke()
        raw = interp.get_tensor(out_idx)
        if is_int8_out:
            # 模拟实际使用：反量化计入推理耗时
            _ = (raw.astype(np.float32) - out_zp) * out_scale
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def format_stats(latencies, unit="ms"):
    """格式化延迟统计。"""
    a = np.array(latencies)
    return (f"mean={np.mean(a):.2f}{unit}  median={np.median(a):.2f}{unit}  "
            f"p95={np.percentile(a, 95):.2f}{unit}  "
            f"min={np.min(a):.2f}{unit}  max={np.max(a):.2f}{unit}  "
            f"std={np.std(a):.2f}{unit}")


def get_pi_info():
    """获取树莓派系统信息（温度、CPU、内存），非 Pi 环境返回空。"""
    info = {}
    # CPU 温度 (Pi)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            info["cpu_temp_c"] = float(f.read().strip()) / 1000.0
    except Exception:
        pass

    # CPU 频率
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            info["cpu_freq_mhz"] = int(f.read().strip()) / 1000
    except Exception:
        pass

    # 内存
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_total_kb"] = int(line.split()[1])
                elif line.startswith("MemAvailable"):
                    info["mem_avail_kb"] = int(line.split()[1])
    except Exception:
        pass

    # 模型架构（ARM 芯片）
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "Model" in line:
                    info["cpu_model"] = line.split(":")[1].strip()[:60]
                    break
    except Exception:
        pass

    return info


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────

def run_benchmark(model_keys, n_runs=100, n_warmup=10):
    """对指定模型列表跑基准测试，打印结果表格。"""
    results = {}  # key → latencies

    for key in model_keys:
        path, label = MODEL_PATHS[key]

        if not os.path.exists(path):
            print(f"\n  [跳过] {label}: 文件不存在 ({os.path.basename(path)})")
            continue

        print(f"\n{'─'*60}")
        print(f"  模型: {label}")
        print(f"  文件: {path}")

        (interp, in_idx, out_idx,
         is_int8_in, in_scale, in_zp,
         is_int8_out, out_scale, out_zp) = load_interp(path)

        # 预热
        print(f"  预热 {n_warmup} 轮...", end=" ", flush=True)
        warmup_interp(interp, in_idx, out_idx,
                      is_int8_in, in_scale, in_zp,
                      is_int8_out, out_scale, out_zp,
                      n_warmup=n_warmup)
        print("完成")

        # 基准测试
        print(f"  基准测试 {n_runs} 轮...", end=" ", flush=True)
        latencies = bench_model(interp, in_idx, out_idx,
                                is_int8_in, in_scale, in_zp,
                                is_int8_out, out_scale, out_zp,
                                n_runs=n_runs)
        results[key] = latencies
        print("完成")
        print(f"  {format_stats(latencies)}")

    return results


def print_comparison_3way(results, model_prefix, label_prefix):
    """三栏对比：FP32 vs INT8 动态 vs INT8 全整型"""
    fp32_key     = f"{model_prefix}_fp32"
    int8_key     = f"{model_prefix}_int8"
    int8full_key = f"{model_prefix}_int8full"

    has_fp32     = fp32_key in results
    has_int8     = int8_key in results
    has_int8full = int8full_key in results

    if not has_fp32 or not has_int8:
        return

    fp32_lat     = results[fp32_key]
    int8_lat     = results[int8_key]
    int8full_lat = results[int8full_key] if has_int8full else None

    fp32_median  = np.median(fp32_lat)
    int8_median  = np.median(int8_lat)

    speedup_vs_fp32 = fp32_median / int8_median if int8_median > 0 else 0
    if int8full_lat is not None:
        int8full_median    = np.median(int8full_lat)
        speedup_full       = fp32_median / int8full_median if int8full_median > 0 else 0
    else:
        int8full_median = None
        speedup_full    = None

    print(f"\n{'='*72}")
    print(f"  {label_prefix} 三模型推理延迟对比 (median, ms)")
    print(f"{'='*72}")
    print(f"  {'指标':<20} {'FP32':>13} {'INT8动态':>13}", end="")
    if has_int8full:
        print(f" {'INT8全整型':>13}", end="")
    print()
    print(f"  {'-'*56}")
    print(f"  {'Median (ms)':<20} {fp32_median:>12.2f}  {int8_median:>12.2f} ", end="")
    if int8full_median is not None:
        print(f" {int8full_median:>12.2f} ", end="")
    print()
    print(f"  {'Speedup vs FP32':<20} {'1.00x':>13} {speedup_vs_fp32:>11.2f}x ", end="")
    if speedup_full is not None:
        print(f" {speedup_full:>11.2f}x ", end="")
    print()
    print(f"  {'-'*56}")
    print(f"  {'P95 (ms)':<20} {np.percentile(fp32_lat, 95):>12.2f}  "
          f"{np.percentile(int8_lat, 95):>12.2f} ", end="")
    if int8full_lat is not None:
        print(f" {np.percentile(int8full_lat, 95):>12.2f} ", end="")
    print()
    print(f"  {'Std (ms)':<20} {np.std(fp32_lat):>12.2f}  "
          f"{np.std(int8_lat):>12.2f} ", end="")
    if int8full_lat is not None:
        print(f" {np.std(int8full_lat):>12.2f} ", end="")
    print()
    print(f"{'='*72}")


def main():
    parser = argparse.ArgumentParser(description="TFLite 模型性能基准测试（树莓派端）")
    parser.add_argument("--model", choices=["all", "sqa", "diag"], default="all",
                        help="测试哪些模型 (default: all)")
    parser.add_argument("--runs", type=int, default=100,
                        help="基准测试推理轮数 (default: 100)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="预热推理轮数 (default: 10)")
    args = parser.parse_args()

    # 系统信息
    pi_info = get_pi_info()
    print(f"{'='*60}")
    print(f"  TFLite 模型性能基准测试")
    print(f"  TFLite 后端: {TFLITE_BACKEND}")
    print(f"  输入形状:   {INPUT_SHAPE}")
    print(f"  预热轮数:   {args.warmup}")
    print(f"  测试轮数:   {args.runs}")
    if pi_info:
        print(f"  {'─'*48}")
        if "cpu_model" in pi_info:
            print(f"  芯片:       {pi_info['cpu_model']}")
        if "cpu_temp_c" in pi_info:
            print(f"  CPU 温度:   {pi_info['cpu_temp_c']:.1f}°C")
        if "cpu_freq_mhz" in pi_info:
            print(f"  CPU 频率:   {pi_info['cpu_freq_mhz']:.0f} MHz")
        if "mem_total_kb" in pi_info and "mem_avail_kb" in pi_info:
            print(f"  内存:       {pi_info['mem_avail_kb']/1024:.0f} MB 可用 / "
                  f"{pi_info['mem_total_kb']/1024:.0f} MB 总量")
    print(f"{'='*60}")

    t_start = time.perf_counter()

    # 确定要测试的模型
    if args.model == "diag":
        keys = ["diag_fp32", "diag_int8", "diag_int8full"]
    elif args.model == "sqa":
        keys = ["sqa_fp32", "sqa_int8", "sqa_int8full"]
    else:
        keys = ["diag_fp32", "diag_int8", "diag_int8full",
                "sqa_fp32", "sqa_int8", "sqa_int8full"]

    results = run_benchmark(keys, n_runs=args.runs, n_warmup=args.warmup)

    # 三栏对比
    if args.model in ("diag", "all"):
        print_comparison_3way(results, "diag", "诊断模型 (Diagnosis)")
    if args.model in ("sqa", "all"):
        print_comparison_3way(results, "sqa", "SQA 质量检测模型")

    # 模型大小汇总
    print(f"\n{'='*60}")
    print(f"  模型文件大小")
    print(f"{'='*60}")
    print(f"  {'模型':<30} {'大小 (KB)':>10}")
    print(f"  {'─'*40}")
    for key in keys:
        if key in results:
            path, label = MODEL_PATHS[key]
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"  {label:<30} {size_kb:>9.1f}")

    elapsed = time.perf_counter() - t_start
    print(f"\n  总耗时: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()