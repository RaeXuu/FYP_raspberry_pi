import os
import sys
import numpy as np
import ai_edge_litert.interpreter as tflite
import yaml

# ==========================================
# 环境初始化
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.preprocess_pipeline import preprocess_wav_for_pi

# ==========================================
# 配置（与 main_pi.py 保持一致）
# ==========================================
SQA_THRESHOLD = 0.05  # 与 main_pi.py 对齐

TEST_WAV = os.path.join(
    PROJECT_ROOT,
    "/home/rasp4b/FypPi/data/raw/DataSet2/training-a/a0002.wav"
)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def quantize_input(data, is_int8, scale, zp):
    if is_int8:
        q = data / scale + zp
        return np.clip(q, -128, 127).astype(np.int8)
    return data.astype(np.float32)


def dequantize_output(raw, is_int8, scale, zp):
    if is_int8:
        return (raw.astype(np.float32) - zp) * scale
    return raw.astype(np.float32)


def main():
    # 加载配置
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # 加载模型
    q_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_quality_int8full.tflite"))
    d_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_model_int8full.tflite"))
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()

    q_in_info  = q_interp.get_input_details()[0]
    q_out_info = q_interp.get_output_details()[0]
    d_in_info  = d_interp.get_input_details()[0]
    d_out_info = d_interp.get_output_details()[0]

    q_in  = q_in_info['index']
    q_out = q_out_info['index']
    d_in  = d_in_info['index']
    d_out = d_out_info['index']

    def _q(info):
        is_int8 = info['dtype'] in (np.int8, np.uint8)
        scale, zp = info.get('quantization', (0.0, 0))
        return (is_int8, scale, zp)

    q_qi = _q(q_in_info)
    q_qo = _q(q_out_info)
    d_qi = _q(d_in_info)
    d_qo = _q(d_out_info)

    print(f"🎬 读取音频: {os.path.basename(TEST_WAV)}")
    tensors = preprocess_wav_for_pi(TEST_WAV, config)
    print(f"📦 共 {len(tensors)} 个片段")
    print("-" * 50)

    mel0 = tensors[0].flatten()
    print("[Debug] Mel[0..<10] =", [f"{v:.6f}" for v in mel0[:10]])
    print("[Debug] Mel[64..<74] =", [f"{v:.6f}" for v in mel0[64:74]])
    print(f"[Debug] Mel min={mel0.min():.6f} max={mel0.max():.6f} "
          f"mean={mel0.mean():.6f}")

    results = []  # (sqa_score, prob_normal)

    for i, tensor in enumerate(tensors):
        # SQA
        t_q = quantize_input(tensor, *q_qi)
        q_interp.set_tensor(q_in, t_q)
        q_interp.invoke()
        q_probs = softmax(dequantize_output(
            q_interp.get_tensor(q_out), *q_qo)[0])
        sqa_score = float(q_probs[0])  # index 0 = Good
        print(f"片段 {i+1:02d}: SQA → Good={sqa_score:.2%} | Bad={q_probs[1]:.2%}")

        if sqa_score < SQA_THRESHOLD:
            print(f"         ⚠️  质量不足，跳过")
            continue

        # 诊断
        t_d = quantize_input(tensor, *d_qi)
        d_interp.set_tensor(d_in, t_d)
        d_interp.invoke()
        d_probs = softmax(dequantize_output(
            d_interp.get_tensor(d_out), *d_qo)[0])
        prob_normal = float(d_probs[0])  # index 0 = Normal
        diag_label  = "Normal" if prob_normal > 0.5 else "Abnormal"
        print(f"         诊断 → {diag_label} | Normal={prob_normal:.2%}")

        results.append((sqa_score, prob_normal))

    # ==========================================
    # 汇总：SQA 加权平均
    # ==========================================
    print("\n" + "=" * 50)

    if not results:
        print("⚠️  所有片段质量不足，无法诊断。")
        return

    weights           = [r[0] for r in results]
    probs             = [r[1] for r in results]
    final_prob_normal = sum(w * p for w, p in zip(weights, probs)) / sum(weights)

    label      = "Normal" if final_prob_normal > 0.5 else "Abnormal"
    confidence = final_prob_normal if label == "Normal" else 1 - final_prob_normal

    print(f"✨ 最终诊断: {label} | 置信度: {confidence:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()

#！Result：

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/main_pi_debug.py
# INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
# 🎬 读取音频: a0002.wav
# 📦 共 19 个片段
# --------------------------------------------------
# [Debug] Mel[0..<10] = ['10.032022', '8.923600', '-11.701008', '-0.642810', '0.892454', '-12.124862', '-4.820803', '13.502468', '9.678809', '-11.007860']
# [Debug] Mel[64..<74] = ['10.045051', '6.022046', '-8.502804', '-2.575232', '-1.280083', '-10.762410', '-8.216228', '10.381856', '0.120296', '-2.083143']
# [Debug] Mel min=-58.917763 max=13.502468 mean=-10.894194
# 片段 01: SQA → Good=31.78% | Bad=68.22%
#          诊断 → Abnormal | Normal=2.82%
# 片段 02: SQA → Good=7.87% | Bad=92.13%
#          诊断 → Abnormal | Normal=2.15%
# 片段 03: SQA → Good=2.54% | Bad=97.46%
#          ⚠️  质量不足，跳过
# 片段 04: SQA → Good=17.22% | Bad=82.78%
#          诊断 → Abnormal | Normal=2.30%
# 片段 05: SQA → Good=9.55% | Bad=90.45%
#          诊断 → Abnormal | Normal=0.31%
# 片段 06: SQA → Good=10.31% | Bad=89.69%
#          诊断 → Abnormal | Normal=0.94%
# 片段 07: SQA → Good=29.97% | Bad=70.03%
#          诊断 → Abnormal | Normal=2.30%
# 片段 08: SQA → Good=20.46% | Bad=79.54%
#          诊断 → Abnormal | Normal=2.46%
# 片段 09: SQA → Good=11.98% | Bad=88.02%
#          诊断 → Abnormal | Normal=1.33%
# 片段 10: SQA → Good=6.46% | Bad=93.54%
#          诊断 → Abnormal | Normal=1.64%
# 片段 11: SQA → Good=11.98% | Bad=88.02%
#          诊断 → Abnormal | Normal=1.24%
# 片段 12: SQA → Good=13.89% | Bad=86.11%
#          诊断 → Abnormal | Normal=2.46%
# 片段 13: SQA → Good=2.06% | Bad=97.94%
#          ⚠️  质量不足，跳过
# 片段 14: SQA → Good=6.46% | Bad=93.54%
#          诊断 → Abnormal | Normal=1.33%
# 片段 15: SQA → Good=70.91% | Bad=29.09%
#          诊断 → Abnormal | Normal=0.27%
# 片段 16: SQA → Good=30.87% | Bad=69.13%
#          诊断 → Abnormal | Normal=0.36%
# 片段 17: SQA → Good=11.98% | Bad=88.02%
#          诊断 → Abnormal | Normal=1.08%
# 片段 18: SQA → Good=5.08% | Bad=94.92%
#          诊断 → Abnormal | Normal=1.24%
# 片段 19: SQA → Good=9.55% | Bad=90.45%
#          诊断 → Abnormal | Normal=3.22%

# ==================================================
# ✨ 最终诊断: Abnormal | 置信度: 98.56%
# ==================================================