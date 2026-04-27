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
    "/home/rasp4b/FypPi/WAV_record/002_2k.wav"
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
