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
SQA_THRESHOLD = 0.6

TEST_WAV = os.path.join(
    PROJECT_ROOT,
    "/home/rasp4b/FypPi/WAV_record/011_2k.wav"
)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    # 加载配置
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # 加载模型
    q_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite"))
    d_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_model_quant.tflite"))
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()

    q_in  = q_interp.get_input_details()[0]['index']
    q_out = q_interp.get_output_details()[0]['index']
    d_in  = d_interp.get_input_details()[0]['index']
    d_out = d_interp.get_output_details()[0]['index']

    print(f"🎬 读取音频: {os.path.basename(TEST_WAV)}")
    tensors = preprocess_wav_for_pi(TEST_WAV, config)
    print(f"📦 共 {len(tensors)} 个片段")
    print("-" * 50)

    results = []  # (sqa_score, prob_normal)

    for i, tensor in enumerate(tensors):
        # 第一级：SQA 质量评估
        q_interp.set_tensor(q_in, tensor)
        q_interp.invoke()
        q_probs   = softmax(q_interp.get_tensor(q_out)[0])
        sqa_score = q_probs[1]  # index 1 = Good
        print(f"片段 {i+1:02d}: SQA → Poor={q_probs[0]:.2%} | Good={sqa_score:.2%}")

        if sqa_score < SQA_THRESHOLD:
            print(f"         ⚠️  质量不足，跳过")
            continue

        # 第二级：诊断模型
        d_interp.set_tensor(d_in, tensor)
        d_interp.invoke()
        d_probs     = softmax(d_interp.get_tensor(d_out)[0])
        prob_normal = d_probs[0]  # index 0 = Normal
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
