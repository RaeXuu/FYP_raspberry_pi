import os
import sys
import numpy as np
import ai_edge_litert.interpreter as tflite
import yaml
import time
import psutil

# ==========================================
# 1. 路径修复：确保 Python 能找到 src 包
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 从你确认存在的 pipeline 脚本中导入预处理函数
from src.preprocess.preprocess_pipeline import preprocess_wav_for_pi

def softmax(x):
    """将神经网络原始输出(Logits)转换为概率(0-1)"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# 2. 配置与文件路径
# ==========================================
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
QUALITY_MODEL_PATH = os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
DIAG_MODEL_PATH = os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")

# 使用你刚才 ls 查出的真实存在的文件名
TEST_WAV = os.path.join(PROJECT_ROOT, "/home/rasp4b/FypPi/data/raw/Dataset2/training-a/a0001.wav")

def main():
    print("🚀 FypProj 双级推理系统 · 最终调试版")
    print("="*60)

    # ========= 性能监控开始 =========
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_mem = process.memory_info().rss / 1024**2  # MB

    # A. 文件完整性检查
    for p in [CONFIG_PATH, QUALITY_MODEL_PATH, DIAG_MODEL_PATH, TEST_WAV]:
        if not os.path.exists(p):
            print(f"❌ 错误: 找不到关键文件 {p}")
            return

    # B. 环境初始化
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # 初始化 TFLite 解释器
    q_interpreter = tflite.Interpreter(model_path=QUALITY_MODEL_PATH)
    d_interpreter = tflite.Interpreter(model_path=DIAG_MODEL_PATH)
    q_interpreter.allocate_tensors()
    d_interpreter.allocate_tensors()

    q_in_idx = q_interpreter.get_input_details()[0]['index']
    q_out_idx = q_interpreter.get_output_details()[0]['index']
    d_in_idx = d_interpreter.get_input_details()[0]['index']
    d_out_idx = d_interpreter.get_output_details()[0]['index']

    # C. 执行预处理 (滤波 -> 切片 -> Mel 转换)
    print(f"🎬 正在读取音频并提取特征: {os.path.basename(TEST_WAV)}")
    # 调用你的 pipeline 进行特征工程
    tensors = preprocess_wav_for_pi(TEST_WAV, config)
    print(f"📦 预处理成功: 已生成 {len(tensors)} 个 2 秒切片")
    print("-" * 60)

    # D. 级联推理循环
    for i, input_tensor in enumerate(tensors):
        # --- 第一级：质量评估 (SQA) ---
        q_interpreter.set_tensor(q_in_idx, input_tensor)
        q_interpreter.invoke()
        q_logits = q_interpreter.get_tensor(q_out_idx)[0]
        q_probs = softmax(q_logits)
        q_pred = np.argmax(q_probs)

        if q_pred == 0:  # 0 代表 Poor Quality
            print(f"片段 {i+1:02d}: ⚠️  [质量拦截] 信号干扰过强 | 噪声概率: {q_probs[0]:.2%}")
            continue
        
        # --- 第二级：疾病诊断 ---
        d_interpreter.set_tensor(d_in_idx, input_tensor)
        d_interpreter.invoke()
        d_logits = d_interpreter.get_tensor(d_out_idx)[0]
        
        # 核心修正：应用 Softmax 得到 0-1 的置信度
        d_probs = softmax(d_logits)
        d_pred = np.argmax(d_probs)
        
        label = "Normal (正常)" if d_pred == 0 else "Abnormal (异常)"
        confidence = d_probs[d_pred]
        
        print(f"片段 {i+1:02d}: ✨ [诊断通过] 结果: {label} | 置信度: {confidence:.2%}")

    print("="*60)

    # ========= 性能统计 =========
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024**2  # MB
    peak_mem = process.memory_info().rss / 1024**2

    total_time = end_time - start_time

    print("📊 性能统计:")
    print(f"⏱ 总运行时间: {total_time:.2f} 秒")
    print(f"💾 起始内存: {start_mem:.2f} MB")
    print(f"💾 结束内存: {end_mem:.2f} MB")
    print(f"📈 当前内存占用: {peak_mem:.2f} MB")

    print("✅ 离线验证任务圆满完成。")

if __name__ == "__main__":
    main()