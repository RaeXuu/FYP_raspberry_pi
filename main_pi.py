import asyncio
import wave
import os
import sys
import time
import numpy as np
import ai_edge_litert.interpreter as tflite
import yaml
from bleak import BleakClient

# ==========================================
# 环境初始化
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.preprocess_pipeline import preprocess_array_for_pi

# ==========================================
# 配置
# ==========================================
ESP32_MAC           = "80:F1:B2:ED:B4:12"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

SAMPLE_RATE         = 8000
SEGMENT_DURATION    = 2     # 每次采集秒数
COLLECTION_INTERVAL = 30    # 两次采集之间的间隔（秒）
NUM_COLLECTIONS     = 3     # 总采集次数
SQA_THRESHOLD       = 0.5   # SQA 低于此值丢弃片段

SEGMENT_BYTES = SAMPLE_RATE * 2 * SEGMENT_DURATION  # 每段字节数 = 8000

SD_SAVE_DIR = os.path.join(PROJECT_ROOT, "abnormal_records")  # 异常音频保存目录

# ==========================================
# 全局采集状态
# ==========================================
_buffer     = bytearray()
_collecting = False


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def notification_handler(sender, data):
    global _buffer, _collecting
    if _collecting:
        _buffer.extend(data)


def save_abnormal_wav(raw_segments):
    """将所有采集片段拼接后保存为 WAV 文件"""
    os.makedirs(SD_SAVE_DIR, exist_ok=True)
    filename = os.path.join(SD_SAVE_DIR, f"abnormal_{int(time.time())}.wav")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(raw_segments))
    print(f"💾 异常音频已保存: {filename}")


async def collect_segment():
    """采集 SEGMENT_DURATION 秒的音频，返回原始字节"""
    global _buffer, _collecting
    _buffer     = bytearray()
    _collecting = True
    while len(_buffer) < SEGMENT_BYTES:
        await asyncio.sleep(0.05)
    _collecting = False
    return bytes(_buffer[:SEGMENT_BYTES])


async def main():
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

    print(f"📡 正在连接 ESP32: {ESP32_MAC}...")

    try:
        async with BleakClient(ESP32_MAC) as client:
            await client._backend._acquire_mtu()
            print(f"✅ 已连接，MTU: {client.mtu_size} 字节")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

            results      = []  # 有效片段：(sqa_score, prob_normal)
            raw_segments = []  # 所有原始字节（用于保存）

            for i in range(NUM_COLLECTIONS):
                if i > 0:
                    print(f"⏳ 等待 {COLLECTION_INTERVAL} 秒后进行第 {i+1} 次采集...")
                    await asyncio.sleep(COLLECTION_INTERVAL)

                print(f"🎙️  第 {i+1}/{NUM_COLLECTIONS} 次采集中（{SEGMENT_DURATION}s）...")
                raw = await collect_segment()
                raw_segments.append(raw)

                # 转换为 float32 numpy array，并降采样 8000Hz → 2000Hz
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio = audio[::4]     //更改的地方

                # 预处理 → mel tensor (1, 1, 32, 64)
                tensor = preprocess_array_for_pi(audio, config)

                # 第一级：SQA 质量评估
                q_interp.set_tensor(q_in, tensor)
                q_interp.invoke()
                q_probs   = softmax(q_interp.get_tensor(q_out)[0])
                sqa_score = q_probs[1]  # index 1 = Good Quality
                print(f"   SQA → Poor={q_probs[0]:.2%} | Good={sqa_score:.2%}")

                if sqa_score < SQA_THRESHOLD:
                    print(f"   ⚠️  质量不足，跳过此片段")
                    continue

                # 第二级：诊断模型
                d_interp.set_tensor(d_in, tensor)
                d_interp.invoke()
                d_probs     = softmax(d_interp.get_tensor(d_out)[0])
                prob_normal = d_probs[0]  # index 0 = Normal
                diag_label  = "Normal" if prob_normal > 0.5 else "Abnormal"
                print(f"   诊断 → {diag_label} | Normal={prob_normal:.2%}")

                results.append((sqa_score, prob_normal))

            await client.stop_notify(CHARACTERISTIC_UUID)

        # ==========================================
        # 汇总：SQA 加权平均
        # ==========================================
        print("\n" + "=" * 50)

        if not results:
            print("⚠️  所有片段质量不足，无法诊断。")
            print("请重新放置听诊器后重试。")
            return

        weights           = [r[0] for r in results]
        probs             = [r[1] for r in results]
        final_prob_normal = sum(w * p for w, p in zip(weights, probs)) / sum(weights)

        label      = "Normal" if final_prob_normal > 0.5 else "Abnormal"
        confidence = final_prob_normal if label == "Normal" else 1 - final_prob_normal

        print(f"✨ 最终诊断: {label} | 置信度: {confidence:.2%}")
        print("=" * 50)

        if label == "Abnormal":
            save_abnormal_wav(raw_segments)

    except Exception as e:
        print(f"❌ 错误: {e!r}")


if __name__ == "__main__":
    asyncio.run(main())
