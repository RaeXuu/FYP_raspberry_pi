"""
record_debug.py — 录音 + 管道诊断工具

功能：
  1. 通过 BLE 从 ESP32 录制指定时长的音频（ESP32 已在硬件侧输出 2000Hz）
  2. 保存 2000Hz WAV
  3. 直接对音频跑 SQA + 诊断，定位管道哪一环出问题

用法：
  python record_debug.py           # 默认录 6 秒
  python record_debug.py 10        # 录 10 秒
"""

import asyncio
import os
import sys
import wave
import numpy as np
import ai_edge_litert.interpreter as tflite
import yaml
from bleak import BleakClient

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.preprocess_pipeline import preprocess_array_for_pi

RECORD_DIR = os.path.join(PROJECT_ROOT, "WAV_record")


def next_index():
    os.makedirs(RECORD_DIR, exist_ok=True)
    existing = [f for f in os.listdir(RECORD_DIR) if f.endswith(".wav")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(f.split("_")[0]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


# ==========================================
# 配置
# ==========================================
ESP32_MAC           = "80:F1:B2:ED:B4:12"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
SAMPLE_RATE         = 2000  # ESP32 已在硬件侧降采样至 2000Hz
RECORD_DURATION     = int(sys.argv[1]) if len(sys.argv) > 1 else 6  # 秒
SQA_THRESHOLD       = 0.05  # 与 main_pi.py 对齐

RECORD_BYTES = SAMPLE_RATE * 2 * RECORD_DURATION

_buffer        = bytearray()
_collecting    = False
_record_ready  = asyncio.Event()


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


def notification_handler(sender, data):
    global _buffer, _collecting
    if _collecting:
        _buffer.extend(data)
        if len(_buffer) >= RECORD_BYTES:
            _record_ready.set()


def save_wav(filepath, samples_int16, sample_rate):
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    print(f"   💾 已保存: {filepath}")


async def main():
    idx = next_index()
    out_2k = os.path.join(RECORD_DIR, f"{idx:03d}_2k.wav")

    # 加载配置和模型
    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

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

    print(f"📡 正在连接 ESP32: {ESP32_MAC}...")

    try:
        async with BleakClient(ESP32_MAC) as client:
            await client._backend._acquire_mtu()
            print(f"✅ 已连接，MTU: {client.mtu_size} 字节")
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

            global _buffer, _collecting
            _buffer    = bytearray()
            _record_ready.clear()
            _collecting = True

            print(f"🎙️  录制中（{RECORD_DURATION}s）...")
            await asyncio.wait_for(_record_ready.wait(), timeout=RECORD_DURATION * 3)
            _collecting = False

            await client.stop_notify(CHARACTERISTIC_UUID)

        raw_bytes = bytes(_buffer[:RECORD_BYTES])

    except EOFError:
        raw_bytes = bytes(_buffer[:RECORD_BYTES])
    except Exception as e:
        print(f"❌ 连接错误: {e!r}")
        return

    if len(raw_bytes) < RECORD_BYTES:
        print(f"⚠️  数据不足: 收到 {len(raw_bytes)} 字节，需要 {RECORD_BYTES} 字节")
        return

    # ==========================================
    # 保存 2000Hz WAV
    # ==========================================
    audio_int16 = np.frombuffer(raw_bytes, dtype=np.int16)

    audio_float_save = audio_int16.astype(np.float32)
    max_val = np.max(np.abs(audio_float_save))
    if max_val > 0:
        audio_float_save = audio_float_save / max_val * 32767.0
    save_wav(out_2k, audio_float_save.astype(np.int16), SAMPLE_RATE)

    # ==========================================
    # 按 2s 切片跑 SQA + 诊断
    # ==========================================
    audio_float = audio_int16.astype(np.float32) / 32768.0
    SEGMENT_SAMPLES = SAMPLE_RATE * 2  # 2s @ 2000Hz = 4000 samples
    n_segments = len(audio_float) // SEGMENT_SAMPLES
    print(f"\n📊 共 {n_segments} 个 2s 片段，逐一诊断（SQA 阈值={SQA_THRESHOLD}）：")
    print("-" * 50)

    results = []
    for i in range(n_segments):
        seg = audio_float[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
        tensors = preprocess_array_for_pi(seg, config)

        for j, tensor in enumerate(tensors):
            # SQA
            t_q = quantize_input(tensor, *q_qi)
            q_interp.set_tensor(q_in, t_q)
            q_interp.invoke()
            q_probs = softmax(dequantize_output(
                q_interp.get_tensor(q_out), *q_qo)[0])
            sqa_score = float(q_probs[0])  # Good 概率，与 main_pi.py 对齐
            print(f"片段 {i+1:02d}-{j+1}: SQA → Good={sqa_score:.2%} | Bad={q_probs[1]:.2%}", end="")

            if sqa_score < SQA_THRESHOLD:
                print("  ⚠️  跳过")
                continue

            # 诊断
            t_d = quantize_input(tensor, *d_qi)
            d_interp.set_tensor(d_in, t_d)
            d_interp.invoke()
            d_probs = softmax(dequantize_output(
                d_interp.get_tensor(d_out), *d_qo)[0])
            prob_normal = float(d_probs[0])
            print(f"  → {'Normal' if prob_normal > 0.5 else 'Abnormal'} ({prob_normal:.2%})")
            results.append((sqa_score, prob_normal))

    print("\n" + "=" * 50)
    if not results:
        print("⚠️  所有片段 SQA 不足。")
        print(f"\n建议：把 {os.path.basename(out_2k)} 用音频软件打开，听一下信号质量是否正常。")
    else:
        weights           = [r[0] for r in results]
        probs             = [r[1] for r in results]
        final_prob_normal = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
        label      = "Normal" if final_prob_normal > 0.5 else "Abnormal"
        confidence = final_prob_normal if label == "Normal" else 1 - final_prob_normal
        print(f"✨ 最终诊断: {label} | 置信度: {confidence:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
