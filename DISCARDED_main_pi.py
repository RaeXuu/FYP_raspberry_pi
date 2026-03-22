import asyncio
import collections
import os
import signal
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

from src.preprocess.filters import apply_bandpass
from src.preprocess.mel import logmel_fixed_size

# ==========================================
# 配置
# ==========================================
ESP32_MAC           = "80:F1:B2:ED:B4:12"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

SAMPLE_RATE   = 2000
SEG_DURATION  = 2.0   # 窗口长度（秒）
OVERLAP       = 0.5   # 与训练对齐
SQA_THRESHOLD = 0.6

SEG_SAMPLES = int(SAMPLE_RATE * SEG_DURATION)        # 4000 samples
HOP_SAMPLES = int(SEG_SAMPLES * (1 - OVERLAP))       # 2000 samples = 1s
HOP_BYTES   = HOP_SAMPLES * 2                        # 4000 bytes (int16)

# ==========================================
# 全局流式状态
# ==========================================
_ring        = collections.deque(maxlen=SEG_SAMPLES)  # 环形缓冲，保留最新 2s
_hop_counter = 0                                       # 距上次触发的新字节数
_hop_event   = None                                    # asyncio.Event，在 main() 中初始化
_running     = True


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def notification_handler(sender, data):
    global _hop_counter
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    _ring.extend(samples)
    _hop_counter += len(data)
    if _hop_counter >= HOP_BYTES:
        _hop_counter -= HOP_BYTES
        if _hop_event is not None:
            _hop_event.set()


async def main():
    global _hop_event, _running

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    mel_cfg = config["mel"]

    # 加载模型
    q_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite"))
    d_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_model_quant.tflite"))
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()

    q_in  = q_interp.get_input_details()[0]['index']
    q_out = q_interp.get_output_details()[0]['index']
    d_in  = d_interp.get_input_details()[0]['index']
    d_out = d_interp.get_output_details()[0]['index']

    _hop_event = asyncio.Event()

    # Ctrl+C 优雅退出
    loop = asyncio.get_event_loop()
    def handle_sigint():
        global _running
        _running = False
        _hop_event.set()   # 解除 wait 阻塞
    loop.add_signal_handler(signal.SIGINT, handle_sigint)

    valid_results = []   # list of (sqa_score, prob_normal)
    window_count  = 0

    def rolling_summary():
        if not valid_results:
            return None, None
        weights = [r[0] for r in valid_results]
        probs   = [r[1] for r in valid_results]
        avg = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
        return avg, "Normal" if avg > 0.5 else "Abnormal"

    print(f"正在连接 ESP32: {ESP32_MAC}...")

    async with BleakClient(ESP32_MAC) as client:
        await client._backend._acquire_mtu()
        print(f"已连接，MTU: {client.mtu_size} 字节")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print(f"流式推理启动（窗口 {SEG_DURATION}s / 滑步 {HOP_SAMPLES/SAMPLE_RATE}s）— Ctrl+C 停止\n")

        # 等待初始缓冲区填满（2s）
        while len(_ring) < SEG_SAMPLES and _running:
            await asyncio.sleep(0.05)

        while _running:
            await _hop_event.wait()
            _hop_event.clear()

            if not _running:
                break

            if len(_ring) < SEG_SAMPLES:
                continue

            window_count += 1
            window = np.array(_ring, dtype=np.float32)

            # 峰值归一化
            max_val = np.max(np.abs(window))
            if max_val > 0:
                window = window / max_val

            # 带通滤波
            filtered = apply_bandpass(window, fs=SAMPLE_RATE, lowcut=25, highcut=400)

            # Log-Mel → tensor (1, 1, 32, 64)
            mel    = logmel_fixed_size(y=filtered, sr=SAMPLE_RATE, mel_cfg=mel_cfg,
                                       target_shape=(mel_cfg["n_mels"], 64))
            tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

            # SQA
            q_interp.set_tensor(q_in, tensor)
            q_interp.invoke()
            q_probs   = softmax(q_interp.get_tensor(q_out)[0])
            sqa_score = float(q_probs[1])

            if sqa_score < SQA_THRESHOLD:
                print(f"[W{window_count:04d}] SQA={sqa_score:.2%}  低质量，跳过")
                continue

            # 诊断
            d_interp.set_tensor(d_in, tensor)
            d_interp.invoke()
            d_probs     = softmax(d_interp.get_tensor(d_out)[0])
            prob_normal = float(d_probs[0])
            diag_label  = "Normal" if prob_normal > 0.5 else "Abnormal"

            valid_results.append((sqa_score, prob_normal))
            avg_prob, avg_label = rolling_summary()
            confidence = avg_prob if avg_label == "Normal" else 1 - avg_prob

            print(f"[W{window_count:04d}] SQA={sqa_score:.2%} | {diag_label} ({prob_normal:.2%}) "
                  f"| 滚动: {avg_label} {confidence:.2%} [{len(valid_results)} 片段]")

        await client.stop_notify(CHARACTERISTIC_UUID)

    # 最终汇总
    print("\n" + "=" * 55)
    avg_prob, avg_label = rolling_summary()
    if avg_label is None:
        print("无有效片段，无法诊断。请检查听诊器位置后重试。")
    else:
        confidence = avg_prob if avg_label == "Normal" else 1 - avg_prob
        print(f"最终诊断: {avg_label}  置信度: {confidence:.2%}  ({len(valid_results)} 有效片段 / {window_count} 总窗口)")
    print("=" * 55)


if __name__ == "__main__":
    asyncio.run(main())
