import asyncio
import csv
import os
import signal
import sys
import time
import wave
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
from src.storage.summary import append_summary
from src.display.oled import OLEDDisplay
from src.ui.button import Button

oled = OLEDDisplay()

# ==========================================
# 配置
# ==========================================
ESP32_MAC           = "80:F1:B2:ED:B4:12"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

SAMPLE_RATE    = 2000
SEG_DURATION   = 2.0    # 滑动窗口长度（秒），与训练对齐
OVERLAP        = 0.5    # 50% overlap，与训练对齐
CHUNK_DURATION = 20     # 每块采集时长（秒）
SQA_THRESHOLD  = 0.6
DIAG_THRESHOLD = 0.5   # prob_normal 高于此值判为 Normal

SEG_SAMPLES   = int(SAMPLE_RATE * SEG_DURATION)      # 4000 samples
HOP_SAMPLES   = int(SEG_SAMPLES * (1 - OVERLAP))     # 2000 samples（1s）
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION         # 40000 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                    # 80000 bytes（int16）
# 每块窗口数：(40000 - 4000) / 2000 + 1 

RECORDS_DIR = os.path.join(PROJECT_ROOT, "debug_records")
LOG_PATH    = os.path.join(RECORDS_DIR, "inference_log.csv")

# ==========================================
# 全局接收状态
# ==========================================
_recv_buf    = bytearray()
_chunk_queue = None   # asyncio.Queue，在 run_session() 中初始化
_running     = True
_chunk_count = 0
_exit        = False


# ==========================================
# 工具函数
# ==========================================
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def save_wav(raw_bytes, chunk_idx, prefix):
    os.makedirs(RECORDS_DIR, exist_ok=True)
    filename = os.path.join(RECORDS_DIR,
                            f"{prefix}_c{chunk_idx:04d}_{int(time.time())}.wav")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)        # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    return filename


# ==========================================
# BLE 回调（事件循环线程）
# ==========================================
def notification_handler(sender, data):
    global _recv_buf, _chunk_count
    _recv_buf.extend(data)
    if len(_recv_buf) >= CHUNK_BYTES:
        chunk = bytes(_recv_buf[:CHUNK_BYTES])
        _recv_buf = bytearray(_recv_buf[CHUNK_BYTES:])  # 保留余量给下一块
        _chunk_count += 1
        if _chunk_queue is not None:
            if _chunk_queue.full():
                # 推理跟不上，丢弃积压的旧块，保留最新数据
                try:
                    dropped_idx, _ = _chunk_queue.get_nowait()
                    print(f"[警告] 推理积压，丢弃块 {dropped_idx:03d}，保留块 {_chunk_count:03d}")
                except asyncio.QueueEmpty:
                    pass
            _chunk_queue.put_nowait((_chunk_count, chunk))


# ==========================================
# 推理函数（在 to_thread 中运行，不阻塞事件循环）
# ==========================================
def run_inference(raw_bytes, mel_cfg, q_interp, d_interp,
                  q_in, q_out, d_in, d_out, on_window=None, chunk_idx=0, last_label=None):
    """
    对一块原始 int16 字节做滑动窗口推理。
    返回: (label, avg_prob, valid_count, total_windows, window_data)
           label 为 None 表示全部低质量（信号差）
           window_data: [(win_idx, sqa_score, sqa_pass, prob_normal or None), ...]
    """
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # 带通滤波（整块做一次，各窗口共用）
    filtered = apply_bandpass(audio, fs=SAMPLE_RATE, lowcut=25, highcut=400)

    valid_results  = []   # [(sqa_score, prob_normal), ...]
    window_data    = []   # [(win_idx, sqa_score, passed, prob_normal or None), ...]
    total_windows  = int((CHUNK_SAMPLES - SEG_SAMPLES) / HOP_SAMPLES) + 1
    sqa_line       = []

    for win_idx, start in enumerate(range(0, CHUNK_SAMPLES - SEG_SAMPLES + 1, HOP_SAMPLES)):
        window = filtered[start : start + SEG_SAMPLES]

        # 峰值归一化（per-window，防止单个噪声峰值压制整块）
        max_val = np.max(np.abs(window))
        if max_val > 0:
            window = window / max_val

        mel    = logmel_fixed_size(y=window, sr=SAMPLE_RATE, mel_cfg=mel_cfg,
                                   target_shape=(mel_cfg["n_mels"], 64))
        tensor = mel[np.newaxis, np.newaxis, ...].astype(np.float32)

        # SQA
        q_interp.set_tensor(q_in, tensor)
        q_interp.invoke()
        q_probs   = softmax(q_interp.get_tensor(q_out)[0])
        sqa_score = float(q_probs[1])

        passed = sqa_score >= SQA_THRESHOLD
        sqa_line.append(f"w{win_idx+1:02d}:{sqa_score:.2f}{'✓' if passed else '✗'}")
        # 每8个窗口打印一行进度
        if (win_idx + 1) % 8 == 0 or (win_idx + 1) == total_windows:
            print("  " + "  ".join(sqa_line), flush=True)
            sqa_line = []

        if not passed:
            window_data.append((win_idx + 1, sqa_score, False, None))
            continue

        # 诊断
        d_interp.set_tensor(d_in, tensor)
        d_interp.invoke()
        d_probs     = softmax(d_interp.get_tensor(d_out)[0])
        prob_normal = float(d_probs[0])

        valid_results.append((sqa_score, prob_normal))
        window_data.append((win_idx + 1, sqa_score, True, prob_normal))

        if on_window is not None and valid_results:
            weights = [r[0] for r in valid_results]
            probs   = [r[1] for r in valid_results]
            avg     = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
            on_window(avg * 100, (1 - avg) * 100, chunk_idx, last_label)

    if not valid_results:
        return None, None, 0, total_windows, window_data

    weights  = [r[0] for r in valid_results]
    probs    = [r[1] for r in valid_results]
    avg_prob = sum(w * p for w, p in zip(weights, probs)) / sum(weights)
    label    = "Normal" if avg_prob > DIAG_THRESHOLD else "Abnormal"

    return label, avg_prob, len(valid_results), total_windows, window_data


# ==========================================
# 看门狗心跳（每 30s 写时间戳到 /tmp/heartbeat.ts）
# ==========================================
HEARTBEAT_PATH     = "/tmp/heartbeat.ts"
HEARTBEAT_INTERVAL = 30

async def heartbeat_writer():
    while not _exit:
        with open(HEARTBEAT_PATH, "w") as f:
            f.write(str(time.time()))
        await asyncio.sleep(HEARTBEAT_INTERVAL)


# ==========================================
# 推理 worker（消费队列，一次处理一块）
# ==========================================
async def inference_worker(mel_cfg, q_interp, d_interp,
                            q_in, q_out, d_in, d_out):
    tally = {"Normal": 0, "Abnormal": 0, "noise": 0, "_last_label": None}
    log_file = None
    writer = None

    while True:
        # 有块就处理，没块就等 1s；_running=False 且队列空时退出
        try:
            chunk_idx, raw_bytes = await asyncio.wait_for(
                _chunk_queue.get(), timeout=1.0
            )
        except asyncio.TimeoutError:
            if not _running:
                break
            continue

        # 第一次收到块时再初始化 CSV（避免启动时阻塞 event loop）
        if log_file is None:
            os.makedirs(RECORDS_DIR, exist_ok=True)
            write_header = not os.path.exists(LOG_PATH)
            log_file = open(LOG_PATH, "a", newline="")
            writer = csv.writer(log_file)
            if write_header:
                writer.writerow(["time", "chunk_idx", "win_idx",
                                 "sqa_score", "sqa_pass", "prob_normal",
                                 "chunk_label", "chunk_prob_normal"])

        print(f"\n[块 {chunk_idx:03d}] 推理中（{CHUNK_DURATION}s / {CHUNK_SAMPLES//HOP_SAMPLES - 1} 窗口）...")
        oled.show_running(chunk_idx=chunk_idx, last_label=tally.get("_last_label"))

        last_label_upper = tally.get("_last_label")
        label, avg_prob, valid_count, total_windows, window_data = await asyncio.to_thread(
            run_inference, raw_bytes, mel_cfg,
            q_interp, d_interp, q_in, q_out, d_in, d_out,
            oled.show_running, chunk_idx, last_label_upper
        )

        chunk_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if label is not None:
            tally["_last_label"] = label.upper()

        if label is None:
            tally["noise"] += 1
            filename = save_wav(raw_bytes, chunk_idx, "noise")
            print(f"[块 {chunk_idx:03d}]  信号差（0/{total_windows} 窗口通过 SQA）")
            print(f"[块 {chunk_idx:03d}]  已保存: {filename}")
        elif label == "Normal":
            tally["Normal"] += 1
            filename = save_wav(raw_bytes, chunk_idx, "normal")
            print(f"[块 {chunk_idx:03d}]  Normal     prob_normal={avg_prob:.2%}"
                  f"  ({valid_count}/{total_windows} 窗口有效）")
            print(f"[块 {chunk_idx:03d}]  已保存: {filename}")
        else:
            tally["Abnormal"] += 1
            filename = save_wav(raw_bytes, chunk_idx, "abnormal")
            print(f"[块 {chunk_idx:03d}]  Abnormal   prob_normal={avg_prob:.2%}"
                  f"  ({valid_count}/{total_windows} 窗口有效）")
            print(f"[块 {chunk_idx:03d}]  已保存: {filename}")

        # 写每个窗口的数据到 CSV
        for win_idx, sqa_score, sqa_pass, prob_normal in window_data:
            writer.writerow([
                chunk_time, chunk_idx, win_idx,
                f"{sqa_score:.4f}", sqa_pass,
                f"{prob_normal:.4f}" if prob_normal is not None else "",
                label if label is not None else "noise",
                f"{avg_prob:.4f}" if avg_prob is not None else ""
            ])
        log_file.flush()
        append_summary(label, avg_prob, valid_count, total_windows)

        _chunk_queue.task_done()

    log_file.close()

    total = tally["Normal"] + tally["Abnormal"] + tally["noise"]
    print("\n" + "=" * 50)
    if total == 0:
        print("未采集到任何数据。")
    else:
        print(f"最终统计: 共 {total} 块")
        print(f"  Normal   {tally['Normal']} 块")
        print(f"  Abnormal {tally['Abnormal']} 块")
        print(f"  低质量    {tally['noise']} 块（全部窗口未通过 SQA）")
    print("=" * 50)


# ==========================================
# 单次采集会话（按键启动/停止）
# ==========================================
async def run_session(mel_cfg, q_interp, d_interp, q_in, q_out, d_in, d_out):
    global _chunk_queue, _running, _recv_buf, _chunk_count

    _running = True
    _chunk_queue = asyncio.Queue(maxsize=1)
    _recv_buf.clear()
    _chunk_count = 0

    print(f"正在连接 ESP32: {ESP32_MAC}...")
    oled.show_connecting(0.0)

    client = BleakClient(ESP32_MAC)
    try:
        await asyncio.wait_for(client.connect(), timeout=15.0)
    except Exception as e:
        print(f"连接失败: {e}")
        oled.show_error("Connect Failed")
        await asyncio.sleep(3)
        return

    try:
        await client._backend._acquire_mtu()
        print(f"已连接，MTU: {client.mtu_size} 字节")
        oled.show_running(chunk_idx=0)
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        await asyncio.sleep(2)   # 等 ESP32 把积压缓冲排空
        _recv_buf.clear()
        _chunk_count = 0
        print(f"流式采集启动（每 {CHUNK_DURATION}s 一块，滑窗 {SEG_DURATION}s / hop {HOP_SAMPLES/SAMPLE_RATE}s）")
        print("短按按键停止\n")

        worker = asyncio.create_task(
            inference_worker(mel_cfg, q_interp, d_interp,
                             q_in, q_out, d_in, d_out)
        )

        while _running:
            await asyncio.sleep(0.1)

        await client.stop_notify(CHARACTERISTIC_UUID)
        await worker
    finally:
        await client.disconnect()


# ==========================================
# 主程序
# ==========================================
async def main():
    global _running, _exit

    oled.show_standby()

    with open(os.path.join(PROJECT_ROOT, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    mel_cfg = config["mel"]

    q_interp = tflite.Interpreter(
        model_path=os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite"))
    d_interp = tflite.Interpreter(
        model_path=os.path.join(PROJECT_ROOT, "heart_model_quant.tflite"))
    q_interp.allocate_tensors()
    d_interp.allocate_tensors()

    q_in  = q_interp.get_input_details()[0]['index']
    q_out = q_interp.get_output_details()[0]['index']
    d_in  = d_interp.get_input_details()[0]['index']
    d_out = d_interp.get_output_details()[0]['index']

    session_event = asyncio.Event()

    async def on_short_press():
        global _running
        if not _running:
            session_event.set()
        else:
            _running = False

    async def on_long_press():
        global _running, _exit
        _running = False
        _exit = True
        oled.show_text("关机中...")
        await asyncio.sleep(1)
        os.system("sudo shutdown -h now")

    btn = Button()
    btn.on_short_press(on_short_press)
    btn.on_long_press(on_long_press)
    btn.start()

    loop = asyncio.get_event_loop()
    def handle_signal():
        global _running, _exit
        _running = False
        _exit = True
        session_event.set()
        print("\n退出信号 — 等待当前块推理完成后退出...")
    loop.add_signal_handler(signal.SIGINT, handle_signal)
    loop.add_signal_handler(signal.SIGTERM, handle_signal)

    hb = asyncio.create_task(heartbeat_writer())

    oled.show_standby()
    print("待机中，短按按键开始采集，长按 3s 关机")

    while not _exit:
        await session_event.wait()
        session_event.clear()
        if _exit:
            break
        await run_session(mel_cfg, q_interp, d_interp, q_in, q_out, d_in, d_out)
        if not _exit:
            oled.show_standby()
            print("采集结束，短按按键重新开始")

    hb.cancel()
    btn.stop()


if __name__ == "__main__":
    asyncio.run(main())
