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
import psutil
from src.display.oled import OLEDDisplay, SysInfoDisplay
from src.ui.button import Button
from src.ui.led import RGBLed
from src.ui.buzzer import Buzzer
from src.power.serial_reader import PowerReader

oled   = OLEDDisplay()
oled2  = SysInfoDisplay()
power  = PowerReader()
led    = RGBLed()
buzzer = Buzzer()

# ==========================================
# 配置
# ==========================================
# ESP32_MAC           = "80:F1:B2:ED:B4:12"
ESP32_MAC           = "AC:A7:04:85:0D:42"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
BLE_CONNECT_TIMEOUT = 15

SAMPLE_RATE    = 2000
SEG_DURATION   = 2.0    # 滑动窗口长度（秒），与训练对齐
OVERLAP        = 0.5    # 50% overlap，与训练对齐
CHUNK_DURATION = 20     # 每块采集时长（秒）
SQA_THRESHOLD  = 0.65
DIAG_THRESHOLD = 0.5   # prob_normal 高于此值判为 Normal

SEG_SAMPLES   = int(SAMPLE_RATE * SEG_DURATION)      # 4000 samples
HOP_SAMPLES   = int(SEG_SAMPLES * (1 - OVERLAP))     # 2000 samples（1s）
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_DURATION         # 40000 samples
CHUNK_BYTES   = CHUNK_SAMPLES * 2                    # 80000 bytes（int16）
# 每块窗口数：(40000 - 4000) / 2000 + 1 

RECORDS_DIR = "/data/debug_records"
LOG_PATH    = "/data/debug_records/inference_log.csv"

# ==========================================
# 全局接收状态
# ==========================================
_recv_buf    = bytearray()
_chunk_queue = None   # asyncio.Queue，在 run_session() 中初始化
_running     = True
_chunk_count = 0
_exit        = False
_oled2_page  = 0      # 0=系统信息, 1=电源信息

# 电池保护阈值（Li-ion，BATTERY 电压）
BAT_WARN_V     = 3.4  # 低电量警告
BAT_SHUTDOWN_V = 3.2  # 软件安全关机（硬件断电默认 3.0V）


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
                  q_in, q_out, d_in, d_out, on_window=None, chunk_idx=0,
                  last_label=None, last_chunk_idx=None, last_prob=None):
    os.sched_setaffinity(0, {3})  # 推理线程固定到 core 3
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
        sqa_score = float(q_probs[0])

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
            on_window(avg * 100, chunk_idx, last_label,
                      last_chunk_idx, last_prob,
                      win_idx + 1, total_windows,
                      win_idx % 2 == 0)

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


async def sysinfo_updater():
    global _oled2_page, _running, _exit
    wifi_blink = True
    tick = 0
    cpu, mem_used, mem_total, temp = 0.0, 0.0, 0.0, 0.0
    bat_shutdown_triggered = False
    bat_warn = False
    lightning_on = False

    while not _exit:
        # 每 2 tick（2s）更新系统信息，每 1 tick（1s）刷新 WiFi 闪烁
        if tick % 2 == 0:
            cpu  = psutil.cpu_percent(interval=None)
            mem  = psutil.virtual_memory()
            try:
                temps = psutil.sensors_temperatures()
                temp  = temps["cpu_thermal"][0].current
            except Exception:
                temp  = 0.0
            mem_used  = mem.used  / 1024**2
            mem_total = mem.total / 1024**2
        tick += 1

        wifi_addrs     = psutil.net_if_addrs().get('wlan0', [])
        wifi_connected = any(a.family == 2 for a in wifi_addrs)
        if wifi_connected:
            wifi_on = True
        else:
            wifi_blink = not wifi_blink
            wifi_on    = wifi_blink

        # ---- 电池保护 ----
        bat_v    = power.battery_v()
        charging = (power.dcinput_v() or 0) > 4.0  # DC 接入视为充电中
        if bat_v is not None and not charging and not bat_shutdown_triggered:
            if bat_v <= BAT_SHUTDOWN_V:
                bat_shutdown_triggered = True
                print(f"[Battery] 电压 {bat_v:.2f}V ≤ {BAT_SHUTDOWN_V}V，执行安全关机")
                _running = False
                _exit    = True
                led.off()
                oled.show_text("Low Battery!\n关机中...")
                await asyncio.sleep(3)
                os.system("sudo shutdown -h now")
                return
            elif bat_v <= BAT_WARN_V:
                bat_warn = True
                _oled2_page = 1      # 强制切到电源页
                lightning_on = not lightning_on  # 每 tick 翻转，产生闪烁
            else:
                bat_warn = False
                lightning_on = False

        if _oled2_page == 0:
            oled2.show(cpu, mem_used, mem_total, temp, wifi_on=wifi_on)
        else:
            oled2.show_power(
                bat_pct=power.bat_pct(),
                voltage_v=power.battery_v(),
                uptime_str=power.uptime(),
                wifi_on=wifi_on,
                low_bat=lightning_on,
            )
        await asyncio.sleep(1)


# ==========================================
# 推理 worker（消费队列，一次处理一块）
# ==========================================
async def inference_worker(mel_cfg, q_interp, d_interp,
                            q_in, q_out, d_in, d_out):
    tally = {"Normal": 0, "Abnormal": 0, "noise": 0,
             "_last_label": None, "_last_chunk_idx": None, "_last_prob": None}
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
                                 "chunk_label", "chunk_prob_normal", "infer_ms"])

        print(f"\n[块 {chunk_idx:03d}] 推理中（{CHUNK_DURATION}s / {CHUNK_SAMPLES//HOP_SAMPLES - 1} 窗口）...")
        led.running()
        oled.show_running(chunk_idx=chunk_idx,
                          last_label=tally.get("_last_label"),
                          last_chunk_idx=tally.get("_last_chunk_idx"),
                          last_prob=tally.get("_last_prob"))

        infer_start = time.time()
        label, avg_prob, valid_count, total_windows, window_data = await asyncio.to_thread(
            run_inference, raw_bytes, mel_cfg,
            q_interp, d_interp, q_in, q_out, d_in, d_out,
            oled.show_running, chunk_idx,
            tally.get("_last_label"), tally.get("_last_chunk_idx"), tally.get("_last_prob")
        )
        infer_ms = (time.time() - infer_start) * 1000

        chunk_time = time.strftime("%Y-%m-%d %H:%M:%S")

        if label is not None:
            tally["_last_label"]     = label.upper()
            tally["_last_chunk_idx"] = chunk_idx
            tally["_last_prob"]      = avg_prob

        print(f"[块 {chunk_idx:03d}]  推理耗时 {infer_ms:.0f} ms（{valid_count}/{total_windows} 窗口有效）")

        if label is None:
            tally["noise"] += 1
            filename = save_wav(raw_bytes, chunk_idx, "noise")
            print(f"[块 {chunk_idx:03d}]  信号差（0/{total_windows} 窗口通过 SQA）")
            print(f"[块 {chunk_idx:03d}]  已保存: {filename}")
            # noise 块保持上一次 LED 状态，不更新
        elif label == "Normal":
            tally["Normal"] += 1
            led.normal()
            buzzer.normal()
            filename = save_wav(raw_bytes, chunk_idx, "normal")
            print(f"[块 {chunk_idx:03d}]  Normal     prob_normal={avg_prob:.2%}"
                  f"  ({valid_count}/{total_windows} 窗口有效）")
            print(f"[块 {chunk_idx:03d}]  已保存: {filename}")
        else:
            tally["Abnormal"] += 1
            led.abnormal()
            buzzer.abnormal()
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
                f"{avg_prob:.4f}" if avg_prob is not None else "",
                f"{infer_ms:.0f}"
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
    led.connecting()
    oled.start_connecting_countdown(timeout=BLE_CONNECT_TIMEOUT)

    client = BleakClient(ESP32_MAC)
    try:
        await asyncio.wait_for(client.connect(), timeout=BLE_CONNECT_TIMEOUT)
    except Exception as e:
        oled.stop_connecting_countdown()
        print(f"连接失败: {e}")
        led.error()
        oled.show_error("Connect Failed")
        await asyncio.sleep(3)
        return
    oled.stop_connecting_countdown()

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

    oled.start_standby_blink()

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
        led.off()
        oled.show_text("关机中...")
        await asyncio.sleep(1)
        os.system("sudo shutdown -h now")

    async def on_page_toggle():
        global _oled2_page
        _oled2_page = 1 - _oled2_page

    btn  = Button(pin=4)
    btn.on_short_press(on_short_press)
    btn.on_long_press(on_long_press)
    btn.start()

    btn2 = Button(pin=18)
    btn2.on_short_press(on_page_toggle)
    btn2.start()

    loop = asyncio.get_event_loop()
    def handle_signal():
        global _running, _exit
        _running = False
        _exit = True
        session_event.set()
        print("\n退出信号 — 等待当前块推理完成后退出...")
    loop.add_signal_handler(signal.SIGINT, handle_signal)
    loop.add_signal_handler(signal.SIGTERM, handle_signal)

    power.start()
    hb   = asyncio.create_task(heartbeat_writer())
    si   = asyncio.create_task(sysinfo_updater())

    led.standby()
    oled.start_standby_blink()
    print("待机中，短按按键开始采集，长按 3s 关机")

    while not _exit:
        await session_event.wait()
        session_event.clear()
        if _exit:
            break
        oled.stop_standby_blink()
        await run_session(mel_cfg, q_interp, d_interp, q_in, q_out, d_in, d_out)
        if not _exit:
            led.standby()
            oled.start_standby_blink()
            print("采集结束，短按按键重新开始")

    hb.cancel()
    si.cancel()
    power.stop()
    btn.stop()
    btn2.stop()
    led.cleanup()
    buzzer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
