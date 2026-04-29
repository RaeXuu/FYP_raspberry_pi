"""
完全对齐 iOS DebugPreprocess.swift 的预处理步骤
加载 WAV → 手动 int16→float → 峰值归一化 → 带通滤波
打印每一步中间值，与 iOS 输出对比
"""
import numpy as np
import struct
from scipy.signal import butter, filtfilt

TEST_WAV = "/home/rasp4b/FypPi/data/raw/DataSet2/training-a/a0002.wav"

# ── Step 1: 手动读 WAV（对齐 iOS readWAV）──
with open(TEST_WAV, 'rb') as f:
    data = f.read()

# 解析 WAV header
sample_rate = struct.unpack_from('<I', data, 24)[0]
bits_per_sample = struct.unpack_from('<H', data, 34)[0]
num_channels = struct.unpack_from('<H', data, 22)[0]
print(f"WAV: sr={sample_rate}, bits={bits_per_sample}, ch={num_channels}")

# 找 data chunk
offset = 36
while offset + 8 <= len(data):
    chunk_id = data[offset:offset+4]
    chunk_size = struct.unpack_from('<I', data, offset+4)[0]
    if chunk_id == b'data':
        raw_start = offset + 8
        raw_end = min(raw_start + chunk_size, len(data))
        break
    offset += 8 + chunk_size

# int16 → float / 32768.0
raw_bytes = data[raw_start:raw_end]
samples = []
if bits_per_sample == 16:
    for i in range(0, len(raw_bytes) - 1, 2):
        val = struct.unpack_from('<h', raw_bytes, i)[0]
        samples.append(val / 32768.0)

# 多声道取左声道
if num_channels > 1:
    samples = samples[::num_channels]

# 降采样 (simple averaging, 对齐 iOS)
if sample_rate != 2000:
    ratio = sample_rate // 2000
    down = []
    for i in range(0, len(samples) - ratio + 1, ratio):
        chunk = samples[i:i+ratio]
        down.append(sum(chunk) / len(chunk))
    samples = down
    print(f"降采样: {sample_rate} -> 2000Hz, 样本数: {len(samples)}")

samples = np.array(samples, dtype=np.float64)

# ── Step 2: 全局峰值归一化 ──
global_max = np.max(np.abs(samples))
if global_max > 0:
    samples = samples / global_max

chunk = samples[:40000]
if len(chunk) < 40000:
    chunk = np.pad(chunk, (0, 40000 - len(chunk)))

print(f"\n[Debug] 原始: first10={[f'{v:.6f}' for v in chunk[:10]]}")

# ── Step 3: 带通滤波 ──
b, a = butter(5, [25/1000, 400/1000], btype='band')
filt = filtfilt(b, a, chunk)
print(f"[Debug] 滤波后: first10={[f'{v:.6f}' for v in filt[:10]]}")
print(f"[Debug] 滤波后[100..<110]={[f'{v:.6f}' for v in filt[100:110]]}")

# ── Step 4: 第一个窗口峰值归一化 ──
window = filt[:4000].copy()
mx = np.max(np.abs(window))
if mx > 0:
    window = window / mx
print(f"[Debug] 归一化后: first10={[f'{v:.6f}' for v in window[:10]]} max={mx:.6f}")

#! Result :

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/check_filter.py
# WAV: sr=2000, bits=16, ch=1

# [Debug] 原始: first10=['0.079518', '0.194913', '0.208568', '0.214726', '0.157965', '0.064793', '0.050067', '0.035877', '0.037216', '0.025971']
# [Debug] 滤波后: first10=['0.060998', '0.164939', '0.218647', '0.206075', '0.150771', '0.091928', '0.055047', '0.042677', '0.046603', '0.061468']
# [Debug] 滤波后[100..<110]=['0.029858', '0.016367', '-0.002358', '-0.021370', '-0.036405', '-0.045480', '-0.048822', '-0.047622', '-0.042870', '-0.035199']
# [Debug] 归一化后: first10=['0.102376'
