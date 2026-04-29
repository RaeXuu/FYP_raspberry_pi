#!/usr/bin/env python3
"""
验证 scipy filtfilt (method='gust') 对完整 40000 样本的输出。
在 Pi 上 run: python3 /home/rasp4b/FypPi/verify_filtfilt.py
"""
import numpy as np
import struct
from scipy.signal import butter, filtfilt, lfilter_zi

# ── 1. 生成与 iOS DebugPreprocess 完全相同的输入 ──
TEST_WAV = "/home/rasp4b/FypPi/data/raw/DataSet2/training-a/a0002.wav"

with open(TEST_WAV, 'rb') as f:
    data = f.read()

sample_rate  = struct.unpack_from('<I', data, 24)[0]
bits         = struct.unpack_from('<H', data, 34)[0]
channels     = struct.unpack_from('<H', data, 22)[0]

offset = 36
while offset + 8 <= len(data):
    cid   = data[offset:offset+4]
    csz   = struct.unpack_from('<I', data, offset+4)[0]
    if cid == b'data':
        raw = data[offset+8 : offset+8+csz]; break
    offset += 8 + csz

samples = [struct.unpack_from('<h', raw, i)[0] / 32768.0
           for i in range(0, len(raw)-1, 2)]
if channels > 1:
    samples = samples[::channels]

if sample_rate != 2000:
    r = sample_rate // 2000
    samples = [sum(samples[i:i+r]) / r for i in range(0, len(samples)-r+1, r)]

samples = np.array(samples, dtype=np.float64)
mx = np.max(np.abs(samples))
if mx > 0:
    samples = samples / mx

chunk = samples[:40000]
if len(chunk) < 40000:
    chunk = np.pad(chunk, (0, 40000 - len(chunk)))

print(f"WAV: sr={sample_rate}, samples_after_ds={len(samples)}, chunk={len(chunk)}")
print(f"[Pi] 原始: first10={[f'{v:.6f}' for v in chunk[:10]]}")

# ── 2. 滤波 ──
b, a = butter(5, [25/1000, 400/1000], btype='band')
zi = lfilter_zi(b, a)
print(f"\n[Pi] lfilter_zi[0..4] = {zi[:5]}")
print(f"[Pi] zi*chunk[0] → y_fwd[0] 应 ≈ 0")

filt = filtfilt(b, a, chunk)  # method='gust' is default in scipy >= 1.8

print(f"\n[Pi] filtfilt first10 = {[f'{v:.6f}' for v in filt[:10]]}")
print(f"[Pi] filtfilt [100:110]= {[f'{v:.6f}' for v in filt[100:110]]}")
print(f"[Pi] filtfilt min={filt.min():.5f} max={filt.max():.5f}")

# ── 3. 第一窗口归一化 ──
win = filt[:4000].copy()
mx2 = np.max(np.abs(win))
if mx2 > 0:
    win = win / mx2
print(f"\n[Pi] 归一化 first10 = {[f'{v:.6f}' for v in win[:10]]} max={mx2:.6f}")
