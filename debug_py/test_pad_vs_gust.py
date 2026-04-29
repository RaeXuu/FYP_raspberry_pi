#!/usr/bin/env python3
"""
验证 scipy filtfilt method='pad' vs method='gust' 的差异。
在 Pi 上跑: python3 test_pad_vs_gust.py
"""
import numpy as np
import struct
from scipy.signal import butter, filtfilt, lfilter_zi

# ── 配置 ──
TEST_WAV = "/home/rasp4b/FypPi/WAV_record/002_2k.wav"  # 改成你的实际路径

# ── 加载 WAV ──
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

print(f"WAV: {TEST_WAV}")
print(f"sr={sample_rate}, chunk_len={len(chunk)}")
print(f"chunk[0..9] = {[f'{v:.6f}' for v in chunk[:10]]}")

# ── 滤波器 ──
b, a = butter(5, [25/1000, 400/1000], btype='band')
zi = lfilter_zi(b, a)
print(f"\nlfilter_zi[0..4] = {[f'{v:.8f}' for v in zi[:5]]}")

# ── method='pad' (scipy 默认) ──
y_pad = filtfilt(b, a, chunk)
print(f"\n=== method='pad' (scipy 默认) ===")
print(f"[0..9]     = {[f'{v:.6f}' for v in y_pad[:10]]}")
print(f"[100..109] = {[f'{v:.6f}' for v in y_pad[100:110]]}")
print(f"min={y_pad.min():.6f} max={y_pad.max():.6f}")

# ── method='gust' ──
y_gust = filtfilt(b, a, chunk, method='gust')
print(f"\n=== method='gust' ===")
print(f"[0..9]     = {[f'{v:.6f}' for v in y_gust[:10]]}")
print(f"[100..109] = {[f'{v:.6f}' for v in y_gust[100:110]]}")
print(f"min={y_gust.min():.6f} max={y_gust.max():.6f}")

# ── 差异 ──
diff = np.abs(y_pad - y_gust)
print(f"\n=== max |pad - gust| = {diff.max():.10f} ===")
print(f"mean |pad - gust|    = {diff.mean():.10f}")

# ── 第一窗口归一化后的值 ──
win_pad = y_pad[:4000].copy()
mx_pad = np.max(np.abs(win_pad))
if mx_pad > 0:
    win_pad = win_pad / mx_pad
print(f"\n=== 归一化后 (pad) ===")
print(f"[0..9] = {[f'{v:.6f}' for v in win_pad[:10]]} max={mx_pad:.6f}")

win_gust = y_gust[:4000].copy()
mx_gust = np.max(np.abs(win_gust))
if mx_gust > 0:
    win_gust = win_gust / mx_gust
print(f"\n=== 归一化后 (gust) ===")
print(f"[0..9] = {[f'{v:.6f}' for v in win_gust[:10]]} max={mx_gust:.6f}")

# ── 结论 ──
print(f"\n=== 结论 ===")
print(f"pad vs gust 差异是否显著: {'是' if diff.max() > 0.001 else '否'}")
print(f"如果差异显著且 iOS 实现的是 gust，则需要改为 pad")



#! Result:

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/test_pad_vs_gust.py
# WAV: /home/rasp4b/FypPi/WAV_record/002_2k.wav
# sr=2000, chunk_len=40000
# chunk[0..9] = ['0.890255', '0.740471', '0.404920', '0.080966', '-0.206946', '-0.471389', '-0.668203', '-0.538438', '-0.001160', '0.561083']

# lfilter_zi[0..4] = ['-0.01698771', '-0.01698771', '0.06795084', '0.06795084', '-0.10192626']

# === method='pad' (scipy 默认) ===
# [0..9]     = ['-0.000933', '-0.173057', '-0.371204', '-0.625141', '-0.934492', '-1.214314', '-1.298330', '-1.053703', '-0.530012', '0.023931']
# [100..109] = ['0.813509', '0.774466', '0.488363', '0.105156', '-0.285451', '-0.616477', '-0.763652', '-0.600762', '-0.157455', '0.334651']
# min=-1.298330 max=0.946456

# === method='gust' ===
# [0..9]     = ['0.754792', '0.668225', '0.443334', '0.114226', '-0.258972', '-0.575558', '-0.688464', '-0.482931', '-0.007063', '0.501302']
# [100..109] = ['0.824309', '0.782432', '0.493568', '0.107689', '-0.285491', '-0.618979', '-0.768492', '-0.607810', '-0.166571', '0.323613']
# min=-0.959554 max=0.940442

# === max |pad - gust| = 0.8412826168 ===
# mean |pad - gust|    = 0.0003864701

# === 归一化后 (pad) ===
# [0..9] = ['-0.000719', '-0.133292', '-0.285909', '-0.481497', '-0.719765', '-0.935289', '-1.000000', '-0.811584', '-0.408226', '0.018432'] max=1.298330

# === 归一化后 (gust) ===
# [0..9] = ['0.802592', '0.710544', '0.471410', '0.121460', '-0.275372', '-0.612008', '-0.732064', '-0.513515', '-0.007511', '0.533050'] max=0.940442

# === 结论 ===
# pad vs gust 差异是否显著: 是
# 如果差异显著且 iOS 实现的是 gust，则需要改为 pad