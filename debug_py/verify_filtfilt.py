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

#! Result:

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/verify_filtfilt.py
# WAV: sr=2000, samples_after_ds=41657, chunk=40000
# [Pi] 原始: first10=['0.079518', '0.194913', '0.208568', '0.214726', '0.157965', '0.064793', '0.050067', '0.035877', '0.037216', '0.025971']

# [Pi] lfilter_zi[0..4] = [-0.01698771 -0.01698771  0.06795084  0.06795084 -0.10192626]
# [Pi] zi*chunk[0] → y_fwd[0] 应 ≈ 0

# [Pi] filtfilt first10 = ['0.060998', '0.164939', '0.218647', '0.206075', '0.150771', '0.091928', '0.055047', '0.042677', '0.046603', '0.061468']
# [Pi] filtfilt [100:110]= ['0.029858', '0.016367', '-0.002358', '-0.021370', '-0.036405', '-0.045480', '-0.048822', '-0.047622', '-0.042870', '-0.035199']
# [Pi] filtfilt min=-1.01153 max=0.79044

# [Pi] 归一化 first10 = ['0.102376', '0.276823', '0.366962', '0.345863', '0.253044', '0.154285', '0.092387', '0.071626', '0.078215', '0.103165'] max=0.595830
