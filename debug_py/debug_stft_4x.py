#!/usr/bin/env python3
"""
对比 Pi/iOS STFT 第一帧，定位 4x power 差异的来源。
在 Pi 上跑: python3 debug_stft_4x.py
"""
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hann
import struct

# ── 加载与 iOS 相同的 WAV ──
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

# ── 滤波 ──
b, a = butter(5, [25/1000, 400/1000], btype='band')
filt = filtfilt(b, a, chunk)

# ── 第一窗口归一化 ──
window_sig = filt[:4000].copy()
mx2 = np.max(np.abs(window_sig))
if mx2 > 0:
    window_sig = window_sig / mx2

print(f"window_sig[0..9] = {[f'{v:.6f}' for v in window_sig[:10]]}")
print(f"window_sig max = {mx2:.6f}")

# ── STFT 第一帧 ──
n_fft = 256
hop_len = 128
win_len = 256

# Pad (center=True, matching librosa + iOS)
pad = n_fft // 2  # 128
padded = np.concatenate([np.zeros(pad), window_sig, np.zeros(pad)])

# ── 1. Window comparison ──
# scipy hann (what librosa uses)
scipy_win = hann(win_len, sym=True)
print(f"\n=== scipy hann window ===")
print(f"  [0..4]  = {[f'{v:.10f}' for v in scipy_win[:5]]}")
print(f"  [123..127] = {[f'{v:.10f}' for v in scipy_win[123:128]]}")
print(f"  [128..132] = {[f'{v:.10f}' for v in scipy_win[128:133]]}")
print(f"  [251..255] = {[f'{v:.10f}' for v in scipy_win[251:256]]}")
print(f"  min={scipy_win.min():.10f} max={scipy_win.max():.10f}")
print(f"  sum={scipy_win.sum():.6f} mean={scipy_win.mean():.10f}")
print(f"  coherent_gain (mean) = {scipy_win.mean():.10f}")

# ── 2. Frame 0 samples (after windowing) ──
frame0 = np.zeros(n_fft)
for j in range(win_len):
    frame0[j] = padded[0 + j] * scipy_win[j]

print(f"\n=== Frame 0 (windowed, first 128 are zeros from center pad) ===")
print(f"  [0..4]   = {[f'{v:.10f}' for v in frame0[:5]]}")
print(f"  [123..132] = {[f'{v:.10f}' for v in frame0[123:133]]}")
print(f"  [251..255] = {[f'{v:.10f}' for v in frame0[251:256]]}")
print(f"  non-zero starts at index 128 (after center pad)")
print(f"  frame0[128..132] = {[f'{v:.10f}' for v in frame0[128:133]]}")

# ── 3. numpy FFT ──
S = np.fft.rfft(frame0)
power = np.abs(S) ** 2
print(f"\n=== numpy rfft power ===")
print(f"  [0..9] = {[f'{v:.2f}' for v in power[:10]]}")

# ── 4. FFT split-complex input packing verification ──
# vDSP real FFT split complex format:
#   Input: split.real[k] = frame[2k], split.imag[k] = frame[2k+1]
# But wait, vDSP also uses the OTHER packing:
#   split.real[0..127] = frame[0..127]
#   split.imag[0..127] = frame[128..255]
# Let's verify BOTH interpretations

# Interpretation A: even/odd split
split_real_A = frame0[0::2]   # even indices: 0,2,4,...,254
split_imag_A = frame0[1::2]   # odd indices:  1,3,5,...,255
xfft_A = split_real_A + 1j * split_imag_A  # 128 complex values
S_A = np.fft.fft(xfft_A)  # complex FFT of 128 points (not real FFT of 256!)
power_A = np.abs(S_A) ** 2
print(f"\n=== Interpretation A: even/odd interleave → 128-pt complex FFT ===")
print(f"  [0..9] = {[f'{v:.2f}' for v in power_A[:10]]}")
print(f"  ratio to rfft @ DC: {power_A[0]/power[0]:.4f}")

# Interpretation B: first half / second half
split_real_B = frame0[:128]   # first 128 samples
split_imag_B = frame0[128:256]  # second 128 samples
xfft_B = split_real_B + 1j * split_imag_B  # 128 complex values
S_B = np.fft.fft(xfft_B)  # complex FFT of 128 points
power_B = np.abs(S_B) ** 2
print(f"\n=== Interpretation B: first/second half → 128-pt complex FFT ===")
print(f"  [0..9] = {[f'{v:.2f}' for v in power_B[:10]]}")
print(f"  ratio to rfft @ DC: {power_B[0]/power[0]:.4f}")

# ── 5. What vDSP really does for real FFT ──
# vDSP_fft_zop input: N=256 real values
# Packing: split.real[0..127] = x[0..127], split.imag[0..127] = x[128..255]
# This is interpretation B

# But the FFT of 256 real values should give the SAME result as np.fft.rfft
# Both are DFT of 256 real values

# ── 6. Test: vDSP-style real FFT vs numpy rfft ──
# Just to confirm: numpy rfft is mathematically the DFT of 256 real values
# DC = sum(frame0)
dc_numpy = np.sum(frame0)
dc_power_numpy = dc_numpy ** 2
print(f"\n=== DC verification ===")
print(f"  sum(frame0) = {dc_numpy:.6f}")
print(f"  |sum(frame0)|^2 = {dc_power_numpy:.2f}")
print(f"  power[0] (from rfft) = {power[0]:.2f}")
print(f"  Match: {abs(dc_power_numpy - power[0]) < 0.01}")

# ── 7. Key comparison with iOS ──
print(f"\n=== Key values for iOS comparison ===")
print(f"Expected iOS DC power = {power[0]:.2f}")
print(f"iOS reported STFT[0][0] = 48.16")
print(f"Ratio iOS/Pi = {48.16/power[0]:.2f}")

# ── 8. Check if 48.16 could come from interpretation A ──
print(f"\n=== If iOS uses interpretation A ===")
print(f"  power_A[0] = {power_A[0]:.2f}")
print(f"  ratio to 48.16: {48.16/power_A[0]:.4f}")

# The ratio should be ~1.0 if interpretation A is what iOS does
print(f"\n=== If iOS uses interpretation B (correct for real FFT) ===")
print(f"  power_B[0] = {power_B[0]:.2f}")
print(f"  ratio to 48.16: {48.16/power_B[0]:.4f}")



#！Result: 

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/debug_stft_4x.py
# window_sig[0..9] = ['0.102376', '0.276823', '0.366962', '0.345863', '0.253044', '0.154285', '0.092387', '0.071626', '0.078215', '0.103165']
# window_sig max = 0.595830

# === scipy hann window ===
#   [0..4]  = ['0.0000000000', '0.0001517740', '0.0006070039', '0.0013654133', '0.0024265418']
#   [123..127] = ['0.9969295684', '0.9981418264', '0.9990516644', '0.9996585301', '0.9999620551']
#   [128..132] = ['0.9999620551', '0.9996585301', '0.9990516644', '0.9981418264', '0.9969295684']
#   [251..255] = ['0.0024265418', '0.0013654133', '0.0006070039', '0.0001517740', '0.0000000000']
#   min=0.0000000000 max=0.9999620551
#   sum=127.500000 mean=0.4980468750
#   coherent_gain (mean) = 0.4980468750

# === Frame 0 (windowed, first 128 are zeros from center pad) ===
#   [0..4]   = ['0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000']
#   [123..132] = ['0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '0.0000000000', '0.1023717471', '0.2767283629', '0.3666141366', '0.3452199258', '0.2522671837']
#   [251..255] = ['0.0005983340', '0.0003330856', '0.0001408772', '0.0000331459', '0.0000000000']
#   non-zero starts at index 128 (after center pad)
#   frame0[128..132] = ['0.1023717471', '0.2767283629', '0.3666141366', '0.3452199258', '0.2522671837']

# === numpy rfft power ===
#   [0..9] = ['12.09', '14.12', '19.00', '63.96', '175.96', '260.52', '225.27', '122.98', '49.31', '27.61']

# === Interpretation A: even/odd interleave → 128-pt complex FFT ===
#   [0..9] = ['6.05', '6.89', '9.04', '29.64', '79.39', '114.34', '96.08', '50.93', '19.82', '10.77']
#   ratio to rfft @ DC: 0.5000

# === Interpretation B: first/second half → 128-pt complex FFT ===
#   [0..9] = ['12.09', '19.00', '175.96', '225.27', '49.31', '30.93', '47.21', '43.58', '6.03', '0.99']
#   ratio to rfft @ DC: 1.0000

# === DC verification ===
#   sum(frame0) = 3.477652
#   |sum(frame0)|^2 = 12.09
#   power[0] (from rfft) = 12.09
#   Match: True

# === Key values for iOS comparison ===
# Expected iOS DC power = 12.09
# iOS reported STFT[0][0] = 48.16
# Ratio iOS/Pi = 3.98

# === If iOS uses interpretation A ===
#   power_A[0] = 6.05
#   ratio to 48.16: 7.9642

# === If iOS uses interpretation B (correct for real FFT) ===
#   power_B[0] = 12.09
#   ratio to 48.16: 3.9821