#!/usr/bin/env python3
"""
用修正后的 iOS 公式（Slaney mel scale + ceil bin mapping）对比 librosa filterbank，
并用完整的 Pi 预处理 pipeline 输出 Mel 值，供 iOS 对比。
"""
import numpy as np
from scipy.signal import butter, filtfilt
import struct
import sys

# ── 参数 ──
SR = 2000
N_FFT = 256
N_MELS = 64
FMIN = 25.0
FMAX = 400.0
TOP_DB = 80.0
EPS = 1e-6

# ── Slaney mel scale (librosa htk=False default) ──
def hz_to_mel(hz):
    if hz < 1000:
        return hz * 3.0 / 200.0
    else:
        logstep = np.log(6.4) / 27.0
        return 15.0 + np.log(hz / 1000.0) / logstep

def mel_to_hz(mel):
    if mel < 15.0:
        return mel * 200.0 / 3.0
    else:
        logstep = np.log(6.4) / 27.0
        return 1000.0 * np.exp(logstep * (mel - 15.0))

# ── 复刻修正后的 iOS filterbank ──
n_freqs = N_FFT // 2 + 1  # 129

mel_min = hz_to_mel(FMIN)
mel_max = hz_to_mel(FMAX)
mel_points_hz = np.array([mel_to_hz(mel_min + (mel_max - mel_min) * i / (N_MELS + 1))
                          for i in range(N_MELS + 2)])

# iOS 修正后的 bin mapping: ceil(nFFT * f / sr)
bins_ios = np.ceil(N_FFT * mel_points_hz / SR).astype(int)

fb_ios = np.zeros((N_MELS, n_freqs))
for m in range(N_MELS):
    start = bins_ios[m]
    center = bins_ios[m + 1]
    end = min(bins_ios[m + 2], n_freqs - 1)
    if center > start:
        for k in range(start, center):
            fb_ios[m, k] = (k - start) / (center - start)
    if end > center:
        for k in range(center, end + 1):
            fb_ios[m, k] = (end - k) / (end - center)
    bandwidth = mel_points_hz[m + 2] - mel_points_hz[m]
    if bandwidth > 0:
        fb_ios[m] *= 2.0 / bandwidth

# ── librosa filterbank ──
import librosa
fb_librosa = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS,
                                 fmin=FMIN, fmax=FMAX, norm='slaney')

print("=== 修正后 iOS filterbank vs librosa ===")
diff = np.abs(fb_ios - fb_librosa).max()
print(f"max diff: {diff:.10f}")
diffs = np.abs(fb_ios - fb_librosa)
n_diff = np.sum(diffs > 1e-10)
print(f"不同元素数: {n_diff} / {N_MELS * n_freqs}")

print(f"\n=== 每个 mel band 权重和（前10）===")
for m in range(10):
    print(f"  m={m:2d}: iOS={fb_ios[m].sum():.6f}  librosa={fb_librosa[m].sum():.6f}  diff={abs(fb_ios[m].sum()-fb_librosa[m].sum()):.10f}")

# ── mel_points 对比 ──
mel_f_librosa = librosa.mel_frequencies(N_MELS + 2, fmin=FMIN, fmax=FMAX)
print(f"\n=== Mel 频点对比 ===")
print(f"iOS:     {mel_points_hz[:5].round(4).tolist()} ... {mel_points_hz[-5:].round(4).tolist()}")
print(f"librosa: {mel_f_librosa[:5].round(4).tolist()} ... {mel_f_librosa[-5:].round(4).tolist()}")
print(f"max hz diff: {np.abs(mel_points_hz - mel_f_librosa).max():.6f}")

print(f"\n=== Bin 索引对比 ===")
bins_librosa_real = []
fftfreqs = np.linspace(0, SR/2, n_freqs)
for f in mel_f_librosa:
    bins_librosa_real.append(np.searchsorted(fftfreqs, f))
print(f"iOS ceil:      {bins_ios[:5].tolist()} ... {bins_ios[-5:].tolist()}")
print(f"librosa searchsorted: {bins_librosa_real[:5]} ... {bins_librosa_real[-5:]}")
print(f"bins 一致: {bins_ios.tolist() == bins_librosa_real}")

# ── 加载 WAV 并走完整预处理 ──
print(f"\n=== 完整预处理对比（WAV: a0002.wav）===")

TEST_WAV = "/home/rasp4b/FypPi/data/raw/DataSet2/training-a/a0002.wav"
with open(TEST_WAV, 'rb') as f:
    data = f.read()

sample_rate = struct.unpack_from('<I', data, 24)[0]
bits = struct.unpack_from('<H', data, 34)[0]
channels = struct.unpack_from('<H', data, 22)[0]

offset = 36
while offset + 8 <= len(data):
    cid = data[offset:offset+4]
    csz = struct.unpack_from('<I', data, offset+4)[0]
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

print(f"原始 first10: {[f'{v:.6f}' for v in chunk[:10]]}")

b, a = butter(5, [25/1000, 400/1000], btype='band')
filt = filtfilt(b, a, chunk)
print(f"滤波后 first10: {[f'{v:.6f}' for v in filt[:10]]}")
print(f"滤波后[100..<110]: {[f'{v:.6f}' for v in filt[100:110]]}")

window_sig = filt[:4000].copy()
mx2 = np.max(np.abs(window_sig))
if mx2 > 0:
    window_sig = window_sig / mx2
print(f"归一化后 first10: {[f'{v:.6f}' for v in window_sig[:10]]} max={mx2:.6f}")

# ── STFT (librosa-compatible) ──
pad = N_FFT // 2
padded = np.concatenate([np.zeros(pad), window_sig, np.zeros(pad)])
n_frames_stft = 1 + (len(padded) - N_FFT) // 128
print(f"STFT n_frames={n_frames_stft}")

# Use librosa STFT for reference
S = librosa.stft(window_sig.astype(np.float64), n_fft=N_FFT, hop_length=128,
                 win_length=N_FFT, window='hann', center=True, pad_mode='constant')
power = np.abs(S) ** 2
print(f"librosa STFT[0] first10: {[f'{v:.2f}' for v in power[0, :10]]}")

# ── Mel spectrogram (iOS 公式) ──
mel_ios = fb_ios @ power  # (64, n_frames)
log_mel_ios = 10.0 * np.log10(np.maximum(mel_ios, EPS))
gmax = log_mel_ios.max()
log_mel_ios = np.maximum(log_mel_ios, gmax - TOP_DB)

# fix_length to 64
current_frames = log_mel_ios.shape[1]
if current_frames > 64:
    log_mel_ios = log_mel_ios[:, :64]
elif current_frames < 64:
    log_mel_ios = np.pad(log_mel_ios, ((0, 0), (0, 64 - current_frames)))

flat_ios = log_mel_ios.flatten()
print(f"\n[iOS公式] Mel[0..<10]  = {[f'{v:.6f}' for v in flat_ios[:10]]}")
print(f"[iOS公式] Mel[64..<74] = {[f'{v:.6f}' for v in flat_ios[64:74]]}")
print(f"[iOS公式] Mel min={flat_ios.min():.6f} max={flat_ios.max():.6f} mean={flat_ios.mean():.6f}")

# ── Mel spectrogram (librosa 公式) ──
mel_lib = fb_librosa @ power
log_mel_lib = 10.0 * np.log10(np.maximum(mel_lib, EPS))
gmax2 = log_mel_lib.max()
log_mel_lib = np.maximum(log_mel_lib, gmax2 - TOP_DB)
if log_mel_lib.shape[1] > 64:
    log_mel_lib = log_mel_lib[:, :64]
elif log_mel_lib.shape[1] < 64:
    log_mel_lib = np.pad(log_mel_lib, ((0, 0), (0, 64 - log_mel_lib.shape[1])))
flat_lib = log_mel_lib.flatten()
print(f"\n[librosa] Mel[0..<10]  = {[f'{v:.6f}' for v in flat_lib[:10]]}")
print(f"[librosa] Mel[64..<74] = {[f'{v:.6f}' for v in flat_lib[64:74]]}")
print(f"[librosa] Mel min={flat_lib.min():.6f} max={flat_lib.max():.6f} mean={flat_lib.mean():.6f}")

# ── 对比 ──
print(f"\n=== iOS公式 vs librosa Mel 差异 ===")
print(f"max diff: {np.abs(flat_ios - flat_lib).max():.6f}")
print(f"mean diff: {np.abs(flat_ios - flat_lib).mean():.6f}")

# ── 也输出 librosa 原生的 melspectrogram 做最终验证 ──
mel_librosa_native = librosa.feature.melspectrogram(
    y=window_sig.astype(np.float64), sr=SR, n_fft=N_FFT,
    hop_length=128, win_length=N_FFT, window='hann',
    n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    power=2.0, center=True, pad_mode='constant'
)
log_mel_native = librosa.power_to_db(mel_librosa_native, top_db=TOP_DB)
if log_mel_native.shape[1] > 64:
    log_mel_native = log_mel_native[:, :64]
elif log_mel_native.shape[1] < 64:
    log_mel_native = np.pad(log_mel_native, ((0, 0), (0, 64 - log_mel_native.shape[1])))
flat_native = log_mel_native.flatten()
print(f"\n[librosa原生] Mel[0..<10]  = {[f'{v:.6f}' for v in flat_native[:10]]}")
print(f"[librosa原生] Mel[64..<74] = {[f'{v:.6f}' for v in flat_native[64:74]]}")
print(f"[librosa原生] Mel min={flat_native.min():.6f} max={flat_native.max():.6f} mean={flat_native.mean():.6f}")

# ── 关键：对比 iOS debug 输出 ──
print(f"\n=== 请与 iOS debug 输出对比 ===")
print(f"iOS Mel[0..<10]  = [\"16.900284\", \"-2.132806\", ...] (旧filterbank)")
print(f"最新 iOS Mel[0..<10] 应该是上面的 [iOS公式] 行")


#! Result:

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/compare_mel_v2.py
# === 修正后 iOS filterbank vs librosa ===
# max diff: 0.1697222292
# 不同元素数: 94 / 8256

# === 每个 mel band 权重和（前10）===
#   m= 0: iOS=0.173333  librosa=0.158889  diff=0.0144444420
#   m= 1: iOS=0.173333  librosa=0.111944  diff=0.0613888889
#   m= 2: iOS=0.173333  librosa=0.111944  diff=0.0613888889
#   m= 3: iOS=0.000000  librosa=0.137222  diff=0.1372222304
#   m= 4: iOS=0.173333  librosa=0.148056  diff=0.0252777799
#   m= 5: iOS=0.173333  librosa=0.111944  diff=0.0613888889
#   m= 6: iOS=0.173333  librosa=0.111944  diff=0.0613888889
#   m= 7: iOS=0.000000  librosa=0.148056  diff=0.1480555534
#   m= 8: iOS=0.173333  librosa=0.137222  diff=0.0361111029
#   m= 9: iOS=0.173333  librosa=0.111944  diff=0.0613888889

# === Mel 频点对比 ===
# iOS:     [25.0, 30.7692, 36.5385, 42.3077, 48.0769] ... [376.9231, 382.6923, 388.4615, 394.2308, 400.0]
# librosa: [25.0, 30.7692, 36.5385, 42.3077, 48.0769] ... [376.9231, 382.6923, 388.4615, 394.2308, 400.0]
# max hz diff: 0.000000

# === Bin 索引对比 ===
# iOS ceil:      [4, 4, 5, 6, 7] ... [49, 49, 50, 51, 52]
# librosa searchsorted: [np.int64(4), np.int64(4), np.int64(5), np.int64(6), np.int64(7)] ... [np.int64(49), np.int64(49), np.int64(50), np.int64(51), np.int64(52)]
# bins 一致: False

# === 完整预处理对比（WAV: a0002.wav）===
# 原始 first10: ['0.079518', '0.194913', '0.208568', '0.214726', '0.157965', '0.064793', '0.050067', '0.035877', '0.037216', '0.025971']
# 滤波后 first10: ['0.060998', '0.164939', '0.218647', '0.206075', '0.150771', '0.091928', '0.055047', '0.042677', '0.046603', '0.061468']
# 滤波后[100..<110]: ['0.029858', '0.016367', '-0.002358', '-0.021370', '-0.036405', '-0.045480', '-0.048822', '-0.047622', '-0.042870', '-0.035199']
# 归一化后 first10: ['0.102376', '0.276823', '0.366962', '0.345863', '0.253044', '0.154285', '0.092387', '0.071626', '0.078215', '0.103165'] max=0.595830
# STFT n_frames=32
# librosa STFT[0] first10: ['12.04', '0.01', '0.00', '0.00', '0.00', '0.00', '0.00', '0.01', '0.00', '0.01']

# [iOS公式] Mel[0..<10]  = ['14.907461', '13.799040', '-6.825631', '4.232626', '5.767892', '-7.249492', '0.054625', '18.377909', '14.554249', '-6.132474']
# [iOS公式] Mel[64..<74] = ['16.628754', '12.170673', '-1.699789', '3.756689', '5.010837', '-4.064660', '-2.181861', '16.481785', '-0.360164', '4.862310']
# [iOS公式] Mel min=-60.000000 max=19.360565 mean=-13.671960

# [librosa] Mel[0..<10]  = ['14.529576', '13.421155', '-7.203517', '3.854740', '5.390006', '-7.627377', '-0.323261', '18.000023', '14.176364', '-6.510360']
# [librosa] Mel[64..<74] = ['14.542605', '10.519600', '-4.005279', '1.922316', '3.217467', '-6.264907', '-3.718702', '14.879411', '4.617847', '2.414405']
# [librosa] Mel min=-60.000000 max=18.000023 mean=-8.664348

# === iOS公式 vs librosa Mel 差异 ===
# max diff: 75.657267
# mean diff: 6.124143

# [librosa原生] Mel[0..<10]  = ['14.529576', '13.421155', '-7.203517', '3.854740', '5.390006', '-7.627377', '-0.323261', '18.000023', '14.176364', '-6.510360']
# [librosa原生] Mel[64..<74] = ['14.542605', '10.519600', '-4.005279', '1.922316', '3.217467', '-6.264907', '-3.718702', '14.879411', '4.617847', '2.414405']
# [librosa原生] Mel min=-60.984614 max=18.000023 mean=-8.664588

# === 请与 iOS debug 输出对比 ===
# iOS Mel[0..<10]  = ["16.900284", "-2.132806", ...] (旧filterbank)
# 最新 iOS Mel[0..<10] 应该是上面的 [iOS公式] 行