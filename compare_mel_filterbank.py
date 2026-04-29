#!/usr/bin/env python3
"""
对比 iOS MelSpectrogram 和 librosa melspectrogram 的输出差异。
重点检查：filterbank 权重矩阵、power_to_db、fix_length。
"""
import numpy as np

# ── 参数（与 iOS AudioProcessor.swift 完全一致）──
SR = 2000
N_FFT = 256
WIN_LEN = 256
HOP_LEN = 128
N_MELS = 64
FMIN = 25.0
FMAX = 400.0
POWER = 2.0
TOP_DB = 80.0

# ── iOS 的 STFT 第一帧（从 debug 输出抄下来的）──
ios_stft_frame0 = np.array([
    12.04, 14.16, 18.87, 64.05, 178.59, 265.46, 229.23, 124.61, 49.47, 27.63,
    # 后面的 bin 用 0 填充（实际会有值，但不影响 filterbank 对比）
], dtype=np.float64)

# 只取前 10 个 bin 用于快速对比
n_freqs = N_FFT // 2 + 1  # 129

# ── 1. 复刻 iOS 的 mel filterbank ──
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

mel_min = hz_to_mel(FMIN)
mel_max = hz_to_mel(FMAX)
mel_points_hz = np.array([
    mel_to_hz(mel_min + (mel_max - mel_min) * i / (N_MELS + 1))
    for i in range(N_MELS + 2)
])
bins_ios = np.floor(n_freqs * mel_points_hz / (SR / 2)).astype(int)

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

    # Slaney normalization
    bandwidth = mel_points_hz[m + 2] - mel_points_hz[m]
    if bandwidth > 0:
        fb_ios[m] *= 2.0 / bandwidth

# ── 2. librosa 的 mel filterbank ──
try:
    import librosa
    fb_librosa = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS,
        fmin=FMIN, fmax=FMAX, norm='slaney'
    )
    print("✅ librosa 已安装，进行对比")
    use_librosa = True
except ImportError:
    print("⚠️  librosa 未安装，仅打印 iOS filterbank 供手动对比")
    fb_librosa = None
    use_librosa = False

# ── 3. 打印对比 ──
print(f"\n=== Mel 参数 ===")
print(f"mel_points_hz (前5后5): {mel_points_hz[:5].round(2).tolist()} ... {mel_points_hz[-5:].round(2).tolist()}")
print(f"bins_ios (前5后5): {bins_ios[:5].tolist()} ... {bins_ios[-5:].tolist()}")
print(f"n_freqs={n_freqs}, bins 范围: {bins_ios.min()}-{bins_ios.max()}")

print(f"\n=== Filterbank 权重对比（前 5 个 mel band, 前 10 个 freq bin）===")
print(f"{'bin':>4s}  ", end="")
for k in range(10):
    print(f"{'k='+str(k):>12s}", end=" ")
print()
for m in range(min(5, N_MELS)):
    print(f"m={m:2d} iOS:", end=" ")
    for k in range(10):
        print(f"{fb_ios[m,k]:12.6f}", end=" ")
    print()
    if use_librosa:
        print(f"     LIB:", end=" ")
        for k in range(10):
            print(f"{fb_librosa[m,k]:12.6f}", end=" ")
        print()

if use_librosa:
    diff = np.abs(fb_ios - fb_librosa).max()
    print(f"\nfilterbank max diff: {diff:.10f}")
    if diff > 1e-6:
        # 找到第一个不同的位置
        diffs = np.abs(fb_ios - fb_librosa)
        max_idx = np.unravel_index(diffs.argmax(), diffs.shape)
        print(f"  最大差异在 m={max_idx[0]}, k={max_idx[1]}")
        print(f"  iOS={fb_ios[max_idx]:.10f}, librosa={fb_librosa[max_idx]:.10f}")

        # 打印完全不同的元素个数
        n_diff = np.sum(diffs > 1e-10)
        print(f"  不同元素数: {n_diff} / {N_MELS * n_freqs}")

# ── 4. 对比每个 mel band 的权重和 ──
print(f"\n=== 每个 mel band 的权重和（应为 1.0，Slaney 归一化后）===")
for m in range(min(10, N_MELS)):
    ios_sum = fb_ios[m].sum()
    lib_sum = fb_librosa[m].sum() if use_librosa else 0
    print(f"  m={m:2d}: iOS={ios_sum:.6f}", end="")
    if use_librosa:
        print(f"  librosa={lib_sum:.6f}  diff={abs(ios_sum-lib_sum):.10f}")
    else:
        print()

# ── 5. 对比 bins 数组 ──
if use_librosa:
    mel_f_librosa = librosa.mel_frequencies(N_MELS + 2, fmin=FMIN, fmax=FMAX)
    bins_librosa = np.floor((N_FFT + 1) * mel_f_librosa / SR).astype(int)
    print(f"\n=== Bin 索引对比 ===")
    print(f"mel_f_hz (librosa): {mel_f_librosa[:5].round(2).tolist()} ... {mel_f_librosa[-5:].round(2).tolist()}")
    print(f"mel_f_hz (iOS):     {mel_points_hz[:5].round(2).tolist()} ... {mel_points_hz[-5:].round(2).tolist()}")
    print(f"bins_librosa: {bins_librosa[:5].tolist()} ... {bins_librosa[-5:].tolist()}")
    print(f"bins_ios:     {bins_ios[:5].tolist()} ... {bins_ios[-5:].tolist()}")
    print(f"bins 是否一致: {np.array_equal(bins_ios, bins_librosa)}")

    # 打印 mel_f_hz 差异
    hz_diff = np.abs(mel_points_hz - mel_f_librosa)
    print(f"mel_f_hz max diff: {hz_diff.max():.6f} Hz")

# ── 6. 模拟 applyMelFB 对第一帧 ──
# 扩展 STFT 到完整 129 bin
stft_full = np.zeros(n_freqs, dtype=np.float64)
stft_full[:len(ios_stft_frame0)] = ios_stft_frame0

mel_ios_frame0 = fb_ios @ stft_full  # (64,)
mel_librosa_frame0 = fb_librosa @ stft_full if use_librosa else None

print(f"\n=== 第一帧 Mel（power domain, 前 10）===")
print(f"iOS:     {[f'{v:.4f}' for v in mel_ios_frame0[:10]]}")
if use_librosa:
    print(f"librosa: {[f'{v:.4f}' for v in mel_librosa_frame0[:10]]}")
    print(f"diff:    {[f'{abs(mel_ios_frame0[i]-mel_librosa_frame0[i]):.6f}' for i in range(10)]}")

# ── 7. power_to_db 对比 ──
eps = 1e-6
log_mel_ios = 10.0 * np.log10(np.maximum(mel_ios_frame0, eps))
# top_db clamp
gmax = log_mel_ios.max()
log_mel_ios = np.maximum(log_mel_ios, gmax - TOP_DB)

if use_librosa:
    log_mel_librosa = librosa.power_to_db(mel_librosa_frame0, top_db=TOP_DB)
    print(f"\n=== 第一帧 log-Mel（dB domain, 前 10）===")
    print(f"iOS:     {[f'{v:.4f}' for v in log_mel_ios[:10]]}")
    print(f"librosa: {[f'{v:.4f}' for v in log_mel_librosa[:10]]}")
    print(f"diff:    {[f'{abs(log_mel_ios[i]-log_mel_librosa[i]):.6f}' for i in range(10)]}")
else:
    print(f"\n=== 第一帧 log-Mel（dB domain, 前 10）===")
    print(f"iOS: {[f'{v:.4f}' for v in log_mel_ios[:10]]}")
    print(f"\n请与 iOS debug 输出的 Mel[0..<10] 对比:")
    print(f"iOS debug: 16.900284, -2.132806, -6.238535, 3.608603, -4.042701, -9.683898, -3.297316, 10.812631, 8.777715, 1.705882")

#! Result:

# (.venv) rasp4b@Rasp4B:~/FypPi $ /home/rasp4b/FypPi/.venv/bin/python /home/rasp4b/FypPi/compare_mel_filterbank.py
# ✅ librosa 已安装，进行对比

# === Mel 参数 ===
# mel_points_hz (前5后5): [25.0, 29.66, 34.36, 39.08, 43.84] ... [372.14, 379.04, 385.98, 392.97, 400.0]
# bins_ios (前5后5): [3, 3, 4, 5, 5] ... [48, 48, 49, 50, 51]
# n_freqs=129, bins 范围: 3-51

# === Filterbank 权重对比（前 5 个 mel band, 前 10 个 freq bin）===
#  bin           k=0          k=1          k=2          k=3          k=4          k=5          k=6          k=7          k=8          k=9 
# m= 0 iOS:     0.000000     0.000000     0.000000     0.213679     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000 
#      LIB:     0.000000     0.000000     0.000000     0.000000     0.158889     0.000000     0.000000     0.000000     0.000000     0.000000 
# m= 1 iOS:     0.000000     0.000000     0.000000     0.000000     0.212313     0.000000     0.000000     0.000000     0.000000     0.000000 
#      LIB:     0.000000     0.000000     0.000000     0.000000     0.014444     0.097500     0.000000     0.000000     0.000000     0.000000 
# m= 2 iOS:     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000 
#      LIB:     0.000000     0.000000     0.000000     0.000000     0.000000     0.075833     0.036111     0.000000     0.000000     0.000000 
# m= 3 iOS:     0.000000     0.000000     0.000000     0.000000     0.000000     0.209607     0.000000     0.000000     0.000000     0.000000 
#      LIB:     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.137222     0.000000     0.000000     0.000000 
# m= 4 iOS:     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000 
#      LIB:     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.000000     0.148056     0.000000     0.000000 

# filterbank max diff: 0.2136788194
#   最大差异在 m=0, k=3
#   iOS=0.2136788194, librosa=0.0000000000
#   不同元素数: 138 / 8256

# === 每个 mel band 的权重和（应为 1.0，Slaney 归一化后）===
#   m= 0: iOS=0.213679  librosa=0.158889  diff=0.0547899280
#   m= 1: iOS=0.212313  librosa=0.111944  diff=0.1003682778
#   m= 2: iOS=0.000000  librosa=0.111944  diff=0.1119444445
#   m= 3: iOS=0.209607  librosa=0.137222  diff=0.0723844431
#   m= 4: iOS=0.000000  librosa=0.148056  diff=0.1480555534
#   m= 5: iOS=0.206935  librosa=0.111944  diff=0.0949906704
#   m= 6: iOS=0.205612  librosa=0.111944  diff=0.0936676873
#   m= 7: iOS=0.000000  librosa=0.148056  diff=0.1480555534
#   m= 8: iOS=0.202991  librosa=0.137222  diff=0.0657692555
#   m= 9: iOS=0.201694  librosa=0.111944  diff=0.0897492709

# === Bin 索引对比 ===
# mel_f_hz (librosa): [25.0, 30.77, 36.54, 42.31, 48.08] ... [376.92, 382.69, 388.46, 394.23, 400.0]
# mel_f_hz (iOS):     [25.0, 29.66, 34.36, 39.08, 43.84] ... [372.14, 379.04, 385.98, 392.97, 400.0]
# bins_librosa: [3, 3, 4, 5, 6] ... [48, 49, 49, 50, 51]
# bins_ios:     [3, 3, 4, 5, 5] ... [48, 48, 49, 50, 51]
# bins 是否一致: False
# mel_f_hz max diff: 19.492337 Hz

# === 第一帧 Mel（power domain, 前 10）===
# iOS:     ['13.6861', '37.9169', '0.0000', '55.6422', '0.0000', '47.4357', '25.6213', '0.0000', '10.0420', '5.5728']
# librosa: ['28.3760', '28.4620', '28.4085', '31.4555', '18.4492', '7.4373', '4.9858', '4.0908', '0.0000', '0.0000']
# diff:    ['14.689839', '9.454947', '28.408467', '24.186736', '18.449203', '39.998473', '20.635503', '4.090775', '10.041989', '5.572797']

# === 第一帧 log-Mel（dB domain, 前 10）===
# iOS:     ['11.3628', '15.7883', '-60.0000', '17.4540', '-60.0000', '16.7611', '14.0860', '-60.0000', '10.0182', '7.4607']
# librosa: ['14.5295', '14.5427', '14.5345', '14.9770', '12.6598', '8.7141', '6.9774', '6.1181', '-65.0230', '-65.0230']
# diff:    ['3.166701', '1.245680', '74.534478', '2.477083', '72.659776', '8.046925', '7.108646', '66.118056', '75.041238', '72.483773']
# (.venv) rasp4b@Rasp4B:~/FypPi $ 