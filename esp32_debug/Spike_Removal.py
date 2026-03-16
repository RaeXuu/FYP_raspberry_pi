import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, medfilt

def pro_pipeline(filename):
    # 1. 读取数据 (复用老代码读取逻辑)
    fs, data = wavfile.read(filename)
    if len(data.shape) > 1: data = data[:, 0]
    data = data / 32768.0  # 基础归一化

    # 2. 中值滤波 (前哨去刺)
    data_med = medfilt(data, kernel_size=5)

    # 3. 文献级滤波组合 
    # 第一道门：3阶高通 (4Hz) 移除漂移
    sos_hp = butter(3, 4, btype='high', fs=fs, output='sos')
    data_hp = sosfilt(sos_hp, data_med)
    
    # 第二道门：2阶带通 (20-220Hz) 锁定心音
    sos_bp = butter(2, [20, 220], btype='band', fs=fs, output='sos')
    cleaned = sosfilt(sos_bp, data_hp)

    # 4. 温和归一化 (让波形看起来“漂亮”的关键)
    # 不用 Z-score，改用最大值缩放，并在上下留出 10% 的空间
    cleaned = cleaned / (np.max(np.abs(cleaned)) + 1e-6)

    return data, cleaned, fs

# --- 绘图区 ---
filename = "heart_sound_1770104436.wav" # 改成你的文件名
raw, cleaned, fs = pro_pipeline(filename)
time = np.linspace(0, len(raw)/fs, len(raw))

plt.figure(figsize=(12, 8))

# 子图1：时域波形 (找回老代码的质感)
plt.subplot(2, 1, 1)
# 设置观察区间 (2秒到7秒，方便看清心跳周期)
mask = (time >= 2.0) & (time <= 7.0)
plt.plot(time[mask], cleaned[mask], color='#007acc', linewidth=0.8) 
plt.title(f"Processed Heart Sound (Refined Pipeline) - {filename}")
plt.ylabel("Normalized Amplitude")
plt.ylim(-1.2, 1.2) # 留出上下边距，不挤
plt.grid(True, linestyle='--', alpha=0.4)

# 子图2：语谱图 (复刻老代码的红色质感)
plt.subplot(2, 1, 2)
# 使用与文献一致的参数 [cite: 815]
plt.specgram(cleaned, NFFT=2048, Fs=fs, noverlap=1024, cmap='magma')
plt.ylim(0, 500)
plt.title("Frequency Energy Distribution (20-220Hz Focus)")
plt.colorbar(label='Intensity (dB)')

plt.tight_layout()
plt.show()