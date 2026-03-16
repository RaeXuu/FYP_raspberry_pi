import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import os

# ==========================================
# 👇 请在这里修改为你录制好的 wav 文件名
filename = "heart_sound_1770104436.wav"  # 可以改成 "150hz.wav", "1000hz.wav" 等
# ==========================================

# --- 核心算法：20-200Hz 带通滤波器 ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

try:
    # 1. 自动获取桌面路径，无缝对接你的 receive.py 下载位置
    desktop_path = os.path.join(os.environ['USERPROFILE'], 'Desktop')
    filepath = os.path.join(desktop_path, filename)
    
    print(f"📂 正在读取文件: {filepath} ...")
    fs, raw_data = wav.read(filepath)
    
    # 如果是立体声(双声道)，只取第一个声道
    if len(raw_data.shape) > 1:
        raw_data = raw_data[:, 0]
        
    # 2. 计算时间轴
    duration = len(raw_data) / fs
    time = np.linspace(0, duration, len(raw_data))
    
    # 3. 执行滤波
    print("🧹 正在进行 20-200Hz 带通滤波处理...")
    filtered_data = butter_bandpass_filter(raw_data, 20, 200, fs)

    # 4. 开始绘图
    plt.figure(figsize=(12, 8))

    # ================= 图1: 微观波形细节 =================
    plt.subplot(2, 1, 1)
    
    # [需求完成] 已移除灰色的原始信号线，只画出清洗后的深蓝色线
    plt.plot(time, filtered_data, color='#007acc', label='Filtered Signal (20-200Hz)', linewidth=2)
    
    # [需求完成] 🎯 核心改动：设置观察窗口为 6.0秒 到 6.05秒
    start_view = 0 
    end_view = 10.0 
    
    # 安全检查：防止录音太短导致画不出图
    if duration > end_view:
        plt.xlim(start_view, end_view)
    else:
        print(f"⚠️ 提示: 录音总长仅 {duration:.2f}s，不足 6.05s，将自动显示最后 0.05s 画面。")
        plt.xlim(max(0, duration - 0.05), duration)

    plt.title(f"Microscopic View (0.05s Window at 6.0s) - File: {filename}")
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    # ================= 图2: 语谱图 (Spectrogram) =================
    plt.subplot(2, 1, 2)
    # 增大 NFFT 提高低频的清晰度
    Pxx, freqs, bins, im = plt.specgram(filtered_data, NFFT=2048, Fs=fs, noverlap=1024, cmap='magma')
    
    plt.title("Frequency Energy Distribution (Global)")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    
    # 只显示心音关心的核心医学频段 (0-500Hz)
    plt.ylim(0, 500)
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    print("✅ 处理完成，正在显示图表！")
    plt.show()

except FileNotFoundError:
    print(f"\n❌ 找不到文件: {filename}")
    print(f"👉 请确认你的电脑桌面上是否有这个文件。")
except Exception as e:
    print(f"\n❌ 发生未知错误: {e}")