"""
gen_test_tone.py — 正弦波测试音生成器（在电脑上运行）

用法：
  python gen_test_tone.py 100          # 生成 100Hz，持续 5 秒
  python gen_test_tone.py 100 200 400  # 一次生成多个频率
  python gen_test_tone.py 100 10       # 生成 100Hz，持续 10 秒
"""

import sys
import wave
import struct
import math

SAMPLE_RATE = 44100   # 播放用高采样率，确保扬声器还原准确
AMPLITUDE   = 0.8     # 振幅（0~1），留一点余量避免削波
DURATION    = 5       # 默认时长（秒）


def gen_tone(freq_hz, duration, sample_rate=SAMPLE_RATE, amplitude=AMPLITUDE):
    n = int(sample_rate * duration)
    samples = []
    for i in range(n):
        val = amplitude * math.sin(2 * math.pi * freq_hz * i / sample_rate)
        samples.append(int(val * 32767))
    return samples


def save_wav(filename, samples, sample_rate=SAMPLE_RATE):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        data = struct.pack(f'<{len(samples)}h', *samples)
        wf.writeframes(data)
    print(f"✅ 已生成: {filename}  ({sample_rate}Hz 采样, {len(samples)/sample_rate:.1f}s)")


def main():
    args = sys.argv[1:]
    if not args:
        print("用法: python gen_test_tone.py <频率Hz> [频率Hz2 ...] [时长s]")
        print("示例: python gen_test_tone.py 100 200 400")
        return

    # 判断最后一个参数是不是时长（大于1000视为频率，否则视为秒数）
    duration = DURATION
    if len(args) >= 2:
        try:
            last = float(args[-1])
            if last <= 1000:
                duration = last
                args = args[:-1]
        except ValueError:
            pass

    for arg in args:
        try:
            freq = float(arg)
        except ValueError:
            print(f"⚠️  跳过无效参数: {arg}")
            continue

        freq_int = int(freq)
        filename = f"{freq_int}hz_og.wav"
        samples  = gen_tone(freq, duration)
        save_wav(filename, samples)


if __name__ == "__main__":
    main()
