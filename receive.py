import asyncio
from bleak import BleakClient, BleakScanner
import struct
import scipy.io.wavfile as wav
import numpy as np
import time
import os
import sys

# --- 配置部分 ---
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
SAMPLE_RATE = 2000
DEVICE_NAME = "ESP32_Steth"

# 缓冲区 (设为全局变量，防止丢失)
audio_buffer = []

def notification_handler(sender, data):
    count = len(data) // 2
    samples = struct.unpack(f'<{count}h', data)
    audio_buffer.extend(samples)
    
    # 打印进度
    if len(audio_buffer) % 8000 < 128:
        print(f"🔴 录制中... {len(audio_buffer)/SAMPLE_RATE:.1f}s | 数据点: {len(audio_buffer)}")

def save_file():
    """强制保存文件的函数"""
    if audio_buffer:
        print("\n" + "="*40)
        print(f"💾 正在紧急保存 {len(audio_buffer)} 个数据点...")
        try:
            audio_np = np.array(audio_buffer, dtype=np.int16)
            filename = f"heart_sound_{int(time.time())}.wav"
            # 获取桌面路径 (兼容不同系统)
            desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
            filepath = os.path.join(desktop, filename)
            
            wav.write(filepath, SAMPLE_RATE, audio_np)
            print(f"✅ 保存成功！位置: {filepath}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
            # 尝试保存在当前目录作为备选
            wav.write(filename, SAMPLE_RATE, audio_np)
            print(f"⚠️ 已保存在脚本同级目录: {filename}")
        print("="*40)
    else:
        print("\n⚠️ 未采集到数据，不保存文件。")

async def main():
    print(f"🔍 正在搜索设备: {DEVICE_NAME} ...")
    
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and DEVICE_NAME in d.name
    )

    if not device:
        print("❌ 未找到设备。请检查 ESP32 是否上电且未连手机。")
        return

    print(f"✅ 连接: {device.name}")

    async with BleakClient(device) as client:
        print(f"🔗 连接成功，等待数据...")
        await client.start_notify(CHAR_UUID, notification_handler)
        
        print("\nSTART RECORDING (按 Ctrl+C 结束)\n")
        
        # 保持运行
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 捕捉 Ctrl+C，不显示红色报错
        pass 
    finally:
        # 【关键】无论怎么退出，最后一定执行这里
        save_file()
        print("程序已安全退出。")