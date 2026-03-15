import asyncio
from bleak import BleakClient
import numpy as np
import ai_edge_litert.interpreter as tflite
import yaml
import os
import sys
import time

# ==========================================
# 1. 环境初始化
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess.preprocess_pipeline import preprocess_wav_for_pi

def softmax(x):
    """Logits 转换为概率"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# ==========================================
# 2. 模型加载
# ==========================================
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

q_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite"))
d_interp = tflite.Interpreter(model_path=os.path.join(PROJECT_ROOT, "heart_model_quant.tflite"))
q_interp.allocate_tensors()
d_interp.allocate_tensors()

# ==========================================
# 3. 蓝牙监控接收逻辑
# ==========================================
# 关键：设置你的 ESP32 信息
ESP32_MAC = "XX:XX:XX:XX:XX:XX"  # 填入你 ESP32 的蓝牙地址
CHARACTERISTIC_UUID = "00002a37-0000-1000-8000-00805f9b34fb"  # 填入 ESP32 对应数据的特征 UUID

received_buffer = bytearray()

def notification_handler(sender, data):
    """当 ESP32 发送数据包时触发的回调"""
    global received_buffer
    received_buffer.extend(data)
    print(f"\r📥 实时接收中: {len(received_buffer)/1024:>7.2f} KB", end="")

async def main():
    global received_buffer
    received_file = os.path.join(PROJECT_ROOT, "received_test.wav")
    
    print(f"📡 正在尝试连接 ESP32: {ESP32_MAC}...")
    
    try:
        async with BleakClient(ESP32_MAC) as client:
            print(f"✅ 已连接! 信号强度: {client.mtu_size} MTU")
            
            # 开始监听数据
            await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
            print("📥 正在同步数据流，请在 ESP32 端开始传输...")
            
            # 监控逻辑：如果连续 5 秒没收到新数据，则认为传输结束
            last_len = 0
            while True:
                await asyncio.sleep(5) 
                if len(received_buffer) == last_len and len(received_buffer) > 0:
                    break
                last_len = len(received_buffer)

            await client.stop_notify(CHARACTERISTIC_UUID)
            
        # 写入文件
        with open(received_file, "wb") as f:
            f.write(received_buffer)
        
        print(f"\n\n💾 接收完毕。总大小: {len(received_buffer)} 字节")
        
        # ==========================================
        # 4. 执行推理
        # ==========================================
        tensors = preprocess_wav_for_pi(received_file, config)
        print(f"🧩 预处理成功：切分出 {len(tensors)} 个片段")
        
        # 择优推理逻辑
        best_score = -1
        best_tensor = None

        for tensor in tensors:
            q_interp.set_tensor(q_interp.get_input_details()[0]['index'], tensor)
            q_interp.invoke()
            q_probs = softmax(q_interp.get_tensor(q_interp.get_output_details()[0]['index'])[0])
            
            if q_probs[1] > best_score: # 1 为 Good Quality
                best_score = q_probs[1]
                best_tensor = tensor

        if best_tensor is not None and best_score > 0.8:
            d_interp.set_tensor(d_interp.get_input_details()[0]['index'], best_tensor)
            d_interp.invoke()
            d_probs = softmax(d_interp.get_tensor(d_interp.get_output_details()[0]['index'])[0])
            
            label = "Normal" if np.argmax(d_probs) == 0 else "Abnormal"
            print(f"✨ 黄金片段结果: {label} | 置信度: {np.max(d_probs):.2%}")
        else:
            print("⚠️ 未发现高质量心音片段。")

    except Exception as e:
        print(f"❌ 推理失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())