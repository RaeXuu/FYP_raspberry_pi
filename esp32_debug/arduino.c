#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "driver/i2s.h"

// ================= 1. 定义部分 =================

// 麦克风引脚定义 (ESP32-C3)
#define I2S_MIC_WS  6
#define I2S_MIC_SD  5
#define I2S_MIC_BCK 4
#define I2S_PORT I2S_NUM_0

// --- 算法参数优化 ---
float filtered_val = 0;      
const float alpha = 0.05;    // 低通滤波系数，保留 20-400Hz 关键心音
const int digital_gain = 30; // 数字增益倍数

// 音频参数
#define SAMPLE_RATE 8000 
#define BLOCK_SIZE 128   // 蓝牙单次包大小 (Bytes)

// BLE UUID
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// 全局变量
BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// 音频缓冲区 (16bit = 2 bytes)
int16_t sBuffer[BLOCK_SIZE / 2]; 

// ================= 2. 辅助函数 =================

// 直流去除算法：通过滑动平均计算直流偏置并扣除
int16_t remove_dc_offset(int16_t sample) {
    static long long sum = 0;
    static int count = 0;
    static const int WINDOW_SIZE = 1000;
    
    sum += sample;
    count++;
    if(count > WINDOW_SIZE) {
        sum -= (sum / WINDOW_SIZE);
        count = WINDOW_SIZE;
    }
    int16_t dc_offset = sum / count;
    return sample - dc_offset;
}

// I2S 初始化：配置 INMP441 采样模式
void i2s_install() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = BLOCK_SIZE, 
        .use_apll = false
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
}

void i2s_setpin() {
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_BCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD
    };
    i2s_set_pin(I2S_PORT, &pin_config);
}

// BLE 服务器回调类：处理连接和连接速度优化
class MyServerCallbacks : public BLEServerCallbacks {
    // 修复报错：使用带 param 的 onConnect 从而获取对方 MAC 地址
    void onConnect(BLEServer* pServer, esp_ble_gatts_cb_param_t *param) {
        deviceConnected = true;
        // 关键优化：连接成功后立即请求提速 (间隔 7.5ms - 15ms)
        pServer->updateConnParams(param->connect.remote_bda, 6, 12, 0, 100);
    };

    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
    }
};

// ================= 3. 主程序 =================

void setup() {
  Serial.begin(115200);
  Serial.println("Starting BLE Stethoscope...");

  // 初始化硬件 I2S
  i2s_install();
  i2s_setpin();
  i2s_start(I2S_PORT);

  // 初始化蓝牙
  BLEDevice::init("ESP32_Steth"); 
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  // 创建服务和特征
  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
                      CHARACTERISTIC_UUID,
                      BLECharacteristic::PROPERTY_READ   |
                      BLECharacteristic::PROPERTY_WRITE  |
                      BLECharacteristic::PROPERTY_NOTIFY |
                      BLECharacteristic::PROPERTY_INDICATE
                    );

  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  // 开始广播
  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(false);
  pAdvertising->setMinPreferred(0x0);
  BLEDevice::startAdvertising();
  Serial.println("Waiting for connection...");
}

void loop() {
    size_t bytesIn = 0;
    // 从 I2S 读取原始数据
    esp_err_t result = i2s_read(I2S_PORT, &sBuffer, sizeof(sBuffer), &bytesIn, portMAX_DELAY);

    if (result == ESP_OK && bytesIn > 0) {
        int samples_read = bytesIn / 2;

        // 音频数字信号处理 (DSP)
        for (int i = 0; i < samples_read; ++i) {
            // 1. 去直流
            int16_t current_sample = remove_dc_offset(sBuffer[i]);
            
            // 2. 低通滤波
            filtered_val = (filtered_val * (1.0 - alpha)) + (current_sample * alpha);
            
            // 3. 数字增益
            int32_t amplified = (int32_t)(filtered_val * digital_gain);

   //         if (amplified < 800 && amplified > -800) { 
   //     amplified = 0; 
   // } 
   // else {
        // 如果有信号，稍微减去一点固定底噪，让波形从 0 开始起步
    //    if(amplified > 0) amplified -= 800;
    //    else amplified += 800;
   // }

    // 5. 限幅保护
    if (amplified > 32000) amplified = 32000;
    else if (amplified < -32000) amplified = -32000;

            sBuffer[i] = (int16_t)amplified;
        }

        // 蓝牙发送音频流
        if (deviceConnected) {
            pCharacteristic->setValue((uint8_t*)sBuffer, bytesIn);
            pCharacteristic->notify();
            
        }
        Serial.println(sBuffer[0]);
    }
    
    // 连接状态管理
    if (!deviceConnected && oldDeviceConnected) {
        delay(500); 
        pServer->startAdvertising(); 
        Serial.println("Re-advertising...");
        oldDeviceConnected = deviceConnected;
    }
    
    if (deviceConnected && !oldDeviceConnected) {
        Serial.println("Device Connected");
        oldDeviceConnected = deviceConnected;
    }
}
