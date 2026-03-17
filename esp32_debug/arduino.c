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

// 音频参数
#define SAMPLE_RATE     8000  // I2S 采样率（受 INMP441 SCK 下限约束，不可低于此值）
#define OUTPUT_RATE     2000  // BLE 输出采样率（模型输入要求）
#define DECIMATE        4     // 降采样倍数 SAMPLE_RATE / OUTPUT_RATE
#define BLOCK_SIZE      128   // I2S DMA 缓冲区大小（字节）
#define OUT_BLOCK_BYTES 128   // BLE 单包大小（字节）= 64 samples @ 2000Hz

// BLE UUID
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// 全局变量
BLEServer *pServer = NULL;
BLECharacteristic *pCharacteristic = NULL;
bool deviceConnected = false;
bool oldDeviceConnected = false;

// I2S 输入缓冲（64 samples @ 8000Hz）
int16_t sBuffer[BLOCK_SIZE / 2];

// BLE 输出缓冲（64 samples @ 2000Hz）
int16_t outBuffer[OUT_BLOCK_BYTES / 2];
int outIndex = 0;

// 抗混叠 LPF 状态（2 阶 IIR 级联）
// alpha = 1 - exp(-2π * fc / fs)，fc=800Hz, fs=8000Hz → alpha ≈ 0.47
// 截止 ~800Hz，Nyquist（1000Hz）处衰减约 -7.6dB，足以抑制心音频段外混叠
float lpf1 = 0.0f;
float lpf2 = 0.0f;
const float LPF_ALPHA = 0.47f;

// 抽取计数器
int decimateCount = 0;

// ================= 2. 辅助函数 =================

// 直流去除：滑动平均估计直流偏置并扣除
int16_t remove_dc_offset(int16_t sample) {
    static long long sum = 0;
    static int count = 0;
    static const int WINDOW_SIZE = 1000;
    sum += sample;
    count++;
    if (count > WINDOW_SIZE) {
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

// BLE 回调：处理连接与连接速度优化
class MyServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer, esp_ble_gatts_cb_param_t *param) {
        deviceConnected = true;
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

    i2s_install();
    i2s_setpin();
    i2s_start(I2S_PORT);

    BLEDevice::init("ESP32_Steth");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

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

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(false);
    pAdvertising->setMinPreferred(0x0);
    BLEDevice::startAdvertising();
    Serial.println("Waiting for connection...");
}

void loop() {
    size_t bytesIn = 0;
    esp_err_t result = i2s_read(I2S_PORT, &sBuffer, sizeof(sBuffer), &bytesIn, portMAX_DELAY);

    if (result == ESP_OK && bytesIn > 0) {
        int samples_read = bytesIn / 2;

        for (int i = 0; i < samples_read; ++i) {
            // 1. 去直流
            float x = (float)remove_dc_offset(sBuffer[i]);

            // 2. 抗混叠低通滤波（2 阶 IIR 级联，截止 ~800Hz @ 8000Hz）
            lpf1 = LPF_ALPHA * x    + (1.0f - LPF_ALPHA) * lpf1;
            lpf2 = LPF_ALPHA * lpf1 + (1.0f - LPF_ALPHA) * lpf2;

            // 3. 4:1 抽取：每 4 个样本保留 1 个
            decimateCount++;
            if (decimateCount >= DECIMATE) {
                decimateCount = 0;
                outBuffer[outIndex++] = (int16_t)lpf2;

                // 4. 输出缓冲满 64 samples → BLE 发送
                if (outIndex >= OUT_BLOCK_BYTES / 2) {
                    outIndex = 0;
                    if (deviceConnected) {
                        pCharacteristic->setValue((uint8_t*)outBuffer, OUT_BLOCK_BYTES);
                        pCharacteristic->notify();
                    }
                }
            }
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
