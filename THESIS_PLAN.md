# 论文规划

---

## 论文结构（负责部分）

### 一、预处理与模型训练
- **数据集**：PhysioNet 2016 心音挑战赛
- **预处理流程**：带通滤波（25–400 Hz）→ 2s 滑动窗口（50% overlap）→ Log-Mel 频谱
- **模型**：CNN + Coordinate Attention 模块（有专门设计）
- **SQA**：信号质量评估模型，过滤低质量片段
- **量化**：FP32 → INT8

### 二、嵌入式系统部署
- **系统架构**：ESP32 → BLE → Raspberry Pi 4B → OLED
- **BLE 实时数据接收与缓冲**
- **滑动窗口推理流水线**（双模型：SQA + 诊断）
- **用户交互**：物理按键 + 双 OLED 屏幕设计
- **系统可靠性**：软件看门狗、安全关机

---

## 需要补的实验数据

| 实验 | 工具 | 在哪跑 |
|------|------|--------|
| 单窗口推理延迟（Mel / SQA / 诊断各阶段） | `benchmark.py` | Pi |
| Chunk 总处理时间 vs 实时性验证 | `benchmark.py` | Pi |
| CPU / 内存占用 | `benchmark.py` | Pi |
| 模型文件大小（FP32 vs INT8） | `benchmark.py` | Pi |
| 量化模型准确率（Accuracy / Sensitivity / Specificity / F1） | `evaluate.py` | Pi |
| FP32 vs INT8 准确率对比 | 训练项目跑 FP32，Pi 跑 INT8 | 两边对比 |

---

## 待办

- [ ] 改模型、重新训练
- [ ] 替换 `.tflite` 文件
- [ ] 整理测试集 → 导出 `test_split.csv`
- [ ] Pi 上跑 `benchmark.py` 采延迟/资源数据
- [ ] Pi 上跑 `evaluate.py` 采准确率数据
- [ ] FP32 模型在训练项目那边跑出准确率，与 INT8 对比
