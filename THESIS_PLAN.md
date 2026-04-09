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

## 评估指标说明

### 混淆矩阵基础

|  | 预测 Normal | 预测 Abnormal |
|--|------------|--------------|
| **实际 Normal** | TN（真阴性）| FP（假阳性）|
| **实际 Abnormal** | FN（假阴性）| TP（真阳性）|

### Sensitivity（灵敏度 / 召回率）

```
Sensitivity = TP / (TP + FN)
```

回答：**所有真正异常的心音，模型抓住了多少？**

- FN 是"漏报"——本来是异常，模型说是正常，在医疗场景里最危险
- Sensitivity 低 = 模型漏掉了太多异常患者

### Specificity（特异度）

```
Specificity = TN / (TN + FP)
```

回答：**所有真正正常的心音，模型正确识别了多少？**

- FP 是"误报"——本来正常，模型说是异常，会导致不必要的复查
- Specificity 低 = 正常人被过度误诊

### 两者的关系

两者存在**天然的权衡（trade-off）**：
- 模型偏向预测 Abnormal → Sensitivity ↑，Specificity ↓
- 模型偏向预测 Normal → Specificity ↑，Sensitivity ↓
- 对不平衡数据（4:1），不加干预的模型会偏向多数类 Normal，导致 Sensitivity 很低

### M-Score（PhysioNet 2016 官方指标）

```
M-Score = (Sensitivity + Specificity) / 2
```

两者的算术平均，要求模型**在两个方向上都做好**，不能只靠偏向某一类刷高单项指标。这也是我们用 M-Score 而不是 Accuracy 作为模型保存标准的原因。

> Accuracy 在 4:1 不平衡下可以达到 80%（全预测 Normal），但 Sensitivity=0，M-Score 只有 50%。

---

## 训练脚本划分逻辑

### `train_lightweight.py`（开发迭代用）

```
数据集：metadata_physionet.csv（2876 条）
划分：80/20，按 recording（fname）分组，seed=42
Train：~2300 切片 | Val：~575 切片
augment：Train=True，Val=False
类别平衡：WeightedRandomSampler（1/class_count）
保存标准：Val M-Score 最大
Test集：无
```

### `train_lightweight_with_test.py`（最终评估用）

```
数据集：metadata_physionet.csv（2876 条）
划分：80/10/10，按 recording（fname）分组，seed=42
Train：~2300 录音 / ~49833 切片 | Val：~288 录音 / ~5897 切片 | Test：~288 录音 / ~6273 切片
augment：Train=True，Val/Test=False
类别平衡：WeightedRandomSampler（1/class_count）
保存标准：Val M-Score 最大
Test集：训练结束后加载最优模型跑一次，输出 M-Score
持久化：首次运行将 test fname 列表写入 data/test_split.csv（已存在则跳过）
```

### `train_sqa_with_test.py`（SQA 模型专用）

```
数据集：metadata_quality_reversed.csv（3240 条，Good=2876 / Bad=364，标签已反转）
划分：80/10/10，按 recording（fname）分组，seed=42
Train：~2592 录音 / ~54842 切片 | Val：~324 录音 / ~6536 切片 | Test：~324 录音 / ~6726 切片
augment：Train=True，Val/Test=False
类别平衡：WeightedRandomSampler（1/class_count，8:1 不平衡）
保存标准：Val M-Score 最大
Test集：训练结束后加载最优模型跑一次，输出 M-Score
持久化：首次运行将 test fname 列表写入 data/test_split_sqa.csv（已存在则跳过）
```

### `evaluate_tflite.py`（Pi 上跑，最终指标）

```
诊断模型：读取 data/test_split.csv（由 train_lightweight_with_test.py 生成）
质量模型：读取 data/test_split_sqa.csv（由 train_sqa_with_test.py 生成）
对各模型加载 FP32 和 INT8 两个 tflite 文件分别评估
输出：Accuracy / M-Score / 推理延迟 / 模型大小 对比表
注意：质量模型 test 集包含 Bad Quality 录音，评估结果更可信
```

---

## 数据集统计

### 诊断模型（metadata_physionet.csv）

| 类别 | 数量 | 占比 |
|------|------|------|
| Normal (0) | 2304 | 80.1% |
| Abnormal (1) | 572 | 19.9% |
| **总计** | **2876** | **4:1** |

### SQA模型（metadata_quality.csv）

| 类别 | 数量 | 占比 |
|------|------|------|
| Good (1) | 2876 | 88.8% |
| Bad (0) | 364 | 11.2% |
| **总计** | **3240** | **8:1** |

---

## 已发现的问题

### 🔴 必须修复（影响训练结果可信度）

1. **验证集用错了数据集对象** (`train_lightweight.py:188`)
   - `val_ds` 是从 `train_dataset`（`augment=True`）取的子集，而创建的 `val_dataset`（`augment=False`）从未被使用
   - 导致验证集被加了数据增强，验证指标不可信
   - 修复：`val_ds = Subset(val_dataset, val_indices)`（参考 `train_quality.py` 的写法）

2. **诊断模型没有处理类别不平衡** (`train_lightweight.py`)
   - PhysioNet 2016 Normal/Abnormal 分布不均，使用普通 `CrossEntropyLoss` + 随机采样
   - 模型会偏向多数类 Normal，Sensitivity（异常检出率）偏低
   - 修复：加 `WeightedRandomSampler` 或对 loss 加类别权重（参考 `train_quality.py`）

3. **缺少独立 Test 集**
   - 目前只有 train/val 划分，val 集既用于调参又用于报告指标
   - 多次迭代后模型已间接拟合 val 集，最终指标偏乐观
   - 修复：划分出固定 test 集（10%），锁住 fname 列表存文件，只在最终汇报时跑一次

### 🟡 建议改进（影响模型性能）

4. **模型保存标准用 Accuracy** (`train_lightweight.py:241`)
   - 对不平衡数据，Accuracy 不代表诊断性能，可能保存了 Sensitivity 很差的模型
   - 建议改为 macro-F1 或 Sensitivity 作为保存标准

5. ⏸️ **【待定】CoordAtt 挂在每个 DSConv 块上** (`lightweight_cnn.py`)
   - 共 3 个 block 都有 CoordAtt，对 32×64 的小输入偏重
   - 若准确率够用 → 只放后两层，减少参数量，对量化更友好
   - 若准确率不足 → 反向考虑加深模型，见"模型复杂度提升方向"

6. ⏸️ **【待定】末层通道数 256 偏大** (`lightweight_cnn.py`)
   - 若准确率够用 → 改为 128，减小模型体积
   - 若准确率不足 → 可扩到 512，增加模型容量

### 🟢 可选优化

7. **没有残差连接**：加 skip connection 可提升训练稳定性，量化后精度保持更好

---

## 模型复杂度提升方向（Pi 4B 推理余量充足）

> **前提**：先修完数据问题（类别平衡 + 保存标准）重训一次，确认 Sensitivity 确实不足再动架构。
> 数据量仅 2876 条录音，过深模型容易过拟合，加复杂度要配合数据增强。

### Pi 4B 实际推理余量参考

| 模型 | 参数量 | Pi 4B INT8 推理延迟（约） |
|------|--------|--------------------------|
| 当前 CNN + CoordAtt | ~65K | <10ms |
| MobileNetV2 | 3.4M | ~50ms |
| EfficientNet-B0 | 5.3M | ~100ms |
| ResNet-18 | 11M | ~200ms |

窗口 2s、50% overlap，每秒处理一个窗口，预算 1000ms，余量非常充足。

### 推荐优先级

**第一步（小改动，先试）**
- 加残差连接（skip connection）：训练更稳，量化精度保持更好
- 多加一个 DSConv block，末层通道可扩到 512

**第二步（中改动，如果第一步不够）**
- 将 CoordAtt 替换为 SE block（Squeeze-and-Excitation）：更轻量，量化更友好，有大量医疗音频论文支撑
- 或保留 CoordAtt 但只放后两层

**第三步（换 backbone，如果准确率仍不足）**
- 换 MobileNetV2：TFLite 官方支持，量化成熟，有大量心音分类论文参考，参数量适中不易过拟合

### 不推荐的方向
- **ViT / DeiT**：attention 量化损失大，TFLite 算子支持有限，32×64 小输入发挥不出优势
- **DiT**：图像生成模型，与分类任务无关
- **Mamba**：依赖自定义 CUDA kernel，Pi CPU 无高效实现，TFLite 不支持

### 消融实验结论（已完成）

四组累进式消融，固定预处理参数（n_mels=32, hop=96）和训练参数（batch=16, lr=1e-3, wd=1e-4）：

| 配置 | 参数量 | Test M-Score | Test Se | Test Sp |
|------|--------|-------------|---------|---------|
| A: Baseline OG（16→128，无注意力） | 12.87K | 0.8851 | 0.9654 | 0.8049 |
| B: + 加宽通道（32→256） | 47.23K | 0.8896 | 0.9595 | 0.8198 |
| C: + CoordAtt + Dropout | 65.12K | 0.8869 | 0.9383 | 0.8355 |
| D: + 残差连接 | 108.10K | 0.8912 | 0.9797 | 0.8027 |

**最终选定 C 组**：Se/Sp 最平衡，训练最稳定（最佳 epoch=5 vs D 组的 2），适合医疗筛查场景。D 组虽 M-Score 最高但 Se/Sp 失衡加剧，不采用。详细分析见 MODEL_FINETUNE.md。

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

### 诊断模型
- [x] 修复 val_ds 数据集对象错误（`train_lightweight.py:188`）
- [x] 诊断模型加类别平衡（WeightedRandomSampler）
- [x] 划分并固定 test 集 → 导出 `test_split.csv`，锁住不动
- [x] 模型保存标准改为 M-Score（Se + Sp）/ 2
- [x] 消融实验完成（A/B/C/D 四组），最终选定 C 组（CoordAtt + Dropout，65.12K），不加残差
- [x] 最终模型确定：Run 6（n_mels=64, hop=128, batch=16），Test M-Score=0.8903
- [x] 将 Run 6 的 `best_model.pth` 用 `scripts/convert_to_tflite.py` 转为 `.tflite`（FP32 + INT8）

### SQA 模型
- [x] 划分并固定 SQA test 集 → `data/test_split_sqa.csv`
- [x] SQA Run-1 完成（Test M-Score=0.8046，Se=0.7173，Se 偏低）
- [x] SQA Run-2 完成（class_weight=[1,8] + lr=5e-4，Test Se=0.7651，M-Score=0.8102）
- [x] SQA Run-3 完成（dropout=0.5，Test Se=0.8274，M-Score=0.8152）**← 最终选定**
- [x] 阈值扫描不需要：部署机制为加权平均，P(Good) 直接作为权重，无二值阈值
- [x] 用 `scripts/convert_to_tflite.py` 将 Run-3 `best_model_sqa.pth` 转为 `.tflite`（FP32 + INT8）
- [ ] 替换 Pi 上的四个 `.tflite` 文件（诊断 FP32/INT8 + SQA FP32/INT8）

### Pi 端评估
- [ ] Pi 上跑 `benchmark.py` 采延迟/资源数据
- [ ] Pi 上跑 `evaluate.py` 采准确率数据（FP32 vs INT8）
- [ ] FP32 模型在训练端跑出准确率，与 Pi INT8 结果对比

