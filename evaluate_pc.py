import os
import sys
import csv
import time
import torch
import numpy as np
import pandas as pd
import yaml
import ai_edge_litert.interpreter as tflite
from sklearn.metrics import classification_report, recall_score
from torch.utils.data import Subset
from pathlib import Path

# === 1. 项目路径与环境配置 ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.train.dataset.dataset_mel import HeartSoundMelDataset

def get_test_subset(dataset, split_csv):
    """
    从指定的 split CSV 文件加载固定 test 集 fname 列表。
    split_csv: test_split.csv（诊断模型）或 quality_test_split.csv（质量模型）
    """
    if not os.path.exists(split_csv):
        raise FileNotFoundError(
            f"找不到 {split_csv}，请先运行对应训练脚本生成 split 文件"
        )

    with open(split_csv, newline="") as f:
        reader = csv.DictReader(f)
        test_fnames = set(row["fname"] for row in reader)

    all_fnames = [dataset.get_fname(i) for i in range(len(dataset))]
    test_indices = [idx for idx, fname in enumerate(all_fnames) if fname in test_fnames]

    print(f"  ✅ 从 {os.path.basename(split_csv)} 加载 test 集，匹配切片数: {len(test_indices)}")
    return Subset(dataset, test_indices)

def evaluate_single_tflite(model_path, subset, target_names):
    """
    通用的 TFLite 推理与指标计算函数
    """
    if not os.path.exists(model_path):
        return None

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_idx = interpreter.get_input_details()[0]['index']
    output_idx = interpreter.get_output_details()[0]['index']
    
    all_preds, all_labels, latencies = [], [], []

    print(f"🚀 正在推理: {os.path.basename(model_path)} (样本数: {len(subset)})")
    
    for i in range(len(subset)):
        mel, label = subset[i]
        input_data = mel.numpy()

        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        start_time = time.perf_counter()
        interpreter.set_tensor(input_idx, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_idx)
        latencies.append(time.perf_counter() - start_time)

        all_preds.append(np.argmax(output_data))
        all_labels.append(label)

    avg_latency = np.mean(latencies) * 1000 
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    
    # M-Score: 两个类别召回率的算术平均值
    m_score = (recall_score(all_labels, all_preds, pos_label=0) + 
               recall_score(all_labels, all_preds, pos_label=1)) / 2

    return {
        "Accuracy": f"{report['accuracy']:.4f}",
        "M-Score": f"{m_score:.4f}",
        "Latency": f"{avg_latency:.2f}ms",
        "Size": f"{os.path.getsize(model_path)/(1024*1024):.2f}MB"
    }

def main():
    # 加载全局配置
    CONFIG_PATH = Path(PROJECT_ROOT) / "config.yaml"
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # 包含 FP32 和 INT8 的任务清单
    tasks = [
        {
            "name": "Diagnosis (疾病诊断)",
            "metadata": os.path.join(PROJECT_ROOT, "data/metadata_physionet.csv"),
            "split_csv": os.path.join(PROJECT_ROOT, "data/test_split.csv"),
            "labels": ['Normal', 'Abnormal'],
            "models": {
                "Diag_FP32": os.path.join(PROJECT_ROOT, "heart_model_fp32.tflite"),
                "Diag_INT8": os.path.join(PROJECT_ROOT, "heart_model_quant.tflite")
            }
        },
        {
            "name": "Quality (质量评估)",
            "metadata": os.path.join(PROJECT_ROOT, "data/metadata_quality_reversed.csv"),
            "split_csv": os.path.join(PROJECT_ROOT, "data/test_split_sqa.csv"),
            "labels": ['Good', 'Bad'],
            "models": {
                "Qual_FP32": os.path.join(PROJECT_ROOT, "heart_quality_fp32.tflite"),
                "Qual_INT8": os.path.join(PROJECT_ROOT, "heart_quality_quant.tflite")
            }
        }
    ]

    results_list = []

    for task in tasks:
        print(f"\n" + "="*50)
        print(f"📊 正在加载任务数据集: {task['name']}")

        full_dataset = HeartSoundMelDataset(
            metadata_path=task["metadata"],
            sr=cfg["data"]["sample_rate"],
            segment_sec=cfg["data"]["segment_length"],
            mel_cfg=cfg["mel"]
        )

        test_subset = get_test_subset(full_dataset, task["split_csv"])
        print(f"✅ 隔离 20% 测试集成功，共 {len(test_subset)} 个切片")

        for model_name, path in task["models"].items():
            metrics = evaluate_single_tflite(path, test_subset, task["labels"])
            if metrics:
                metrics["Task"] = task["name"]
                metrics["Model"] = model_name
                results_list.append(metrics)

    # 打印最终实验结果对比大表
    if results_list:
        df = pd.DataFrame(results_list)
        # 重新排序以便于观察 FP32 与 INT8 的对比
        df = df.sort_values(by=["Task", "Model"])
        print("\n" + "#"*90)
        print("🏆 FypProj 双阶段系统：FP32 vs INT8 综合性能对比报告")
        print("#"*90)
        print(df[["Task", "Model", "Accuracy", "M-Score", "Latency", "Size"]].to_string(index=False))
        print("#"*90)

if __name__ == "__main__":
    main()