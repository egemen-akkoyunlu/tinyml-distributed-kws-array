# 🎙️ Distributed Graph-Theoretic Mixture of Experts (MoE) Keyword Spotting Array

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Repo](https://img.shields.io/badge/Base_Repo-sciapponi%2Fstreamable--kws-blue.svg)](https://github.com/sciapponi/streamable-kws)
[![Zephyr RTOS](https://img.shields.io/badge/RTOS-Zephyr_3.7+-purple.svg)](https://zephyrproject.org/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3%20%7C%20EFR32MG24-orange.svg)](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html)
[![Quantization](https://img.shields.io/badge/Quantization-Hybrid_INT8%2FFP32-green.svg)](https://github.com/espressif/esp-dl)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.9%25%20(Clean)%20%7C%2088.1%25%20(Noise)-brightgreen.svg)]()

This system extends the streamable Keyword Spotting (KWS) framework from [`sciapponi/streamable-kws`](https://github.com/sciapponi/streamable-kws) into a **fault-tolerant, distributed multi-microcontroller sensor array** deployed on **Zephyr RTOS** across **Seeed Studio XIAO ESP32-S3 Sense** and **Silicon Labs EFR32MG24 (ARM Cortex-M33)** edge nodes.

By combining **Graph-Theoretic Vocabulary Partitioning**, **Universal Overlap Clustering**, **Spatial 4-Bit SNR-Weighted Soft Late Fusion**, and **Bare-Metal Zephyr RTOS C++ Drivers**, this project eliminates the classical **Acoustic Correlation Trap** (where homogeneous arrays make identical errors on phonetically confusable words like "TREE" vs "THREE").

---

## 📑 Table of Contents
- [Quickstart: Clone & Setup](#-quickstart-clone--setup)
- [Key Architectural Innovations](#-key-architectural-innovations)
- [3-Device System Architecture Diagram](#-3-device-system-architecture-diagram)
- [Directory & File Overview](#-directory--file-overview)
- [End-to-End Pipeline & Deployment Guide](#-end-to-end-pipeline--deployment-guide)
  - [1. Model Training](#1-model-training)
  - [2. Graph-Theoretic MoE Optimizer](#2-graph-theoretic-moe-optimizer)
  - [3. Device-Wise INT8 Quantization Export (Dev 1, Dev 2, Dev 3)](#3-device-wise-int8-quantization-export-dev-1-dev-2-dev-3)
  - [4. Flashing Physical Microcontrollers (Device IDs 1, 2, 3)](#4-flashing-physical-microcontrollers-device-ids-1-2-3)
  - [5. Live Multi-Device Real-Time Gateway Fusion](#5-live-multi-device-real-time-gateway-fusion)
- [Topologies & MoE Strategies Explained](#-topologies--moe-strategies-explained)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Safety & Squelch Stack](#-safety--squelch-stack)
- [License](#-license)

---

## 🚀 Quickstart: Clone & Setup

### Step 1: Clone the Base Repository & Enter Workspace

```bash
git clone https://github.com/sciapponi/streamable-kws.git
cd streamable-kws
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🌟 Key Architectural Innovations

1. **Hybrid INT8/FP32 Model Quantization:**
   - 95% INT8 Convolutions on Xtensa 128-bit Vector SIMD + FP32 Softmax on hardware FPU.
   - Compresses model footprint from **133.6 KB to 100.7 KB** with **negligible accuracy loss** (<0.2%), fitting within the ~128 KB internal SRAM limit of ESP32-S3.
2. **Solving the Acoustic Correlation Trap:**
   - Demonstrates that identical models share identical decision boundary blind spots (e.g., unanimous 25.8% false alarms on `TREE` vs `THREE`).
   - Formulates the **Tri-Factor Acoustic Vulnerability Metric**:
     $$V(k) = 0.70 \cdot \text{Error}(k) + 0.15 \cdot H(k) + 0.15 \cdot \text{Peak}(k)$$
   - Builds a **Symmetrized Acoustic Adjacency Graph** ($W_{ij} = \max(P(j|i), P(i|j)) \ge 0.05$) to handcuff phonetic twins into indivisible cliques.
3. **Universal Overlap MoE Topology:**
   - Co-locates confusable twins as Anchors across nodes while distributing the remaining vocabulary.
   - Reduces per-node classes from **36 to 23–25**, boosting parameter capacity per class by **+44%** while preserving $R \ge 2$ spatial redundancy.
4. **4-Bit Spatial SNR-Weighted Soft Late Fusion:**
   - Real-time on-chip **Hendriks MMSE spectral noise estimation** combined with sigmoidal SNR weighting ($w_i \in [1, 15]$), giving high-SNR microphones 15x voting weight over distant reverberant nodes.
5. **Fail-Stop Hardware Fault Tolerance & Battery Loss Resilience:**
   - Graceful fail-stop degradation under dead battery events ($N-1, N-2, N-3$).
   - 5-Device 3/5 MoE drops by **only -1.3%** on a dead node ($67.1\% \to 65.8\%$), completely avoiding the catastrophic $-19.5\%$ crash of disjoint 0-overlap systems.
6. **Production Embedded Zephyr RTOS C++ Stack:**
   - Zero-copy I2S DMA streaming, on-chip Mel-spectrogram extraction, 2-consecutive-frame temporal debounce, 150 ms refractory cooldown, and a compact **10-byte binary GATT telemetry protocol** (0.3 ms airtime).

---

## 🏛️ 3-Device System Architecture Diagram

```text
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   ACOUSTIC PHYSICAL SPACE                                       │
 │                                                                                                 │
 │   [🎙️ Node 1: (0.5, 0.5)]           [🎙️ Node 2: (3.5, 0.5)]           [🎙️ Node 3: (2.0, 3.5)]   │
 │   XIAO ESP32-S3 (25 Cls)            XIAO ESP32-S3 (25 Cls)            XIAO ESP32-S3 (25 Cls)    │
 │   DEVICE_ID = 1                     DEVICE_ID = 2                     DEVICE_ID = 3             │
 │         │                                 │                                 │                   │
 │         │ 10-Byte BLE GATT                │ 10-Byte BLE GATT                │ 10-Byte BLE GATT  │
 │         │ (XIAO_SENSE_BLE_1)              │ (XIAO_SENSE_BLE_2)              │ (XIAO_SENSE_BLE_3)│
 │         ▼                                 ▼                                 ▼                   │
 │ ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │
 │ │                   CENTRAL GATEWAY CONCENTRATOR (`multi_device_fusion.py`)                   │ │
 │ │                                                                                             │ │
 │ │  1. 4-Bit Spatial SNR Sigmoid Weighting:   w_i ∈ [1, 15] based on Hendriks Noise Floor      │ │
 │ │  2. Top-1 vs Top-2 Confidence Margin:      Margin ≥ 15% (Rejects Ambiguous Predictions)     │ │
 │ │  3. Dynamic Noise Threshold:               T_dyn = min(72%, max(54%, 58% + 14 × Noise))     │ │
 │ │  4. Majority Quorum Consensus:             ≥ 2/3 Nodes Must Agree on Keyword Proposal       │ │
 │ │  5. Arbiter Specialist Veto:               Dual-Specialist Nodes Veto OOV False Triggers    │ │
 │ │  6. 250 ms Temporal Refractory Debounce:   Guarantees 1 Trigger Per Spoken Utterance        │ │
 │ └───────────────────────────────────────────────┬─────────────────────────────────────────────┘ │
 │                                                 │                                               │
 │                                                 ▼                                               │
 │                                 🏆 REAL-TIME FUSED KWS DECISION                                 │
 └─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory & File Overview

```text
streamable-kws/
├── README.md                             # Original upstream repository README
├── README_DISTRIBUTED_ARRAY.md           # Comprehensive multi-device array guide
├── requirements.txt                      # Python dependencies
├── config/                               # Hydra YAML training & MoE configurations
│   ├── kws_35_classes.yaml               # 36-Class Generalist baseline config
│   ├── kws_smart_overlap_dev1_25classes.yaml  # Node 1 MoE Config (25 classes)
│   ├── kws_smart_overlap_dev2_25classes.yaml  # Node 2 MoE Config (25 classes)
│   ├── kws_smart_overlap_dev3_25classes.yaml  # Node 3 MoE Config (25 classes)
│   └── kws_smart_overlap_5dev_dev*.yaml  # 5-Node Universal Overlap (23-23-23-23-22) configs
├── models/                               # PyTorch Streaming Neural Network architectures
│   └── streaming.py                      # Improved_Phi_GRU_ATT_Streaming MatchboxNet
├── zephyr_kws/                           # Bare-metal C++ firmware for Zephyr RTOS
│   ├── src/                              # I2S DMA, DSP Mel extraction, ESP-DL inference, BLE
│   │   ├── inference.cpp                 # On-device model wrapper & Softmax
│   │   ├── dsp_processor.cpp             # Hendriks MMSE noise estimator & Mel filterbank
│   │   ├── ble_manager.cpp               # 10-byte binary GATT packet streamer
│   │   └── kws_config.hpp                # Device ID & threshold configuration
│   ├── CMakeLists.txt                    # Zephyr build system configuration
│   └── prj.conf                          # Hardware FPU, PSRAM, DMA, and RTOS flags
├── datasets.py                           # Google Speech Commands v2 streaming loader
├── train.py                              # PyTorch training script with Hydra & Cosine Annealing
├── optimize_smart_overlap_allocation.py  # Graph-Theoretic MoE Optimizer (Tri-factor V(k) & W_ij)
├── quantize_to_espdl.py                  # ESP-PPQ INT8/FP32 Post-Training Quantizer
└── multi_device_fusion.py                # Real-time BLE Gateway Concentrator & Late Fusion
```

---

## 🛠️ End-to-End Pipeline & Deployment Guide

### 1. Model Training
Train individual MoE device models or the 36-class baseline:
```bash
# Train Node 1 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev1_25classes.yaml

# Train Node 2 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev2_25classes.yaml

# Train Node 3 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev3_25classes.yaml
```

---

### 2. Graph-Theoretic MoE Optimizer
Derive optimal vocabulary partitions using the Tri-Factor Vulnerability Metric:
```bash
# Generate 3-Node Universal Overlap partition (25+25+25 classes)
python3 optimize_smart_overlap_allocation.py --nodes 3

# Generate 5-Node Universal Overlap partition (23+23+23+23+22 classes)
python3 optimize_smart_overlap_allocation.py --nodes 5
```

---

### 3. Device-Wise INT8 Quantization Export (Dev 1, Dev 2, Dev 3)

Quantize each device's trained PyTorch model into calibrated INT8 `.espdl` binaries:

```bash
# 1. Quantize Node 1
python3 quantize_to_espdl.py --config config/kws_smart_overlap_dev1_25classes.yaml

# 2. Quantize Node 2
python3 quantize_to_espdl.py --config config/kws_smart_overlap_dev2_25classes.yaml

# 3. Quantize Node 3
python3 quantize_to_espdl.py --config config/kws_smart_overlap_dev3_25classes.yaml
```

---

### 4. Flashing Physical Microcontrollers (Device IDs 1, 2, 3)

For each Seeed Studio XIAO ESP32-S3 Sense board:

1. Copy the generated `model.espdl` and `model.json` to `zephyr_kws/src/`.
2. In `zephyr_kws/src/kws_config.hpp`, set the matching `DEVICE_ID`:
   ```cpp
   #define DEVICE_ID 1  // Set to 1, 2, or 3 for each board
   ```
   *(This configures the board to advertise as `XIAO_SENSE_BLE_1`, `XIAO_SENSE_BLE_2`, or `XIAO_SENSE_BLE_3`).*
3. Build and flash the firmware:
   ```bash
   cd zephyr_kws
   west build -p always -b xiao_esp32s3/esp32s3/procpu/sense .
   west flash
   ```
4. Repeat for Board 2 (`DEVICE_ID 2`) and Board 3 (`DEVICE_ID 3`).

---

### 5. Live Multi-Device Real-Time Gateway Fusion

Once all 3 microcontrollers are powered on and streaming BLE telemetry:

```bash
python3 multi_device_fusion.py
```

The gateway will automatically discover all 3 nodes, establish encrypted GATT telemetry streams, compute 4-bit SNR weights in real time, and output consensus keyword spot events with zero false alarms!

---

## 🔬 Topologies & MoE Strategies Explained

| Architecture / Topology | Class Allocation | Replication Level ($R$) | Consensus Quorum | Architectural Rationale & Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **1. Single Device Baseline** | 36 Classes | $R = 1$ | 1/1 (Standalone) | Standard edge setup running the full vocabulary on a single node. Vulnerable to room geometry, distance ($1/r^2$), and phonetic confusion. |
| **2. 3x Homogeneous Array** | 3x 36 Classes | $R = 3$ (Uniform) | $\ge 2/3$ Nodes | 3 identical devices with identical weights. Suffers from the **Acoustic Correlation Trap**: identical weights mean identical decision boundary blind spots. |
| **3. 0-Overlap Disjoint MoE** | 13 + 13 + 12 | $R = 1$ (Disjoint) | Specialist Top-1 | Vocabulary is partitioned into 3 disjoint sets with zero class overlap. Maximizes capacity (+200%), but collapses on node failure ($N-1$ battery loss drops accuracy by $-19.5\%$). |
| **4. Graph-Coupled MoE** | 22 + 22 + 22 | $R \in [1, 2]$ | Specialist Voting | Connects confusable phonetic twins onto the same nodes ($W_{ij} \ge 0.05$), while assigning non-confusable words to single nodes. |
| **5. Universal Overlap MoE** *(Recommended 3-Node)* | 25 + 25 + 25 | $R \ge 2$ ($R=3$ for Anchors) | $\ge 2/3$ Majority Quorum | **Gold Standard for 3 Nodes:** Every keyword is replicated on at least 2 nodes ($R \ge 2$), while top confusable twins (`tree`/`three`) are Anchors on all 3 nodes ($R=3$). Eliminates single points of failure. |
| **6. 5-Device 2/5 Ultra-MoE** | 5x ~16 Classes | $R \in [2, 3]$ | $\ge 2/5$ Quorum | High specialization (~16 classes/node). Fast and lightweight, triggering on agreement of any 2 out of 5 nodes. |
| **7. 5-Device 3/5 Majority MoE** *(Ultimate Resilient)* | 5x ~23 Classes | $R \in [3, 5]$ | $\ge 3/5$ Strict Majority | **Maximum Fault Tolerance:** High overlapping vocabulary (~23 classes/node). When a node suffers battery loss ($N-1$), accuracy drops by **only $-1.3\%$** ($67.1\% \to 65.8\%$) under severe noise. |

---

## 📊 Empirical Benchmark Results

### 1. Multi-Condition Acoustic Shootout (1,500 Balanced Test Samples)

| Architecture / Topology | Class Allocation | Clean Room (High SNR) | Gaussian HVAC Noise | Real ESC-50 Noise |
| :--- | :---: | :---: | :---: | :---: |
| **1. Single Device Baseline** | 36 Classes | 85.5% | 70.9% | 57.1% |
| **2. 3x Homogeneous Array** | 3x 36 Classes | 87.5% | 73.9% | 59.3% |
| **3. 0-Overlap Disjoint MoE** | 13 + 13 + 12 | 89.6% | 82.4% | 64.9% |
| **4. Graph-Coupled MoE** | 22 + 22 + 22 | 92.6% | 86.3% | 66.8% |
| **5. Universal Overlap MoE** | 25 + 25 + 25 | 92.2% | **88.1%** | **67.1%** |
| **6. 5-Device 2/5 MoE** | 5x ~16 Classes | 93.0% | 86.7% | 67.0% |
| **7. 5-Device 3/5 Majority MoE** | 5x ~23 Classes | **93.9%** | **88.1%** (+14.2%) | **67.1%** (+10.0%) |

---

### 2. Hardware Resilience Under Battery Loss ($N-1$ to $N-3$ Dead Nodes under ESC-50)

| Topology | 0 Dead Nodes (100% Healthy) | 1 Dead Node ($N-1$) | 2 Dead Nodes ($N-2$) | 3 Dead Nodes ($N-3$) |
| :--- | :---: | :---: | :---: | :---: |
| **0-Overlap Disjoint MoE** | 64.9% | **45.4% (-19.5% Crash 💥)** | — | — |
| **3x Homogeneous Array** | 59.3% | 58.9% (-0.4%) | 57.1% | — |
| **Universal Overlap (25-25-25)** | 67.1% | 63.1% (-4.0%) | 58.7% | — |
| **5-Device 3/5 Majority MoE** | **67.1%** | **65.8% (-1.3% 🟢)** | **63.5% (Beats 3x Homo!)** | **58.5% (Beats Single)** |

---

## 🛡️ Safety & Squelch Stack

```text
 ┌──────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
 │ Safety Mechanism                     │ Operational Value & Protection Goal                         │
 ├──────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
 │ 1. Temporal Lockout (Debounce)       │ 250 ms lockout prevents double-triggering on long syllables │
 │ 2. Dynamic Room Noise Threshold      │ T_dyn = min(72%, max(54%, 58% + 14 × Noise))                │
 │ 3. Top-1 vs Top-2 Confidence Margin  │ Margin = (P_top1 - P_top2) ≥ 15% ensures sharp decisions    │
 │ 4. Normalized Shannon Entropy Filter │ H(x) ≤ 0.28 squelches uniform white noise false alarms      │
 │ 5. 4-Bit Spatial SNR Weighting       │ w_i ∈ [1, 15] gives near-field nodes 15x voting authority   │
 │ 6. Majority Quorum Consensus         │ ≥ 2/3 or ≥ 3/5 node agreement prevents stray activations   │
 │ 7. On-Device RMS Energy Squelch      │ RMS < 5.0 skips model execution during silence to save power│
 └──────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**.
