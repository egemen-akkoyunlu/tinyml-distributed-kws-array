# 🎙️ Distributed Smart-Overlap Keyword Spotting (KWS) Sensor Array

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12.3-blue.svg)](https://www.python.org/)
[![Base Repo](https://img.shields.io/badge/Base_Repo-sciapponi%2Fstreamable--kws-blue.svg)](https://github.com/sciapponi/streamable-kws)
[![Zephyr RTOS](https://img.shields.io/badge/RTOS-Zephyr_3.7+-purple.svg)](https://zephyrproject.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Seeed_XIAO_ESP32--S3_Sense-orange.svg)](https://www.seeedstudio.com/XIAO-ESP32S3-Sense-p-5639.html)
[![Quantization](https://img.shields.io/badge/Quantization-Hybrid_INT8%2FFP32-green.svg)](https://github.com/espressif/esp-dl)
[![Power](https://img.shields.io/badge/Power-64.8_mA_%7C_240_mW-yellow.svg)]()
[![Latency](https://img.shields.io/badge/Latency-46.3_ms_(Inf)_%7C_1.4_ms_(DSP)-blue.svg)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-93.9%25%20(Clean)%20%7C%2088.1%25%20(Noise)-brightgreen.svg)]()

An end-to-end, fault-tolerant **Distributed TinyML Acoustic Sensor Network** for streaming Keyword Spotting (KWS) across resource-constrained edge microcontrollers (**Seeed Studio XIAO ESP32-S3 Sense**).

> [!NOTE]
> **Special Thanks & Upstream Foundation:**  
> This project is proudly built upon the foundational single-device streamable Keyword Spotting framework from [`sciapponi/streamable-kws`](https://github.com/sciapponi/streamable-kws). We express our sincere gratitude to **Stefano Ciapponi** for open-sourcing the base streaming MatchboxNet training architecture, upon which we engineered our distributed smart-overlap vocabulary partitioning, Zephyr RTOS C++ drivers, and real-time BLE gateway fusion stack.

This system eliminates the classical **Acoustic Correlation Trap** (where homogeneous arrays make identical errors on phonetically confusable words like "TREE" vs "THREE") through **Phonetic Confusion-Aware Vocabulary Partitioning**, **Universal Overlap Clustering**, **Spatial 4-Bit SNR-Weighted Soft Late Fusion**, and **Bare-Metal Zephyr RTOS C++ Drivers**.

---

## 📑 Table of Contents
- [Quickstart & Environment Setup](#-quickstart--environment-setup)
- [Key Architectural Innovations](#-key-architectural-innovations)
- [3-Device System Architecture Diagram](#-3-device-system-architecture-diagram)
- [Directory & File Overview](#-directory--file-overview)
- [End-to-End Pipeline & Deployment Guide](#-end-to-end-pipeline--deployment-guide)
  - [1. Model Training](#1-model-training)
  - [2. Acoustic Vocabulary Partition Optimizer](#2-acoustic-vocabulary-partition-optimizer)
  - [3. Device-Wise INT8 Quantization Export (Dev 1, Dev 2, Dev 3)](#3-device-wise-int8-quantization-export-dev-1-dev-2-dev-3)
  - [4. Flashing Physical Microcontrollers (Device IDs 1, 2, 3)](#4-flashing-physical-microcontrollers-device-ids-1-2-3)
  - [5. Live Multi-Device Real-Time Gateway Fusion](#5-live-multi-device-real-time-gateway-fusion)
- [Topologies & Vocabulary Allocation Strategies Explained](#-topologies--vocabulary-allocation-strategies-explained)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Confusion Matrix Heatmaps (Gaussian Noise)](#-confusion-matrix-heatmaps-under-stationary-gaussian-hvac-noise)
- [Safety & Squelch Stack](#-safety--squelch-stack)
- [Author & Affiliations](#-author--affiliations)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 🚀 Quickstart & Environment Setup

### 🐍 Step 1: Python Environment Setup (Python 3.12.3)

```bash
# 1. Clone the repository & enter directory
git clone https://github.com/egemen-akkoyunlu/tinyml-distributed-kws-array.git
cd tinyml-distributed-kws-array

# 2. Create & activate virtual environment (Python 3.12.3)
python3 -m venv kws_env
source kws_env/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

### ⚡ Step 2: Zephyr RTOS & Toolchain Setup (Firmware Development)

The embedded C++ firmware runs on **Zephyr RTOS (v3.7+)**. To set up the Zephyr development environment, West meta-tool, and toolchain for the **Seeed Studio XIAO ESP32-S3 Sense**, refer to the official documentation:
- 📖 [Zephyr RTOS Getting Started Guide](https://docs.zephyrproject.org/latest/develop/getting_started/index.html)
- 🔌 [Seeed Studio XIAO ESP32S3 Zephyr Board Documentation](https://docs.zephyrproject.org/latest/boards/seeed/xiao_esp32s3/doc/index.html)
- 🧠 [Espressif ESP-DL Repository](https://github.com/espressif/esp-dl) *(Vector SIMD acceleration library)*

---

## 🌟 Key Architectural Innovations

### 1. Hybrid INT8/FP32 Microcontroller Execution Engine
- **Vector SIMD Acceleration:** Offloads 95% of neural network MAC operations (1D depthwise-separable dilated convolutions in MatchboxNet) to the Xtensa LX7 dual-core 128-bit Vector SIMD instructions (`esp-dl` kernel `base::mat_vec_dotprod`).
- **Fixed-Point LUTs + Hardware FPU Softmax:** Fast fixed-point Lookup Tables (LUTs) compute Tanh and Sigmoid activations, while output layer Softmax probabilities are computed losslessly via the hardware single-precision Floating Point Unit (`CONFIG_FPU=y`).
- **SRAM Budgeting & Automatic PSRAM Fallback:** Compresses the model footprint from **133.6 KB (FP32) down to 100.7 KB (INT8)** (1.33x compression) with **negligible accuracy degradation (<0.2%)**. Fits cleanly within the **512 KB internal SRAM** (allocating a strict $<128\text{ KB}$ runtime buffer for intermediate activations and 32 KB Mel spectrogram sliding buffers), with our custom `esp_heap_caps.cpp` fallback allocator auto-overflowing to **8 MB Octal PSRAM** if needed.
- **Empirical On-Chip Latency Profiling:** Microcontroller UART hardware benchmarks on the ESP32-S3 confirm real-time execution per streaming window:
  - *Audio Feature Extraction (Mel Filterbank + FFT):* **$1.41\text{ ms}$**
  - *INT8 SIMD Neural Inference (MatchboxNet):* **$46.28\text{ ms}$**
  - *Total On-Device Turnaround:* **$\approx 47.7\text{ ms}$** (guaranteeing continuous real-time streaming with zero I2S DMA buffer overruns).

<p align="center">
  <img src="docs/assets/01_quantization_comparison.png" alt="INT8 Quantization Comparison" width="650"/>
</p>

### 2. Eliminating the Acoustic Correlation Trap via Tri-Factor Vulnerability Metric
- **The Classical Failure Mode:** In traditional multi-microphone arrays where all nodes run the identical generalist model, devices share **identical decision boundary blind spots**. Under room noise and reverberation, homogeneous nodes fail unanimously (e.g. 25.8% false alarm rate on phonetically confusable pairs like `TREE` vs `THREE`).
- **Mathematical Decomposition of $V(k)$:** The Tri-Factor Acoustic Vulnerability Metric combines three distinct statistical properties from the baseline confusion matrix:

```math
V(k) = 0.70 \cdot \text{Error}(k) + 0.15 \cdot H(k) + 0.15 \cdot \text{Peak}(k)
```

  1. **$\text{Error}(k) = 1.0 - \text{Recall}(k)$ (Empirical Classification Error):** Measures the raw baseline failure rate of class $k$, prioritizing inherently difficult acoustic classes.
  2. **$H(k) = -\frac{1}{\ln(K-1)} \sum_{j \ne k, P_{kj} > 0} P_{kj} \ln(P_{kj})$ (Normalized Confusion Entropy):** Measures the **diffuseness** of false predictions over off-diagonal transitions ($P_{kj} = \frac{C_{kj}}{\sum_{m \ne k} C_{km}}$). Diffuse, non-specific background noise produces high entropy ($H \to 1.0$).
  3. **$\text{Peak}(k) = \max_{j \ne k} P_{kj}$ (Maximum Pairwise Off-Diagonal Peak):** Measures the **sharpness** of the worst pairwise confusion spike. 
  - **The Entropy Paradox Solved:** If a word has a single massive phonetic twin (e.g., `TREE` misclassifying almost exclusively into `THREE`), $H(k)$ is artificially *depressed* ($H \approx 0$). Without $\text{Peak}(k)$, entropy would penalize binary twins. Combining $H(k)$ and $\text{Peak}(k)$ guarantees that both diffuse noise and sharp phonetic twins (`{"tree", "three"}`, `{"four", "forward"}`) receive high vulnerability scores.
- **Symmetrized Confusion Matrix Coupling ($W_{ij}$):** Constructs a bidirectional phonetic confusion coupling matrix where $W_{ij} = \max(P(j|i), P(i|j)) \ge 0.05$. Indivisible phonetic twin pairs are locked together and co-located on specialized nodes to eliminate cross-talk.

### 3. Universal Overlap Acoustic Array Topology
- **The Pitfall of Disjoint Partitioning ($R=1$):** Partitioning the 36-class vocabulary into disjoint subsets (13+13+12) boosts per-class capacity (+200%) but provides zero spatial redundancy: a single battery failure ($N-1$) crashes accuracy catastrophically ($-19.5\%$).
- **Bounded Overlap Clustering ($R \ge 2$ with $R=3$ Anchors):** Enforces a strict minimum replication level of $R \ge 2$ across every single keyword, while top confusable Anchor cliques (`tree`/`three`) are replicated on all 3 nodes ($R=3$).
- **Capacity & Redundancy Co-Optimization:** Reduces per-node vocabulary from 36 down to 25 classes (+44% parameter capacity per class) while guaranteeing robust $2/3$ majority consensus across every keyword.

### 4. 4-Bit Spatial SNR-Weighted Soft Late Fusion
- **On-Chip Hendriks MMSE Spectral Tracking:** Each edge node executes real-time 40-bin Minimum Mean-Square Error (MMSE) noise estimation directly in C++ on the microcontroller to track continuous background noise floors.
- **Physical SNR Calculation:** Derives acoustic signal-to-noise ratio in decibels:

```math
\text{SNR}_{\text{dB}} = 10 \cdot \log_{10}\left(\frac{\text{RMS}^2}{\max(20.0, \text{Noise} \times 1000.0)}\right)
```

- **Sigmoidal Weighting ($w_i \in [1, 15]$):** Maps physical SNR into a compact 4-bit integer spatial weight:

```math
w_i = \text{round}\left( \frac{15}{1 + e^{-(\text{SNR}_{\text{dB}} - 10.0) / 3.5}} \right) \in [1, 15]
```

  granting near-field microphones up to $15\times$ higher voting authority over distant, reverberant microphones.
- **Soft Late Fusion Formulation:** Decouples physical acoustic reliability from neural network classification confidence:

```math
\text{Fused Confidence} = \frac{\sum_{i \in \mathcal{A}} (\text{Confidence}_i \times w_i)}{\sum_{i \in \mathcal{A}} w_i}
```

### 5. Multi-Stage False Alarm Defense & Acoustic Arbiter Veto
- **Dual-Tier Hardware & Gateway Squelch:**
  1. *Embedded C++ Squelch:* RMS energy bar (<5.0) skips neural execution during silence, 2-consecutive-frame temporal debounce (120 ms), and 150 ms refractory lockout prevent double-triggering.
  2. *Gateway Confidence Margin:* Rejects predictions where the Top-1 vs Top-2 probability margin is $< 15\%$.
  3. *Dynamic Room Noise Threshold:* Adapts consensus trigger bar to room acoustics: `T_dyn = min(72%, max(54%, 58% + 14 * Noise))`.
  4. *Acoustic Arbiter Veto:* When a specialist node covering both words predicts word $A$ with $\ge 70\%$ confidence, it automatically vetoes any out-of-vocabulary proposal of word $B$ from sibling nodes that lack $A$ in their vocabulary.

### 6. Fail-Stop Hardware Fault Tolerance & Battery Depletion Resilience
- **Graceful Fail-Stop Degradation:** Under battery exhaustion or hardware failure ($N-1, N-2, N-3$), the gateway seamlessly switches consensus quorums.
- **Extreme Noise Benchmarks:** Under severe environmental background noise (ESC-50), a dead node ($N-1$) causes **only a $-1.3\%$ drop** on the 5-device 3/5 Majority array ($67.1\% \to 65.8\%$) and **$-4.0\%$** on the 3-device Universal Overlap array ($67.1\% \to 63.1\%$), outperforming healthy homogeneous baseline arrays ($59.3\%$).

### 7. Production Embedded Zephyr RTOS C++ Stack & Binary BLE Protocol
- **Zero-Copy I2S DMA Streaming:** PDM microphone data is streamed directly via DMA memory slabs into circular sliding window buffers without CPU memcpy overhead.
- **Exact 10-Byte Bit-Packed Binary GATT Telemetry:**
  - Encodes the complete state into the packed C struct `BleKwsPacket` (`zephyr_kws/src/ble_server.h`):
    - `ts_ms`: Hardware Timestamp in milliseconds (`uint32_t`, **4 Bytes**)
    - `noise_x1k`: Linear Noise Floor $\times 1000$ (`uint16_t`, **2 Bytes**)
    - `dev_id`: Device ID 1/2/3 (`uint8_t`, **1 Byte**)
    - `top_class`: Predicted Class Index (`uint8_t`, **1 Byte**)
    - `conf`: Confidence percentage 0–100 (`uint8_t`, **1 Byte**)
    - `rms`: Audio RMS Energy intensity (`uint8_t`, **1 Byte**)
    - **Total Bit-Packed Payload:** $4 + 2 + 1 + 1 + 1 + 1 = \mathbf{10\text{ Bytes}}$ (with zero padding).
  - *Physical Frame Airtime Derivation:* Total packet over-the-air (Preamble + Access Address + LL Header + ATT Handle + 10B Payload + 3B CRC) = **28 bytes (224 bits)**. Over Bluetooth LE 1M PHY (1 Mbps = 1.0 $\mu$s/bit), the transmission takes **$0.224\text{ ms} \approx 0.3\text{ ms}$ airtime** (duty cycle $<0.1\%$).
  - *Energy Impact:* Eliminates continuous $256\text{ kbps}$ raw audio streaming, cutting BLE radio active power by **$>99\%$** and enabling multi-month coin-cell battery life.
- **Empirically Measured Power Consumption:** Continuous streaming audio inference, Mel filterbank extraction, and active BLE GATT notifications draw an average current of **$64.8\text{ mA}$** (mean power: **$240\text{ mW}$** at 3.7V LiPo), confirming practical battery-powered TinyML deployment.
- **Adaptive OLS Clock Synchronization:** An Adaptive Sliding Window Ordinary Least Squares (OLS) estimator continuously synchronizes node clocks to the master gateway time with a physical $\pm 500$ ppm quartz crystal clamp.

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
├── README.md                             # Project Documentation
├── requirements.txt                      # Python dependencies
├── config/                               # Hydra YAML training & partition configurations
│   ├── kws_35_classes.yaml               # 36-Class Generalist baseline config
│   ├── kws_smart_overlap_dev1_25classes.yaml  # Node 1 Partition Config (25 classes)
│   ├── kws_smart_overlap_dev2_25classes.yaml  # Node 2 Partition Config (25 classes)
│   ├── kws_smart_overlap_dev3_25classes.yaml  # Node 3 Partition Config (25 classes)
│   └── kws_smart_overlap_5dev_dev*.yaml  # 5-Node Universal Overlap (23-23-23-23-22) configs
├── docs/assets/                          # High-resolution benchmark figures & dashboards
├── models/                               # PyTorch Streaming Neural Network architectures
│   └── streaming.py                      # Improved_Phi_GRU_ATT_Streaming MatchboxNet
├── zephyr_kws/                           # Bare-metal C++ firmware for Zephyr RTOS
│   ├── CMakeLists.txt                    # Zephyr build configuration & ESP-DL links
│   ├── prj.conf                          # Hardware FPU, PSRAM, DMA, and RTOS flags
│   ├── app.overlay                       # Device Tree overlay for I2S PDM microphone
│   └── src/                              # Embedded C++ drivers & inference engine
│       ├── main.cpp                      # RTOS thread entry & audio streaming loop
│       ├── inference.cpp / .hpp          # ESP-DL neural inference & hardware FPU Softmax
│       ├── audio_i2s.cpp / .hpp          # Zero-copy I2S DMA microphone driver
│       ├── audio_processing.cpp / .hpp   # Mel filterbank & 100-frame sliding window
│       ├── ble_server.c / .h             # 10-byte binary GATT telemetry streamer
│       ├── kws_config.hpp                # Device ID (1/2/3) & threshold configuration
│       ├── microsd.c / .h                # Optional MicroSD telemetry logger
│       └── compat/                       # ESP-IDF to Zephyr RTOS compatibility layer
├── datasets.py                           # Google Speech Commands v2 streaming loader
├── train.py                              # PyTorch training script with Hydra & Cosine Annealing
├── optimize_smart_overlap_allocation.py  # Acoustic Partition Optimizer (Tri-factor V(k) & W_ij)
├── quantize_to_espdl.py                  # ESP-PPQ INT8/FP32 Post-Training Quantizer
├── multi_device_fusion.py                # Real-time BLE Gateway Concentrator & Late Fusion
└── logs/                                 # Pre-trained checkpoints for 3-Device Universal Overlap
    ├── kws_smart_overlap_dev1_25classes/ # Node 1 pre-trained PyTorch weights (.pth) & configs
    ├── kws_smart_overlap_dev2_25classes/ # Node 2 pre-trained PyTorch weights (.pth) & configs
    └── kws_smart_overlap_dev3_25classes/ # Node 3 pre-trained PyTorch weights (.pth) & configs
```

---

## 🛠️ End-to-End Pipeline & Deployment Guide

> [!TIP]
> **Pre-Trained Weights Included:**  
> Pre-trained PyTorch models for all 3 Universal Overlap nodes are already included in `logs/kws_smart_overlap_dev[1-3]_25classes/`. You can **skip Step 1 (Training)** and jump directly to **Step 3 (INT8 Quantization)**!

### 1. Model Training (Optional)
Train individual partition device models or the 36-class baseline:
```bash
# Train Node 1 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev1_25classes.yaml

# Train Node 2 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev2_25classes.yaml

# Train Node 3 (25 classes)
python3 train.py --config-name=kws_smart_overlap_dev3_25classes.yaml
```

---

### 2. Acoustic Vocabulary Partition Optimizer
Derive optimal vocabulary partitions using the Tri-Factor Vulnerability Metric:
```bash
# Generate 3-Node Universal Overlap partition (25+25+25 classes)
python3 optimize_smart_overlap_allocation.py --nodes 3

# Generate 5-Node Universal Overlap partition (23+23+23+23+22 classes)
python3 optimize_smart_overlap_allocation.py --nodes 5
```

---

### 3. Device-Wise INT8 Quantization Export (Dev 1, Dev 2, Dev 3)

Quantize each device's pre-trained PyTorch model into calibrated INT8 `.espdl` binaries:

```bash
# 1. Quantize Node 1 (25 classes)
python3 quantize_to_espdl.py logs/kws_smart_overlap_dev1_25classes

# 2. Quantize Node 2 (25 classes)
python3 quantize_to_espdl.py logs/kws_smart_overlap_dev2_25classes

# 3. Quantize Node 3 (25 classes)
python3 quantize_to_espdl.py logs/kws_smart_overlap_dev3_25classes
```

---

### 4. Flashing Physical Microcontrollers (Device IDs 1, 2, 3)

> **Note on ESP-DL:** The firmware utilizes Espressif's accelerated `esp-dl` library. If building firmware, ensure `esp-dl` is present in the root workspace:
> ```bash
> git clone https://github.com/espressif/esp-dl.git
> ```

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

The gateway automatically discovers all 3 nodes, establishes encrypted binary GATT telemetry streams, continuously tracks Hendriks MMSE noise floors, computes 4-bit SNR weights ($w_i \in [1, 15]$), compensates for clock skew via Ordinary Least Squares (OLS), and outputs real-time consensus keyword triggers:

#### 📡 Real-World Physical Hardware Live Telemetry Output (3x XIAO ESP32-S3 Sense):
```text
*** [FUSED TRIGGER: STOP (99.1% >= 61% | 2-NODE SNR CONSENSUS | OLS Skew: 2.5ms)] ***
    ├── Dev 2 (XIAO_SENSE_BLE_2): STOP (98.0%)  | Audio RMS: 52.0  | Linear Noise: 0.065 | 4-Bit Weight: 13/15 | OLS TS: 71789ms | XTAL Drift: -500.0ppm
    ├── Dev 3 (XIAO_SENSE_BLE_3): STOP (100.0%) | Audio RMS: 188.0 | Linear Noise: 0.089 | 4-Bit Weight: 15/15 | OLS TS: 71786ms | XTAL Drift: -201.5ppm

*** [FUSED TRIGGER: THREE (96.3% >= 61% | 2-NODE SNR CONSENSUS | OLS Skew: 25.5ms)] ***
    ├── Dev 2 (XIAO_SENSE_BLE_2): THREE (98.0%) | Audio RMS: 41.0  | Linear Noise: 0.070 | 4-Bit Weight: 11/15 | OLS TS: 317871ms | XTAL Drift: -337.2ppm
    ├── Dev 3 (XIAO_SENSE_BLE_3): THREE (95.0%) | Audio RMS: 82.0  | Linear Noise: 0.084 | 4-Bit Weight: 14/15 | OLS TS: 317897ms | XTAL Drift: +6.5ppm

*** [FUSED TRIGGER: TREE (99.5% >= 61% | 2-NODE SNR CONSENSUS | OLS Skew: 10.9ms)] ***
    ├── Dev 2 (XIAO_SENSE_BLE_2): TREE (99.0%)  | Audio RMS: 48.0  | Linear Noise: 0.073 | 4-Bit Weight: 12/15 | OLS TS: 319571ms | XTAL Drift: -187.4ppm
    ├── Dev 3 (XIAO_SENSE_BLE_3): TREE (100.0%) | Audio RMS: 83.0  | Linear Noise: 0.078 | 4-Bit Weight: 14/15 | OLS TS: 319582ms | XTAL Drift: +72.9ppm

*** [FUSED TRIGGER: FORWARD (96.1% >= 61% | 2-NODE SNR CONSENSUS | OLS Skew: 190.1ms)] ***
    ├── Dev 1 (XIAO_SENSE_BLE_1): FORWARD (92.0%)  | Audio RMS: 70.0 | Linear Noise: 0.087 | 4-Bit Weight: 13/15 | OLS TS: 170296ms | XTAL Drift: +500.0ppm
    ├── Dev 3 (XIAO_SENSE_BLE_3): FORWARD (100.0%) | Audio RMS: 88.0 | Linear Noise: 0.094 | 4-Bit Weight: 14/15 | OLS TS: 170106ms | XTAL Drift: -500.0ppm
```

#### 🔬 Real-World 4-Bit SNR Weighting & Soft Fusion Regimes Explained

The gateway decouples **Physical Acoustic Reliability (Signal Physics)** from **Neural Network Confidence (Model Probability)** via the two-stage soft late fusion formulation:

```math
\text{Fused Confidence} = \frac{\sum_{i \in \mathcal{A}} (\text{Confidence}_i \times w_i)}{\sum_{i \in \mathcal{A}} w_i}
```

Where `w_i = round(sigma(SNR_i) * 15) in [1, 15]` is the 4-bit integer spatial weight derived on-chip from speech energy vs Hendriks MMSE noise floor.

From our live physical hardware deployment, three distinct acoustic regimes govern weighting and consensus:

1. **Spatial Proximity & Inverse-Square Law Domination ($13/15$ vs $1/15$) — e.g. `RIGHT`:**
   - **Observed Telemetry:** Node 1 ($\text{RMS}=60.0, \text{Noise}=0.070 \to w=13$), Node 2 ($\text{RMS}=8.0, \text{Noise}=0.067 \to w=1$).
   - **Acoustic Rationale:** The speaker was positioned close to Node 1. Due to $1/r^2$ attenuation, Node 2 captured distant, reverberant speech.
   - **Fusion Impact:** Node 1 receives **$92.8\%$ of the voting authority** ($\frac{13}{14}$), preventing far-field room echoes from distorting the fused prediction ($99.1\%$).

2. **Noise Floor Penalization at Similar Speech Volume ($8/15$ vs $6/15$) — e.g. `VISUAL` & `NINE`:**
   - **Observed Telemetry:** Node 1 ($\text{RMS}=28.0, \text{Noise}=0.077 \to w=8$), Node 3 ($\text{RMS}=25.0, \text{Noise}=0.083 \to w=6$).
   - **Acoustic Rationale:** Both nodes capture similar moderate voice energy, but Node 3 operates in a slightly noisier acoustic zone (higher noise floor).
   - **Fusion Impact:** The cleaner channel automatically receives higher authority ($8/15$ vs $6/15$), rewarding microphones with higher signal clarity.

3. **Multi-Node Balanced Consensus ($11/15$ vs $14/15$) — e.g. `THREE` & `TREE`:**
   - **Observed Telemetry:** Node 2 ($\text{Conf}=98.0\%, w=11$), Node 3 ($\text{Conf}=95.0\%, w=14$).
   - **Acoustic Rationale:** Both nodes operate in high SNR ($\text{RMS} \ge 40.0$).
   - **Fusion Impact:** Produces an ultra-sharp consensus:

```math
\text{Fused Conf} = \frac{(98.0 \times 11) + (95.0 \times 14)}{11 + 14} = \mathbf{96.3\%}
```

   Completely eliminating false alarms between confusable acoustic twins.

---

## 🔬 Topologies & Vocabulary Allocation Strategies Explained

| Architecture / Topology | Class Allocation | Replication Level ($R$) | Consensus Quorum | Architectural Rationale & Behavior |
| :--- | :---: | :---: | :---: | :--- |
| **1. Single Device Baseline** | 36 Classes | $R = 1$ | 1/1 (Standalone) | Standard edge setup running the full vocabulary on a single node. Vulnerable to room geometry, distance ($1/r^2$), and phonetic confusion. |
| **2. 3x Homogeneous Array** | 3x 36 Classes | $R = 3$ (Uniform) | $\ge 2/3$ Nodes | 3 identical devices with identical weights. Suffers from the **Acoustic Correlation Trap**: identical weights mean identical decision boundary blind spots. |
| **3. 0-Overlap Disjoint Partition** | 13 + 13 + 12 | $R = 1$ (Disjoint) | Specialist Top-1 | Vocabulary is partitioned into 3 disjoint sets with zero class overlap. Maximizes capacity (+200%), but collapses on node failure ($N-1$ battery loss drops accuracy by $-19.5\%$). |
| **4. Phonetic-Coupled Partition** | 22 + 22 + 22 | $R \in [1, 3]$ | Specialist Voting | Connects confusable phonetic twins onto the same nodes ($W_{ij} \ge 0.05$), with $R=3$ for top anchor clusters (`tree`/`three`) and $R=1$ for non-confusable words. |
| **5. Universal Overlap Array** *(Recommended 3-Node)* | 25 + 25 + 25 | $R \ge 2$ ($R=3$ for Anchors) | $\ge 2/3$ Majority Quorum | **Gold Standard for 3 Nodes:** Every keyword is replicated on at least 2 nodes ($R \ge 2$), while top confusable twins (`tree`/`three`) are Anchors on all 3 nodes ($R=3$). Eliminates single points of failure. |
| **6. 5-Device 2/5 Overlap Array** | 5x ~16 Classes | $R \in [2, 3]$ | $\ge 2/5$ Quorum | High specialization (~16 classes/node). Fast and lightweight, triggering on agreement of any 2 out of 5 nodes. |
| **7. 5-Device 3/5 Majority Array** *(Ultimate Resilient)* | 5x ~23 Classes | $R \in [3, 5]$ | $\ge 3/5$ Strict Majority | **Maximum Fault Tolerance:** High overlapping vocabulary (~23 classes/node). When a node suffers battery loss ($N-1$), accuracy drops by **only $-1.3\%$** ($67.1\% \to 65.8\%$) under severe noise. |

---

## 📊 Empirical Benchmark Results

### 1. Multi-Condition Acoustic Shootout (Google Speech Commands v2 Test Partition)

| Architecture / Topology | Class Allocation | Clean Room (High SNR) | Gaussian HVAC Noise | Real ESC-50 Noise |
| :--- | :---: | :---: | :---: | :---: |
| **1. Single Device Baseline** | 36 Classes | 85.5% | 70.9% | 57.1% |
| **2. 3x Homogeneous Array** | 3x 36 Classes | 87.5% | 73.9% | 59.3% |
| **3. 0-Overlap Disjoint Partition** | 13 + 13 + 12 | 89.6% | 82.4% | 64.9% |
| **4. Phonetic-Coupled Partition** | 22 + 22 + 22 | 92.6% | 86.3% | 66.8% |
| **5. Universal Overlap Array** | 25 + 25 + 25 | 92.2% | **88.1%** | **67.1%** |
| **6. 5-Device 2/5 Overlap Array** | 5x ~16 Classes | 93.0% | 86.7% | 67.0% |
| **7. 5-Device 3/5 Majority Array** | 5x ~23 Classes | **93.9%** | **88.1%** (+14.2%) | **67.1%** (+10.0%) |

#### 🔊 1. Clean Silent Room Acoustic Shootout (High SNR)
<p align="center">
  <img src="docs/assets/dashboard_clean_room.png" alt="Clean Room Acoustic Dashboard" width="950"/>
</p>

#### 💨 2. Stationary Gaussian HVAC Noise Shootout
<p align="center">
  <img src="docs/assets/dashboard_gaussian_noise.png" alt="Gaussian Stationary Noise Dashboard" width="950"/>
</p>

#### 🌧️ 3. Real ESC-50 Environmental Background Noise Shootout
<p align="center">
  <img src="docs/assets/dashboard_esc50_noise.png" alt="ESC-50 Environmental Noise Dashboard" width="950"/>
</p>

---

### 2. Hardware Resilience Under Battery Loss ($N-1$ to $N-3$ Dead Nodes under ESC-50)

| Topology | 0 Dead Nodes (100% Healthy) | 1 Dead Node ($N-1$) | 2 Dead Nodes ($N-2$) | 3 Dead Nodes ($N-3$) |
| :--- | :---: | :---: | :---: | :---: |
| **0-Overlap Disjoint Partition** | 64.9% | **45.4% (-19.5% Crash 💥)** | — | — |
| **3x Homogeneous Array** | 59.3% | 58.9% (-0.4%) | 57.1% | — |
| **Universal Overlap (25-25-25)** | 67.1% | 63.1% (-4.0%) | 58.7% | — |
| **5-Device 3/5 Majority Array** | **67.1%** | **65.8% (-1.3% 🟢)** | **63.5% (Beats 3x Homo!)** | **58.5% (Beats Single)** |

<p align="center">
  <img src="docs/assets/03_n_minus_1_fault_tolerance.png" alt="N-1 Fault Tolerance Degradation Comparison" width="900"/>
</p>

---

## 🟪 Confusion Matrix Heatmaps (Under Stationary Gaussian HVAC Noise)

Evaluated across the official Google Speech Commands v2 test set under **Stationary Gaussian HVAC Room Noise (Calibrated SNR)**. The 3-Device Universal Overlap Array ($25+25+25$) achieves **88.07% overall accuracy** under active room noise, successfully driving the baseline 25.8% `THREE` $\to$ `TREE` false alarm spike down to **0.0%** (achieving **98.1% recall on "THREE"** and **80.6% on "TREE"**):

<p align="center">
  <img src="docs/assets/04_universal_overlap_confusion_matrix.png" alt="Universal Overlap Confusion Matrix under Gaussian HVAC Noise" width="850"/>
</p>

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

## 👤 Author & Affiliations

**Egemen Acar Akkoyunlu**  
- 🔬 Research Intern, **Fondazione Bruno Kessler (FBK)**, Trento, Italy  
- 🎓 Electrical and Electronics Engineering, **Boğaziçi University**, Istanbul, Turkey  
- 🌐 [GitHub: @egemen-akkoyunlu](https://github.com/egemen-akkoyunlu)

---

## 🙏 Acknowledgements

This work extends the baseline single-device streaming MatchboxNet architecture from [`sciapponi/streamable-kws`](https://github.com/sciapponi/streamable-kws) by Stefano Ciapponi into a distributed, multi-microcontroller sensor array.

---

## 📜 License

This project is licensed under the **Apache License 2.0**.
