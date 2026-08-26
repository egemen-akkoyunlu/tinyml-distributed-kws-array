/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef KWS_CONFIG_HPP
#define KWS_CONFIG_HPP

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif

#define SAMPLE_RATE 16000
#define FRAME_LEN_SAMPLES 256
#define FRAME_SHIFT_SAMPLES 160
#define NUM_MEL_BINS 40
#define TIME_FRAMES 100

#define I2S_BLOCK_SIZE 1024
#define SAMPLES_PER_BLOCK 128

// ============================================================================
// TOPOLOGY SELECTION:
// 0: Homogeneous 36-Class Generalist (Legacy)
// 1: Universal Overlap 25-Class MoE (Optimal Energy & 91.2% Spatial Accuracy)
// ============================================================================
#define USE_GRAPH_COUPLED_MOE 1

#if USE_GRAPH_COUPLED_MOE
#define NUM_CLASSES 25
#define NOTHING_CLASS_INDEX 24              // Index of 'nothing' (silence) class
#else
#define NUM_CLASSES 36
#define NOTHING_CLASS_INDEX 35              // Index of 'nothing' (silence) class
#endif

#define CONFIDENCE_THRESHOLD 0.50f          // Keyword confidence threshold
#define NOTHING_TRIGGER_THRESHOLD 0.30f     // If nothing > 30%, definitely silence
#define CONFIDENCE_MARGIN 0.15f             // Minimum margin over second best class (15%)
#define COOLDOWN_FRAMES 15                  // 150ms cooldown after keyword detection
#define INFERENCE_STRIDE 6
#define VOICE_ACTIVITY_RMS_THRESHOLD 5.0f // Energy squelch: skip model run if RMS < threshold

// ============================================================================
// MULTI-DEVICE CONFIGURATION FOR LATE FUSION
// Set DEVICE_ID to 1 for Device 1 (XIAO_SENSE_BLE_1)
// Set DEVICE_ID to 2 for Device 2 (XIAO_SENSE_BLE_2)
// Set DEVICE_ID to 3 for Device 3 (XIAO_SENSE_BLE_3)
// ============================================================================
#ifndef DEVICE_ID
#define DEVICE_ID 3
#endif

#if DEVICE_ID == 3
#define CONFIG_CUSTOM_DEVICE_NAME "XIAO_SENSE_BLE_3"
#elif DEVICE_ID == 2
#define CONFIG_CUSTOM_DEVICE_NAME "XIAO_SENSE_BLE_2"
#else
#define CONFIG_CUSTOM_DEVICE_NAME "XIAO_SENSE_BLE_1"
#endif

// ============================================================================
// AUDIO STREAMING TEST MODE (For Audacity Analysis via record_to_wav.py)
// ============================================================================
// 0: Normal KWS Inference Mode (ASCII serial logs enabled)
// 1: Stream PRE-FILTER Raw PCM over Serial (Before DC-Block HPF & Butterworth LPF)
// 2: Stream POST-FILTER Clean PCM over Serial (After DC-Block HPF & Butterworth LPF)
#define STREAM_RAW_PCM_MODE 0

#endif // KWS_CONFIG_HPP
