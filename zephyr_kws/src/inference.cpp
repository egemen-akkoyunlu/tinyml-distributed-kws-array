/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 */

#include "inference.hpp"
#include "kws_config.hpp"
#include "compat/esp_heap_caps.h"
#include "ble_server.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include <cstring>
#include <cmath>
#include <algorithm>

LOG_MODULE_DECLARE(kws_app, LOG_LEVEL_INF);

/* Otii Sync GPIO Specifications */
#define SYNC_AP_NODE DT_ALIAS(sync_ap)
#define SYNC_INF_NODE DT_ALIAS(sync_inf)

static const struct gpio_dt_spec sync_ap = GPIO_DT_SPEC_GET(SYNC_AP_NODE, gpios);
static const struct gpio_dt_spec sync_inf = GPIO_DT_SPEC_GET(SYNC_INF_NODE, gpios);

#if USE_GRAPH_COUPLED_MOE
#if DEVICE_ID == 1
static const char *CLASS_LABELS[NUM_CLASSES] = {
    "backward", "bird", "dog", "eight", "five", "follow", "forward", "four",
    "learn", "left", "nine", "no", "on", "one", "right", "seven",
    "six", "three", "tree", "two", "up", "visual", "yes", "zero", "nothing"
};
#elif DEVICE_ID == 2
static const char *CLASS_LABELS[NUM_CLASSES] = {
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "go",
    "happy", "house", "learn", "left", "marvin", "nine", "no", "off",
    "one", "right", "sheila", "stop", "three", "tree", "wow", "yes", "nothing"
};
#else // DEVICE_ID == 3
static const char *CLASS_LABELS[NUM_CLASSES] = {
    "bed", "cat", "down", "five", "follow", "forward", "four", "go",
    "happy", "house", "marvin", "off", "on", "seven", "sheila", "six",
    "stop", "three", "tree", "two", "up", "visual", "wow", "zero", "nothing"
};
#endif
#else
static const char *CLASS_LABELS[NUM_CLASSES] = {
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", 
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", 
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
    "tree", "two", "up", "visual", "wow", "yes", "zero", "nothing"
};
#endif

static const unsigned char model_espdl_bin[] __attribute__((aligned(16))) = {
#include "model_espdl.inc"
};

KWSInference::KWSInference()
    : model(nullptr), fbank(nullptr), input_tensor(nullptr), output_tensor(nullptr),
      sliding_window_buf(nullptr), normalized_buf(nullptr), cooldown_counter(0),
      last_detected_class(-1), consecutive_count(0), dropout_count(0),
      last_ap_start(0), last_ap_end(0), last_ap_time_ms(0.0),
      last_inf_start(0), last_inf_end(0), last_inf_time_ms(0.0)
{
}

KWSInference::~KWSInference()
{
    if (model) delete model;
    if (fbank) delete fbank;
    if (sliding_window_buf) heap_caps_free(sliding_window_buf);
    if (normalized_buf) heap_caps_free(normalized_buf);
}

void KWSInference::compute_softmax(const float *logits, float *probs, int len)
{
    float max_logit = logits[0];
    for (int i = 1; i < len; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < len; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < len; i++) {
        probs[i] /= sum_exp;
    }
}

bool KWSInference::init()
{
    LOG_INF("[FLAG: STAGE_0_START] Loading Model from Flash...");
    LOG_INF("Model binary at: %p (size: %zu bytes)", 
            (void *)model_espdl_bin, sizeof(model_espdl_bin));
    
    /*
     * Initialize the Flatbuffer Model object.
     * 
     * IMPORTANT OPTIMIZATION: 
     * We pass "false" as the 6th argument (param_copy) to run model weights
     * directly from Flash RODATA. This saves ~80 KB of internal SRAM,
     * preventing allocation failures on the system heap.
     */
    model = new dl::Model((const char *)model_espdl_bin, 
                          fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                          4096,
                          dl::MEMORY_MANAGER_GREEDY,
                          nullptr,
                          false);

    if (model == nullptr) {
        LOG_ERR("[FLAG: ERROR] Failed to initialize dl::Model!");
        return false;
    }

    LOG_INF("[FLAG: SUCCESS] Model initialized successfully!");

    auto inputs = model->get_inputs();
    auto outputs = model->get_outputs();
    
    if (inputs.empty() || outputs.empty()) {
        LOG_ERR("[FLAG: ERROR] Model has no inputs or outputs!");
        return false;
    }
    
    LOG_INF("--- Model Inputs ---");
    for (auto const& [name, tensor] : inputs) {
        std::string shape_str = "";
        for (int d : tensor->shape) shape_str += std::to_string(d) + " ";
        LOG_INF("  Input '%s': shape=[ %s] (bytes: %d)", name.c_str(), shape_str.c_str(), tensor->get_bytes());
    }
    
    // Default to first input, which should be the spectrogram
    input_tensor = inputs.begin()->second;

    LOG_INF("--- Model Outputs ---");
    output_tensor = nullptr;
    for (auto const& [name, tensor] : outputs) {
        std::string shape_str = "";
        for (int d : tensor->shape) shape_str += std::to_string(d) + " ";
        LOG_INF("  Output '%s': shape=[ %s] (bytes: %d)", name.c_str(), shape_str.c_str(), tensor->get_bytes());
        
        // Calculate total elements in this output tensor
        int elements = 1;
        for (int d : tensor->shape) elements *= d;
        
        // The logits tensor must have exactly NUM_CLASSES elements (5)
        if (elements == NUM_CLASSES) {
            output_tensor = tensor;
            LOG_INF("  -> Selected '%s' as the active logits output tensor", name.c_str());
        }
    }

    if (output_tensor == nullptr) {
        LOG_WRN("[FLAG: WARNING] No output tensor with %d elements found! Defaulting to first output.", NUM_CLASSES);
        output_tensor = outputs.begin()->second;
    }

    // Verify input/output dimension sizes
    if (input_tensor->shape[0] != 1 || input_tensor->shape[1] != TIME_FRAMES || input_tensor->shape[2] != NUM_MEL_BINS) {
        LOG_ERR("[FLAG: ERROR] Input tensor shape mismatch! Expected [1, %d, %d], got [%d, %d, %d]", 
                TIME_FRAMES, NUM_MEL_BINS, input_tensor->shape[0], input_tensor->shape[1], input_tensor->shape[2]);
        return false;
    }

    if (output_tensor->shape[0] != 1 || output_tensor->shape[1] != NUM_CLASSES) {
        LOG_ERR("[FLAG: ERROR] Output tensor shape mismatch! Expected [1, %d], got [%d, %d]", 
                NUM_CLASSES, output_tensor->shape[0], output_tensor->shape[1]);
        return false;
    }

    LOG_INF("[FLAG: STAGE_0_START] Initializing 40-channel Mel Filterbank...");
    dl::audio::SpeechFeatureConfig fbank_cfg;
    fbank_cfg.sample_rate = SAMPLE_RATE;       // 16000
    fbank_cfg.frame_length = 16;               // 16 ms window (256 samples at 16kHz)
    fbank_cfg.frame_shift = 10;                // 10 ms step (160 samples at 16kHz)
    fbank_cfg.num_mel_bins = NUM_MEL_BINS;     // 40 mel bins
    fbank_cfg.low_freq = 0.0f;                 // 0.0 Hz (matches PyTorch's f_min = 0.0)
    fbank_cfg.use_log_fbank = 2;               // logf(x + epsilon)
    fbank_cfg.log_epsilon = 1e-10f;
    fbank_cfg.preemphasis = 0.0f;              // No preemphasis
    fbank_cfg.remove_dc_offset = true;         // Remove DC
    fbank_cfg.window_type = dl::audio::WinType::HANN;

    fbank = new dl::audio::Fbank(fbank_cfg);
    if (fbank == nullptr) {
        LOG_ERR("[FLAG: ERROR] Failed to initialize dl::audio::Fbank!");
        return false;
    }
    LOG_INF("[FLAG: SUCCESS] Fbank initialized successfully!");

    sliding_window_buf = (float *)heap_caps_malloc(
        NUM_MEL_BINS * TIME_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    
    if (!sliding_window_buf) {
        LOG_ERR("[FLAG: ERROR] Failed to allocate sliding window buffer in PSRAM!");
        return false;
    }
    memset(sliding_window_buf, 0, NUM_MEL_BINS * TIME_FRAMES * sizeof(float));
    LOG_INF("[FLAG: SUCCESS] Sliding window buffer allocated and zero-initialized in PSRAM");

    normalized_buf = (float *)heap_caps_malloc(
        NUM_MEL_BINS * TIME_FRAMES * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!normalized_buf) {
        LOG_ERR("[FLAG: ERROR] Failed to allocate normalized buffer in PSRAM!");
        heap_caps_free(sliding_window_buf);
        sliding_window_buf = nullptr;
        return false;
    }
    memset(normalized_buf, 0, NUM_MEL_BINS * TIME_FRAMES * sizeof(float));
    LOG_INF("[FLAG: SUCCESS] Normalized buffer allocated and zero-initialized in PSRAM");
    LOG_INF("[FLAG: SUCCESS] All buffers allocated successfully!");
    
    /* Initialize Otii Sync GPIOs as output, inactive by default */
    if (device_is_ready(sync_ap.port)) {
        gpio_pin_configure_dt(&sync_ap, GPIO_OUTPUT_INACTIVE);
        LOG_INF("[FLAG: SUCCESS] Audio Processing Sync GPIO initialized on Pin D1 (GPIO2)");
    } else {
        LOG_WRN("[FLAG: WARNING] Audio Processing Sync GPIO device not ready");
    }

    if (device_is_ready(sync_inf.port)) {
        gpio_pin_configure_dt(&sync_inf, GPIO_OUTPUT_INACTIVE);
        LOG_INF("[FLAG: SUCCESS] Inference Sync GPIO initialized on Pin D2 (GPIO3)");
    } else {
        LOG_WRN("[FLAG: WARNING] Inference Sync GPIO device not ready");
    }

    return true;
}

void KWSInference::process_audio_window(const int16_t *audio_window, int frame_counter, bool log_details)
{
    gpio_pin_set_dt(&sync_ap, 1); /* Set AP pin High */
    last_ap_start = k_cycle_get_32();
    float mel_frame_out[NUM_MEL_BINS] = {0};
    fbank->process_frame(audio_window, FRAME_LEN_SAMPLES, mel_frame_out);

    // Convert ln(x) from esp-dl Fbank to 10*log10(x) to match PyTorch's AmplitudeToDB
    // 10*log10(x) = ln(x) * 10/ln(10) ≈ ln(x) * 4.342945
    static const float LN_TO_DB = 10.0f / logf(10.0f);  // 4.342945f
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        mel_frame_out[m] *= LN_TO_DB;
    }

    // Shift each Mel channel's time-series left by 1 frame and append the new Mel coefficient.
    // This maintains the [NUM_MEL_BINS, TIME_FRAMES] shape required by the ONNX model's layout.
    for (int m = 0; m < NUM_MEL_BINS; m++) {
        float *row = sliding_window_buf + m * TIME_FRAMES;
        memmove(row, row + 1, (TIME_FRAMES - 1) * sizeof(float));
        row[TIME_FRAMES - 1] = mel_frame_out[m];
    }
    last_ap_end = k_cycle_get_32();
    gpio_pin_set_dt(&sync_ap, 0); /* Set AP pin Low */
    last_ap_time_ms = (double)(last_ap_end - last_ap_start) / (sys_clock_hw_cycles_per_sec() / 1000.0);

    if (frame_counter % INFERENCE_STRIDE == 0) {
        // Calculate audio RMS first over the current audio frame (256 samples) to check for voice activity
        int64_t win_sum_sq = 0;
        for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
            win_sum_sq += (int32_t)audio_window[i] * (int32_t)audio_window[i];
        }
        float win_rms = sqrtf((float)win_sum_sq / FRAME_LEN_SAMPLES);

        float probs[NUM_CLASSES] = {0.0f};

        // ============================================================
        // EARLY ENERGY SQUELCH (POWER SAVING VAD GATE)
        // ============================================================
        if (win_rms < VOICE_ACTIVITY_RMS_THRESHOLD) {
            // Force probabilities to 100% silence (nothing) without running the model
            for (int c = 0; c < NOTHING_CLASS_INDEX; c++) probs[c] = 0.0f;
            probs[NOTHING_CLASS_INDEX] = 1.0f;

            if (win_rms > 5.0f && (frame_counter % 20 == 0)) {
                // LOG_INF("[FLAG: VAD_SQUELCH] Quiet audio (RMS: %.1f < Thresh: %.1f). Skipping model run to save power.",
                //         (double)win_rms, (double)VOICE_ACTIVITY_RMS_THRESHOLD);
            }
        } else {
            // Active speech detected: proceed with feature normalization and inference
            gpio_pin_set_dt(&sync_inf, 1); /* Set INF pin High */
            last_inf_start = k_cycle_get_32();
            
            // 1. Compute Mean
            float sum = 0.0f;
            int num_elements = NUM_MEL_BINS * TIME_FRAMES;
            for (int i = 0; i < num_elements; i++) {
                sum += sliding_window_buf[i];
            }
            float mean = sum / num_elements;

            // 2. Compute Standard Deviation
            float sum_sq_diff = 0.0f;
            for (int i = 0; i < num_elements; i++) {
                float diff = sliding_window_buf[i] - mean;
                sum_sq_diff += diff * diff;
            }
            float std_dev = sqrtf(sum_sq_diff / num_elements) + 1e-5f;

            // Debug prints to verify log-mel value ranges, mean, and standard deviation
            if (frame_counter % 100 == 0) {
                // LOG_INF("Feature Stats: Mean = %.3f, Std = %.3f", (double)mean, (double)std_dev);
                // LOG_INF("Latest Mel Bins (0-4): %.3f, %.3f, %.3f, %.3f, %.3f",
                //         (double)sliding_window_buf[0 * TIME_FRAMES + 99],
                //         (double)sliding_window_buf[1 * TIME_FRAMES + 99],
                //         (double)sliding_window_buf[2 * TIME_FRAMES + 99],
                //         (double)sliding_window_buf[3 * TIME_FRAMES + 99],
                //         (double)sliding_window_buf[4 * TIME_FRAMES + 99]);
            }

            // Noise Explosion Flag: Alert if standard deviation is too small
            if (std_dev < 1.5f && (frame_counter % 50 == 0)) {
                LOG_WRN("[FLAG: SPECTROGRAM_LOW_STD] Std Dev is very low (%.3f). Noise features might be over-amplified!", (double)std_dev);
            }

            // 3. Normalize and Transpose from [NUM_MEL_BINS, TIME_FRAMES] -> [TIME_FRAMES, NUM_MEL_BINS]
            for (int t = 0; t < TIME_FRAMES; t++) {
                for (int m = 0; m < NUM_MEL_BINS; m++) {
                    int src_idx = m * TIME_FRAMES + t;
                    int dst_idx = t * NUM_MEL_BINS + m;
                    normalized_buf[dst_idx] = (sliding_window_buf[src_idx] - mean) / std_dev;
                }
            }

            dl::TensorBase float_sliding_tensor({1, TIME_FRAMES, NUM_MEL_BINS}, 
                                                normalized_buf, 
                                                0, 
                                                dl::DATA_TYPE_FLOAT);

            input_tensor->assign(&float_sliding_tensor);
            model->reset(); // Reset GRU hidden states to zero to match PyTorch's forward(inputs, None) evaluation
            model->run();   // Run the neural network
            last_inf_end = k_cycle_get_32();
            gpio_pin_set_dt(&sync_inf, 0); /* Set INF pin Low */
            last_inf_time_ms = (double)(last_inf_end - last_inf_start) / (sys_clock_hw_cycles_per_sec() / 1000.0);

            int8_t *int8_logits = (int8_t *)output_tensor->data;
            float scale = DL_SCALE(output_tensor->exponent);

            float fp32_logits[NUM_CLASSES];
            for (int c = 0; c < NUM_CLASSES; c++) {
                fp32_logits[c] = int8_logits[c] * scale;
            }

            compute_softmax(fp32_logits, probs, NUM_CLASSES);
        }

        /* Update BLE advertising payload with live class probabilities */
        ble_update_probabilities(probs, NUM_CLASSES);

        // ============================================================
        // DETECTION LOGIC - "nothing" should trigger when no keyword
        // ============================================================
        
        // 1. Find best class
        int best_class = 0;
        float best_prob = probs[0];
        for (int c = 1; c < NUM_CLASSES; c++) {
            if (probs[c] > best_prob) {
                best_prob = probs[c];
                best_class = c;
            }
        }

        // --- Hendriks MMSE 40-Bin Noise Estimator & Frame Audio Intensity ---
        double sum_sq_rms = 0.0;
        for (int i = 0; i < FRAME_LEN_SAMPLES; i++) {
            double sample_f = (double)audio_window[i];
            sum_sq_rms += sample_f * sample_f;
        }
        float frame_rms = (float)sqrt(sum_sq_rms / (double)FRAME_LEN_SAMPLES);

        static float noise_psd[NUM_MEL_BINS] = {0.0f};
        static bool noise_init = false;

        if (!noise_init) {
            for (int m = 0; m < NUM_MEL_BINS; m++) {
                noise_psd[m] = sliding_window_buf[m * TIME_FRAMES + (TIME_FRAMES - 1)];
            }
            noise_init = true;
        } else {
            // Speech Lock Guard: Only update noise PSD during silence (P(nothing) > 60%)
            bool is_silence = (probs[NOTHING_CLASS_INDEX] > 0.60f);
            float alpha = is_silence ? 0.95f : 1.00f;

            for (int m = 0; m < NUM_MEL_BINS; m++) {
                float current_bin = sliding_window_buf[m * TIME_FRAMES + (TIME_FRAMES - 1)];
                noise_psd[m] = alpha * noise_psd[m] + (1.0f - alpha) * current_bin;
            }
        }

        float avg_noise_psd = 0.0f;
        for (int m = 0; m < NUM_MEL_BINS; m++) {
            avg_noise_psd += noise_psd[m];
        }
        avg_noise_psd /= (float)NUM_MEL_BINS;

        // Pure mathematical conversion from Log-Mel dB noise floor to Linear Noise Power
        // Formula: Linear Noise = 10^(avg_noise_psd / 20.0) * 100.0 
        // Quiet room (-60 dB) -> 0.10 Linear Noise
        // Moderate noise (-40 dB) -> 1.00 Linear Noise
        // Loud AC Fan (-20 dB) -> 10.00 Linear Noise
        float linear_noise_power = powf(10.0f, avg_noise_psd / 20.0f) * 100.0f;
        if (linear_noise_power < 0.001f) linear_noise_power = 0.001f;

        /* Send structured GATT notification payload with PEAK confidence, RMS, and Hendriks Noise Floor */
        static int active_peak_class = -1;
        static float max_peak_prob = 0.0f;
        static float peak_frame_rms = 0.0f;
        static float peak_noise_power = 0.0f;
        static float peak_probs[NUM_CLASSES] = {0.0f};
        static char peak_gatt_payload[120] = {0};
        static bool payload_ready = false;

        bool is_keyword_candidate = (best_class >= 0 && best_class < NOTHING_CLASS_INDEX && best_prob > 0.40f);

        if (is_keyword_candidate) {
            if (frame_rms > peak_frame_rms) {
                peak_frame_rms = frame_rms;
            }
            if (linear_noise_power > peak_noise_power) {
                peak_noise_power = linear_noise_power;
            }

            // Global Burst Peak Winner: Track the single highest probability class across the entire speech burst!
            if (best_prob >= max_peak_prob) {
                active_peak_class = best_class;
                max_peak_prob = best_prob;
                for (int c = 0; c < NUM_CLASSES; c++) {
                    peak_probs[c] = probs[c];
                }
            }
            payload_ready = true;
        } else {
            // Keyword burst finished (silence): format and transmit ONLY the single winning peak over BLE!
            if (payload_ready && active_peak_class >= 0 && max_peak_prob >= 0.40f) {
                float send_rms = peak_frame_rms;
                if (send_rms < 0.1f) send_rms = frame_rms;
                if (send_rms < 0.1f) send_rms = win_rms;
                if (send_rms < 0.1f) send_rms = 1.0f;

                if (send_rms > 255.0f) send_rms = 255.0f;

                float send_noise = peak_noise_power;
                if (send_noise < 0.001f) send_noise = linear_noise_power;
                if (send_noise < 0.001f) send_noise = 0.05f;

                uint32_t ts_ms = k_uptime_get_32();

                /* Construct 10-Byte Bit-Packed Binary Packet */
                struct BleKwsPacket packet;
                packet.dev_id = (uint8_t)DEVICE_ID;
                packet.top_class = (uint8_t)active_peak_class;
                packet.conf = (uint8_t)(max_peak_prob * 100.0f + 0.5f);
                packet.rms = (uint8_t)(send_rms + 0.5f);
                packet.noise_x1k = (uint16_t)(send_noise * 1000.0f + 0.5f);
                packet.ts_ms = ts_ms;

                ble_send_binary(&packet, sizeof(packet));
                payload_ready = false;
            }
            active_peak_class = -1;
            max_peak_prob = 0.0f;
            peak_frame_rms = 0.0f;
            peak_noise_power = 0.0f;
        }

        // 2. Find second best for margin check
        float second_best = 0.0f;
        for (int c = 0; c < NUM_CLASSES; c++) {
            if (c != best_class && probs[c] > second_best) {
                second_best = probs[c];
            }
        }

        float confidence_margin = best_prob - second_best;

        // 3. Check if "nothing" is the best class
        bool is_nothing = (best_class == NOTHING_CLASS_INDEX);

        // 4. Check if "nothing" probability is significant
        bool nothing_is_high = (probs[NOTHING_CLASS_INDEX] > NOTHING_TRIGGER_THRESHOLD);
        (void)nothing_is_high;

        // 5. Find best keyword (for logging)
        float best_keyword_prob = 0.0f;
        int best_keyword_class = -1;
        for (int c = 0; c < NOTHING_CLASS_INDEX; c++) {
            if (probs[c] > best_keyword_prob) {
                best_keyword_prob = probs[c];
                best_keyword_class = c;
            }
        }

        if (log_details) {
            LOG_INF("Best: [%s] %.1f%%, Margin: %.2f | RMS: %.1f", 
                    CLASS_LABELS[best_class], best_prob * 100.0f, confidence_margin, (double)win_rms);
        }

        // ============================================================
        // CASE 1: "NOTHING" DETECTED - Silence or non-keyword speech
        // ============================================================
        if (log_details && cooldown_counter == 0 && is_nothing && (frame_counter % 50 == 0)) {
            LOG_INF("[SILENCE/OTHER] Nothing detected (%.1f%%) | Frame %d | RMS: %.1f", best_prob * 100.0f, frame_counter, (double)win_rms);
        }

        // ============================================================
        // CASE 2: KEYWORD DETECTED (With 2-consecutive-frames debounce filter)
        // ============================================================
        bool is_keyword_detected = false;
        int detected_class = -1;

        if (cooldown_counter == 0 && 
            best_class >= 0 && best_class < NOTHING_CLASS_INDEX &&    // IS a keyword
            best_prob > CONFIDENCE_THRESHOLD && 
            confidence_margin > CONFIDENCE_MARGIN) {
            
            is_keyword_detected = true;
            detected_class = best_class;
        }

        if (is_keyword_detected) {
            if (detected_class == last_detected_class) {
                consecutive_count++;
            } else {
                last_detected_class = detected_class;
                consecutive_count = 1;
            }
            dropout_count = 1; // Allow 1 frame of silence/noise/nothing dropout without resetting

            if (log_details) {
                LOG_INF("  [FLAG: DEBOUNCE] Keyword '%s' detected (Consecutive: %d/2, Prob: %.1f%%)", 
                        CLASS_LABELS[detected_class], consecutive_count, best_prob * 100.0f);
            }

            if (consecutive_count >= 2) {
                LOG_INF("==================================================");
                LOG_INF(" [FLAG: TRIGGER] >>> KEYWORD DETECTED: [%s] (%.2f%%) <<<", 
                        CLASS_LABELS[detected_class], best_prob * 100.0f);
                LOG_INF("  [Margin: %.2f, Nothing: %.1f%%]", 
                        confidence_margin, probs[NOTHING_CLASS_INDEX] * 100.0f);
                LOG_INF("==================================================");
                
                /*
                 * Keyword trigger logged locally on micro-controller UART console.
                 * BLE GATT notification is handled exclusively by Path A (Peak Event Summarizer)
                 * to ensure full RMS and Hendriks Noise Floor payload integrity.
                 */
                cooldown_counter = COOLDOWN_FRAMES; // Set 150ms cooldown
                consecutive_count = 0;              // Reset counter
                last_detected_class = -1;           // Reset last class
                dropout_count = 0;
            }
        } else {
            // No keyword detected. Check if we can tolerate the dropout (hangover time)
            if (dropout_count > 0 && last_detected_class != -1) {
                dropout_count--;
                if (log_details) {
                    LOG_INF("  [FLAG: DEBOUNCE] Tolerated 1 frame of dropout/silence for '%s' (Consecutive: %d/2)", 
                            CLASS_LABELS[last_detected_class], consecutive_count);
                }
            } else {
                // Too many frames of silence: reset consecutive count
                consecutive_count = 0;
                last_detected_class = -1;
                dropout_count = 0;
            }
        }

        // ============================================================
        // REGULAR STATUS UPDATE & HEARTBEAT
        // ============================================================
        if (log_details && frame_counter % 100 == 0) {
            LOG_INF("[HEARTBEAT] Frame %d | Status: %s | RMS: %.1f | Best: [%s] (%.1f%%)",
                    frame_counter,
                    (cooldown_counter > 0) ? "COOLDOWN" : (is_nothing ? "LISTENING (SILENCE)" : "LISTENING (ACTIVE)"),
                    (double)win_rms,
                    CLASS_LABELS[best_class], best_prob * 100.0f);
        } else if (log_details && frame_counter % 25 == 0) {
            if (cooldown_counter > 0) {
                LOG_INF("Listening... [COOLDOWN: %d frames left] | Frame: %d", cooldown_counter, frame_counter);
            } else if (!is_nothing && best_keyword_class >= 0) {
                LOG_INF("Listening... Active: [%s] (%.1f%%) | Frame: %d", 
                        CLASS_LABELS[best_keyword_class], best_keyword_prob * 100.0f, frame_counter);
            }
        }
    }

    if (cooldown_counter > 0) {
        cooldown_counter--;
    }

    if (log_details) {
        LOG_INF("[TIMING] Audio Processing: Start: %u, End: %u (time: %.3f ms)",
                last_ap_start, last_ap_end, last_ap_time_ms);
        if (last_inf_start != 0) {
            LOG_INF("[TIMING] Inference: Start: %u, End: %u (time: %.3f ms)",
                    last_inf_start, last_inf_end, last_inf_time_ms);
        }
    }
}
