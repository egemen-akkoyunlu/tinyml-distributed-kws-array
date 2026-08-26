#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include <cstring>
#include <cstdio>

#include "kws_config.hpp"
#include "audio_i2s.hpp"
#include "audio_processing.hpp"
#include "inference.hpp"
#include "ble_server.h"

LOG_MODULE_REGISTER(kws_app, LOG_LEVEL_INF);

/* Built-in LED: GPIO21, active-low (LED_ON = low, LED_OFF = high) */
#define LED0_NODE DT_ALIAS(led0)
static const struct gpio_dt_spec led = GPIO_DT_SPEC_GET(LED0_NODE, gpios);

int main(void)
{
    LOG_INF("==================================================");
    LOG_INF("  XIAO ESP32-S3 Sense: Streamable KWS (Zephyr + esp-dl)");
    LOG_INF("==================================================");

    /* --------------------------------------------------------
     * STARTUP BLINK: 3 fast blinks to confirm power-up and
     * firmware boot. Visible without a serial console.
     * -------------------------------------------------------- */
    if (device_is_ready(led.port)) {
        gpio_pin_configure_dt(&led, GPIO_OUTPUT_INACTIVE);
        for (int i = 0; i < 3; i++) {
            gpio_pin_set_dt(&led, 1);   /* LED ON  */
            k_sleep(K_MSEC(200));
            gpio_pin_set_dt(&led, 0);   /* LED OFF */
            k_sleep(K_MSEC(200));
        }
    }

    KWSInference kws_engine;
    if (!kws_engine.init()) {
        LOG_ERR("[FLAG: ERROR] KWS Inference engine initialization failed!");
        return -1;
    }

    if (!i2s_mic_init()) {
        LOG_ERR("[FLAG: ERROR] I2S PDM Microphone initialization failed!");
        return -1;
    }

    /* 
     * Initialize Bluetooth BLE Server.
     * This registers the custom BLE service and starts GAP advertising,
     * allowing the client board to discover and connect to us.
     */
    ble_server_init();

    LOG_INF("==================================================");
    LOG_INF("[FLAG: LIVE] Using real PDM microphone!");
    LOG_INF("Ready for real-time INT8 inference!");
    LOG_INF("==================================================");

    AudioProcessor audio_processor;

    int frame_counter = 0;
    int audio_buffer_index = 0;
    int16_t audio_pool[FRAME_SHIFT_SAMPLES] = {0};
    int16_t audio_window[FRAME_LEN_SAMPLES] = {0};

    while (1) {
        void *mem_block = nullptr;
        size_t block_size = 0;
        int ret = i2s_mic_read_block(&mem_block, &block_size);

        if (ret == 0 && mem_block != nullptr && block_size == I2S_BLOCK_SIZE) {
            uint8_t *byte_ptr = (uint8_t *)mem_block;
            
            for (int i = 0; i < SAMPLES_PER_BLOCK; i++) {
                int16_t raw_l = (int16_t)(byte_ptr[i * 8 + 2] | (byte_ptr[i * 8 + 3] << 8));
                int16_t raw_r = (int16_t)(byte_ptr[i * 8 + 6] | (byte_ptr[i * 8 + 7] << 8));
                
                int16_t clean_sample = audio_processor.process_sample_pair(raw_l, raw_r);

#if STREAM_RAW_PCM_MODE > 0
                int16_t stream_val = (STREAM_RAW_PCM_MODE == 1) ? raw_l : clean_sample;
                putchar(stream_val & 0xFF);
                putchar((stream_val >> 8) & 0xFF);
#else
                audio_pool[audio_buffer_index++] = clean_sample;

                if (audio_buffer_index >= FRAME_SHIFT_SAMPLES) {
                    audio_buffer_index = 0;

                    // Shift window
                    memmove(audio_window, audio_window + FRAME_SHIFT_SAMPLES, 
                            (FRAME_LEN_SAMPLES - FRAME_SHIFT_SAMPLES) * sizeof(int16_t));
                    
                    memcpy(audio_window + (FRAME_LEN_SAMPLES - FRAME_SHIFT_SAMPLES), 
                           audio_pool, FRAME_SHIFT_SAMPLES * sizeof(int16_t));

                    // --- Adaptive RMS VAD Noise-Floor Tracker ---
                    float sum_sq = 0.0f;
                    for (int s = 0; s < FRAME_SHIFT_SAMPLES; s++) {
                        sum_sq += (float)audio_pool[s] * (float)audio_pool[s];
                    }
                    float current_block_rms = sqrtf(sum_sq / (float)FRAME_SHIFT_SAMPLES);

                    static float ambient_noise_rms = 5.0f;
                    float adaptive_rms_threshold = ambient_noise_rms * 1.5f; // User multiplier 1.5x
                    if (adaptive_rms_threshold < 0.01f) {
                        adaptive_rms_threshold = 0.01f; // Safety floor
                    }

                    bool is_speech_active = (current_block_rms > adaptive_rms_threshold);

                    if (!is_speech_active) {
                        // Room is quiet: update ambient background noise floor safely
                        ambient_noise_rms = (0.995f * ambient_noise_rms) + (0.005f * current_block_rms);
                    }

                    bool log_details = (frame_counter % 100 == 0);
                    if (log_details) {
                        // LOG_INF("--- [FLAG: STAGE_1_START] Frame %d ---", frame_counter);
                    }

                    kws_engine.process_audio_window(audio_window, frame_counter, log_details);

                    frame_counter++;
                    if (frame_counter > 1000000) frame_counter = 0;
                }
#endif
            }
#if STREAM_RAW_PCM_MODE == 0
            audio_processor.check_and_report_diagnostics();
#endif
            i2s_mic_free_block(mem_block);
        } else {
            if (mem_block != nullptr) {
                i2s_mic_free_block(mem_block);
            }
            k_sleep(K_MSEC(1));
        }
    }
    return 0;
}