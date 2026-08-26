#ifndef MICROSD_H
#define MICROSD_H

#include <stdint.h>
#include <stddef.h>

/**
 * @brief Initialize the microSD card
 * @return 0 on success, -1 on failure
 */
int microsd_init(void);

/**
 * @brief Start recording audio to a WAV file
 * @param filename Name of the WAV file to create (e.g., "audio.wav")
 * @param sample_rate Sample rate in Hz
 * @param channels Number of channels (1 or 2)
 * @param bits_per_sample Bits per sample (16)
 * @return 0 on success, -1 on failure
 */
int microsd_start_recording(const char *filename, uint32_t sample_rate, 
                            uint16_t channels, uint16_t bits_per_sample);

/**
 * @brief Write audio samples to the WAV file
 * @param samples Pointer to audio samples (int16_t)
 * @param num_samples Number of samples to write
 * @return 0 on success, -1 on failure
 */
int microsd_write_audio(int16_t *samples, size_t num_samples);

/**
 * @brief Stop recording and finalize the WAV file
 * @return 0 on success, -1 on failure
 */
int microsd_stop_recording(void);

/**
 * @brief Check if recording is active
 * @return 1 if recording, 0 if not
 */
int microsd_is_recording(void);

#endif /* MICROSD_H */