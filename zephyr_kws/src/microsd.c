#include "microsd.h"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/fs/fs.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/fs/fat_fs.h>
#include <string.h>
#include <stdio.h>

LOG_MODULE_REGISTER(microsd, LOG_LEVEL_INF);

#define WAV_HEADER_SIZE 44
#define MAX_FILENAME_LEN 64

static struct fs_file_t wav_file;
static bool is_recording = false;
static char current_filename[MAX_FILENAME_LEN];
static uint32_t total_samples_written = 0;
static uint32_t wav_sample_rate = 16000;
static uint16_t wav_channels = 1;
static uint16_t wav_bits_per_sample = 16;

// WAV header structure (packed for direct writing)
typedef struct __attribute__((packed)) {
    // RIFF header
    char chunk_id[4];           // "RIFF"
    uint32_t chunk_size;        // File size - 8
    char format[4];             // "WAVE"
    
    // fmt subchunk
    char subchunk1_id[4];       // "fmt "
    uint32_t subchunk1_size;    // 16 for PCM
    uint16_t audio_format;      // 1 for PCM
    uint16_t num_channels;      // 1 or 2
    uint32_t sample_rate;       // e.g., 16000
    uint32_t byte_rate;         // sample_rate * num_channels * bits_per_sample/8
    uint16_t block_align;       // num_channels * bits_per_sample/8
    uint16_t bits_per_sample;   // 16
    
    // data subchunk
    char subchunk2_id[4];       // "data"
    uint32_t subchunk2_size;    // Number of bytes of data
} wav_header_t;

static wav_header_t wav_header;

static void create_wav_header(uint32_t sample_rate, uint16_t channels, 
                              uint16_t bits_per_sample)
{
    memset(&wav_header, 0, sizeof(wav_header_t));
    
    // RIFF header
    memcpy(wav_header.chunk_id, "RIFF", 4);
    wav_header.chunk_size = 0;  // Will be filled later
    memcpy(wav_header.format, "WAVE", 4);
    
    // fmt subchunk
    memcpy(wav_header.subchunk1_id, "fmt ", 4);
    wav_header.subchunk1_size = 16;  // PCM
    wav_header.audio_format = 1;     // PCM
    wav_header.num_channels = channels;
    wav_header.sample_rate = sample_rate;
    wav_header.byte_rate = sample_rate * channels * (bits_per_sample / 8);
    wav_header.block_align = channels * (bits_per_sample / 8);
    wav_header.bits_per_sample = bits_per_sample;
    
    // data subchunk
    memcpy(wav_header.subchunk2_id, "data", 4);
    wav_header.subchunk2_size = 0;  // Will be filled later
    
    wav_sample_rate = sample_rate;
    wav_channels = channels;
    wav_bits_per_sample = bits_per_sample;
}

int microsd_init(void)
{
    LOG_INF("Initializing microSD card...");
    
    // Initialize the file system
    static FATFS fat_fs;
    static struct fs_mount_t mp = {
        .type = FS_FATFS,
        .fs_data = &fat_fs,
        .mnt_point = "/SD:",
    };
    
    int ret = fs_mount(&mp);
    if (ret < 0) {
        LOG_ERR("Failed to mount SD card: %d", ret);
        return -1;
    }
    
    LOG_INF("SD card mounted successfully");
    return 0;
}

int microsd_start_recording(const char *filename, uint32_t sample_rate, 
                            uint16_t channels, uint16_t bits_per_sample)
{
    if (is_recording) {
        LOG_WRN("Already recording, stopping previous recording");
        microsd_stop_recording();
    }
    
    // Construct full path
    snprintf(current_filename, sizeof(current_filename), "/SD:/%s", filename);
    
    LOG_INF("Starting recording to: %s", current_filename);
    
    // Open file for writing
    fs_file_t_init(&wav_file);
    int ret = fs_open(&wav_file, current_filename, FS_O_CREATE | FS_O_WRITE | FS_O_TRUNC);
    if (ret < 0) {
        LOG_ERR("Failed to open file: %d", ret);
        return -1;
    }
    
    // Create WAV header
    create_wav_header(sample_rate, channels, bits_per_sample);
    
    // Write header (will update later)
    ret = fs_write(&wav_file, &wav_header, sizeof(wav_header_t));
    if (ret < 0) {
        LOG_ERR("Failed to write WAV header: %d", ret);
        fs_close(&wav_file);
        return -1;
    }
    
    total_samples_written = 0;
    is_recording = true;
    
    LOG_INF("Recording started. Format: %d Hz, %d channel(s), %d-bit", 
            sample_rate, channels, bits_per_sample);
    
    return 0;
}

int microsd_write_audio(int16_t *samples, size_t num_samples)
{
    if (!is_recording) {
        LOG_WRN("Not recording, ignoring audio write");
        return -1;
    }
    
    if (samples == NULL || num_samples == 0) {
        return 0;
    }
    
    size_t bytes_to_write = num_samples * sizeof(int16_t);
    ssize_t bytes_written = fs_write(&wav_file, samples, bytes_to_write);
    
    if (bytes_written < 0) {
        LOG_ERR("Failed to write audio data: %d", (int)bytes_written);
        return -1;
    }
    
    total_samples_written += num_samples;
    
    return 0;
}

int microsd_stop_recording(void)
{
    if (!is_recording) {
        LOG_WRN("Not recording, nothing to stop");
        return 0;
    }
    
    LOG_INF("Stopping recording... Total samples: %d", total_samples_written);
    
    // Update WAV header with actual sizes
    uint32_t data_size = total_samples_written * sizeof(int16_t);
    uint32_t file_size = data_size + sizeof(wav_header_t) - 8;
    
    // Seek to beginning of file
    int ret = fs_seek(&wav_file, 0, FS_SEEK_SET);
    if (ret < 0) {
        LOG_ERR("Failed to seek to beginning: %d", ret);
        fs_close(&wav_file);
        is_recording = false;
        return -1;
    }
    
    // Update header fields
    wav_header.chunk_size = file_size;
    wav_header.subchunk2_size = data_size;
    
    // Write updated header
    ret = fs_write(&wav_file, &wav_header, sizeof(wav_header_t));
    if (ret < 0) {
        LOG_ERR("Failed to update WAV header: %d", ret);
        fs_close(&wav_file);
        is_recording = false;
        return -1;
    }
    
    // Close file
    fs_close(&wav_file);
    is_recording = false;
    
    LOG_INF("Recording stopped. File saved: %s", current_filename);
    LOG_INF("File size: %d bytes, Samples: %d", 
            file_size + 8, total_samples_written);
    
    return 0;
}

int microsd_is_recording(void)
{
    return is_recording ? 1 : 0;
}