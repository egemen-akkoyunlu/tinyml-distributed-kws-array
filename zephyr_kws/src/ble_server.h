/**
 * @file ble_server.h
 * @brief Bluetooth Low Energy (BLE) Server interface for Xiao ESP32-S3 Sense.
 *
 * This header declares the functions required to initialize, manage, and transmit
 * data over a custom BLE service. It supports both raw audio chunk streaming
 * and text-based KWS detection result transmission.
 *
 * @copyright Copyright (c) 2026 Seeed Studio
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef BLE_SERVER_H
#define BLE_SERVER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the Bluetooth controller and starts GAP advertising.
 *
 * This function enables the Bluetooth stack and registers the GAP advertising
 * packet containing the device name ("XIAO_SENSE_BLE").
 */
void ble_server_init(void);

/**
 * @brief Checks if a client is connected and has enabled notifications.
 *
 * @return true if a peer is connected and the Client Characteristic Configuration
 *         Descriptor (CCCD) has notifications enabled, false otherwise.
 */
bool ble_is_connected_and_ready(void);

/**
 * @brief Transmits raw PCM audio data chunks to the connected client.
 *
 * Splits the raw audio samples into BLE-friendly small chunks (e.g. 20 bytes/10 samples)
 * to avoid queue/heap congestion, and notifies the subscriber.
 *
 * @param data Pointer to the array of 16-bit PCM audio samples.
 * @param total_samples Total number of samples to transmit.
 */
void ble_send_audio_data(int16_t *data, int total_samples);

/**
 * @brief 10-Byte Bit-Packed Binary GATT Notification Packet
 */
struct __attribute__((packed)) BleKwsPacket {
    uint8_t  dev_id;      /**< Device ID (1, 2, 3) */
    uint8_t  top_class;   /**< Class Index (0=go, 1=stop, 2=left, 3=right, 4=nothing) */
    uint8_t  conf;        /**< Top Confidence (0 to 100%) */
    uint8_t  rms;         /**< Audio RMS intensity (0 to 255) */
    uint16_t noise_x1k;   /**< Linear Noise Floor * 1000 */
    uint32_t ts_ms;       /**< Hardware timestamp in ms (k_uptime_get_32()) */
};

/**
 * @brief Transmits raw binary packet to the connected client.
 *
 * @param data Pointer to binary data buffer.
 * @param len Length in bytes.
 */
void ble_send_binary(const void *data, uint16_t len);

/**
 * @brief Transmits a formatted detection string to the connected client.
 *
 * Sends a null-terminated string (e.g. keyword + confidence) over the custom
 * BLE GATT notification characteristic.
 *
 * @param str Null-terminated string to be sent.
 */
void ble_send_string(const char *str);

/**
 * @brief Updates the live probability scores in the BLE Manufacturer Advertising payload.
 *
 * @param probs Array of class probabilities (5 classes).
 * @param num_classes Number of classes in probs.
 */
void ble_update_probabilities(const float *probs, int num_classes);

#ifdef __cplusplus
}
#endif

#endif // BLE_SERVER_H