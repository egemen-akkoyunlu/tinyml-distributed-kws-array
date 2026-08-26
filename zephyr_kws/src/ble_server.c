/**
 * @file ble_server.c
 * @brief Bluetooth Low Energy (BLE) Server implementation for Xiao ESP32-S3 Sense.
 *
 * Implements a GATT Server with a custom service for transmitting KWS results
 * (and raw audio logs) using Bluetooth Notifications (notify property).
 * Handles connection status tracking and transmission rate-limiting/chunking
 * to prevent stack overflow.
 *
 * @copyright Copyright (c) 2026 Seeed Studio
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ble_server.h"
#include <zephyr/kernel.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>
#include <stdio.h>

/* Custom UUID for the primary KWS BLE Service */
#define CUSTOM_SERVICE_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef0))

/* Custom UUID for the notification characteristic */
#define CUSTOM_CHAR_UUID \
    BT_UUID_DECLARE_128(BT_UUID_128_ENCODE(0x12345678, 0x1234, 0x5678, 0x1234, 0x56789abcdef1))

/* State variables tracking active connection and subscriber subscription state */
static bool is_connected = false;
static bool notify_enabled = false;
static struct bt_conn *default_conn = NULL;

/**
 * @brief Callback invoked when the Client Characteristic Configuration Descriptor (CCCD) changes.
 *
 * Triggered when the connected peer enables or disables notifications.
 */
static void ccc_cfg_changed(const struct bt_gatt_attr *attr, uint16_t value) {
    notify_enabled = (value == BT_GATT_CCC_NOTIFY);
    printf("[BLE] Client notification (Notify) state changed: %s!\n", notify_enabled ? "ENABLED" : "DISABLED");
}

/* Primary Service and Characteristic definitions using Zephyr's macro API */
BT_GATT_SERVICE_DEFINE(my_custom_svc,
    BT_GATT_PRIMARY_SERVICE(CUSTOM_SERVICE_UUID),
    BT_GATT_CHARACTERISTIC(CUSTOM_CHAR_UUID, BT_GATT_CHRC_NOTIFY, BT_GATT_PERM_NONE, NULL, NULL, NULL),
    BT_GATT_CCC(ccc_cfg_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE)
);

#include "kws_config.hpp"

static char ble_dev_name[] = CONFIG_CUSTOM_DEVICE_NAME;

/* 
 * Live Probability Payload inside BLE Manufacturer Advertising Data:
 * Index 0-1: Company Identifier (0xFFFF - Special Test ID)
 * Index 2:   Device ID (1 for XIAO_SENSE_BLE_1, 2 for XIAO_SENSE_BLE_2)
 * Index 3:   Sequence Counter
 * Index 4..8: Class Probabilities in % (0-100) for [go, stop, left, right, nothing]
 */
static uint8_t mfg_data[9] = { 0xFF, 0xFF, DEVICE_ID, 0, 0, 0, 0, 0, 100 };

/* GAP Advertising packets config */
static struct bt_data ad[] = {
    BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
    BT_DATA(BT_DATA_MANUFACTURER_DATA, mfg_data, sizeof(mfg_data)),
};

/* Scan Response packet carrying the custom device name */
static struct bt_data sd[] = {
    BT_DATA(BT_DATA_NAME_COMPLETE, ble_dev_name, sizeof(ble_dev_name) - 1),
};

/* 
 * Fast BLE Advertising Parameters:
 * 40 - 80 units (25ms to 50ms interval) for ultra-fast, low-latency late fusion updates.
 */
#define MY_ADV_PARAM BT_LE_ADV_PARAM(BT_LE_ADV_OPT_CONN, \
                                     40, \
                                     80, \
                                     NULL)

/**
 * @brief Callback invoked when a connection is established.
 */
static void connected(struct bt_conn *conn, uint8_t err) {
    if (!err) {
        if (default_conn != NULL) {
            bt_conn_unref(default_conn);
        }
        default_conn = bt_conn_ref(conn);
        is_connected = true;
        printf("\n[BLE] >>> CENTRAL DEVICE CONNECTED (%s)! <<<\n", ble_dev_name);
    }
}

/**
 * @brief Callback invoked when a connection is terminated.
 */
static void disconnected(struct bt_conn *conn, uint8_t reason) {
    if (default_conn != NULL) {
        bt_conn_unref(default_conn);
        default_conn = NULL;
    }
    is_connected = false;
    notify_enabled = false;
    printf("\n[BLE] --- CONNECTION LOST (Reason: 0x%02X) ---\n", reason);
    
    /* Restart advertising */
    int err = bt_le_adv_start(MY_ADV_PARAM, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    if (err == 0) {
        printf("[BLE] Advertising restarted for %s.\n", ble_dev_name);
    } else {
        printf("[BLE ERROR] Failed to restart advertising (err: %d)\n", err);
    }
}

/* Register connection callbacks */
BT_CONN_CB_DEFINE(conn_callbacks) = {
    .connected = connected,
    .disconnected = disconnected,
};

void ble_server_init(void) {
    /* Initialize the Bluetooth Subsystem */
    int err = bt_enable(NULL);
    if (err) {
        printf("[BLE ERROR] Initialization failed (err: %d)\n", err);
        return;
    }

    bt_set_name(CONFIG_CUSTOM_DEVICE_NAME);
    
    /* Start advertising our presence to scanning clients */
    err = bt_le_adv_start(MY_ADV_PARAM, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
    if (err == 0) {
        printf("[BLE] Advertising started for [%s] (ID: %d).\n", CONFIG_CUSTOM_DEVICE_NAME, DEVICE_ID);
    } else {
        printf("[BLE ERROR] Failed to start advertising (err: %d)\n", err);
    }
}

bool ble_is_connected_and_ready(void) {
    return (is_connected && notify_enabled);
}

void ble_send_audio_data(int16_t *data, int total_samples) {
    int samples_sent = 0;
    
    /* 
     * MEMORY AND FLOW CONTROL TUNING:
     * High data rates can saturate the HCI TX ring buffers, causing memory exhaustion
     * (-ENOMEM/-ENOBUFS). We chunk data into 10 samples (20 Bytes), which fits
     * cleanly within the standard BLE MTU (23 bytes) without fragmentation.
     */
    int chunk_size = 10; 
    int enomem_count = 0;

    printf("\n[BLE] Transmitting %d PCM audio samples over BLE...\n", total_samples);

    while (samples_sent < total_samples && ble_is_connected_and_ready()) {
        int to_send = total_samples - samples_sent;
        if (to_send > chunk_size) {
            to_send = chunk_size;
        }

        /* Send notification using characteristic index 2 */
        int ret = bt_gatt_notify(NULL, &my_custom_svc.attrs[2], &data[samples_sent], to_send * sizeof(int16_t));

        if (ret == 0) {
            samples_sent += to_send;
            enomem_count = 0; /* Reset error counter on success */
            k_sleep(K_MSEC(5)); /* Brief sleep to let the RF controller process the packet */
            
        } else if (ret == -ENOMEM || ret == -ENOBUFS || ret == -EAGAIN) {
            /* Congestion detected: back off and wait for TX buffers to clear */
            enomem_count++;
            if (enomem_count % 50 == 0) {
                printf("[BLE WARNING] Buffer congestion! Retrying at sample %d...\n", samples_sent);
            }
            k_sleep(K_MSEC(20)); /* Sleep longer to allow channel clearance */
            
        } else {
            /* Fatal transmission error */
            printf("[BLE FATAL] Notification rejected! (err: %d)\n", ret);
            k_sleep(K_MSEC(100));
        }
    }
    printf("\n[BLE] Transmission complete! All audio packets sent.\n");
}

void ble_send_binary(const void *data, uint16_t len) {
    if (is_connected && default_conn != NULL && data != NULL && len > 0) {
        /* Send binary packet notification using valid connection handle default_conn and value attr attrs[2] */
        bt_gatt_notify(default_conn, &my_custom_svc.attrs[2], data, len);
    }
}

void ble_send_string(const char *str) {
    if (is_connected && default_conn != NULL) {
        int len = strlen(str);
        /* Send string notification using valid connection handle default_conn and value attr attrs[2] */
        bt_gatt_notify(default_conn, &my_custom_svc.attrs[2], str, len);
    }
}

void ble_update_probabilities(const float *probs, int num_classes) {
    if (!probs) return;
    mfg_data[2] = DEVICE_ID;
    mfg_data[3]++; // Increment sequence counter
    for (int i = 0; i < 5 && i < num_classes; i++) {
        int pct = (int)(probs[i] * 100.0f + 0.5f);
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        mfg_data[4 + i] = (uint8_t)pct;
    }
    bt_le_adv_update_data(ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
}