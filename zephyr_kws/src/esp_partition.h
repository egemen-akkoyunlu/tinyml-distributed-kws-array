/*
 * Copyright (c) 2026 Seeed Studio / Senior ML Engineer
 * SPDX-License-Identifier: Apache-2.0
 *
 * @file esp_partition.h
 * @brief Zephyr RTOS compatibility wrapper for ESP-IDF's esp_partition.h.
 *
 * ESP-DL's Model loader (fbs_loader) relies on the ESP-IDF partition API
 * to locate, map, and read model binaries from flash storage partitions.
 * 
 * In this Zephyr implementation, we compile and store the model directly in 
 * the Flash Read-Only Data (RODATA) segment. Therefore, we do not have a physical 
 * ESP-IDF partition table.
 * 
 * This compatibility header stub out the partition mapping APIs cleanly:
 * - esp_partition_find_first: Always returns NULL (no partition found).
 * - esp_partition_mmap: Bypasses the partition lookup and maps the memory address 
 *   offset directly to the output pointer.
 * - esp_partition_munmap: Idle stub.
 */

#pragma once

#include <zephyr/kernel.h>
#include <stddef.h>
#include <stdint.h>
#include "spi_flash_mmap.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Handle representing a mapped memory region.
 */
typedef uint32_t esp_partition_mmap_handle_t;

/**
 * @brief ESP-IDF style partition descriptor structure.
 */
typedef struct {
    uint32_t size;      /**< Partition size in bytes */
    const char *label;  /**< Partition text label */
} esp_partition_t;

/**
 * @brief ESP-IDF style partition type enumeration.
 */
typedef enum {
    ESP_PARTITION_TYPE_DATA = 0
} esp_partition_type_t;

/**
 * @brief ESP-IDF style partition subtype enumeration.
 */
typedef enum {
    ESP_PARTITION_SUBTYPE_ANY = 0
} esp_partition_subtype_t;

/**
 * @brief Re-map the spi_flash memory structure.
 */
typedef spi_flash_mmap_memory_t esp_partition_mmap_memory_t;
#define ESP_PARTITION_MMAP_DATA SPI_FLASH_MMAP_DATA

/**
 * @brief Stubs partition lookup.
 * @return Always returns NULL since partitions are not managed via tables in Zephyr.
 */
static inline const esp_partition_t *esp_partition_find_first(esp_partition_type_t type,
                                                              esp_partition_subtype_t subtype,
                                                              const char *label)
{
    return NULL;
}

/**
 * @brief Maps a flash offset to a virtual memory address.
 * 
 * In our RODATA-based model setup, the offset parameter is the direct memory pointer.
 * Therefore, we assign the offset directly to the output pointer.
 *
 * @param partition Unused.
 * @param offset Direct flash pointer offset.
 * @param size Unused.
 * @param memory Unused.
 * @param out_ptr Pointer to receive the mapped memory address.
 * @param out_handle Pointer to receive the mapping handle.
 * @return Always returns 0 (Success).
 */
static inline int esp_partition_mmap(const esp_partition_t *partition, uint32_t offset,
                                     uint32_t size, esp_partition_mmap_memory_t memory,
                                     const void **out_ptr, esp_partition_mmap_handle_t *out_handle)
{
    *out_ptr = (const void *)offset;
    *out_handle = 1;
    return 0;
}

/**
 * @brief Unmaps a previously mapped flash memory region.
 * @param handle Unused.
 */
static inline void esp_partition_munmap(esp_partition_mmap_handle_t handle)
{
    (void)handle;
}

#ifdef __cplusplus
}
#endif
