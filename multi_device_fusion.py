#!/usr/bin/env python3
"""
===============================================================================
Multi-Device Late Fusion with Triple BLE GATT Connections (3 Devices)
===============================================================================
Description:
  This script connects directly to all 3 edge microcontrollers over BLE GATT:
    - XIAO_SENSE_BLE_1 (Device 1)
    - XIAO_SENSE_BLE_2 (Device 2)
    - XIAO_SENSE_BLE_3 (Device 3)

Key Features:
  1. Guaranteed L2CAP Packet Delivery: Direct GATT notifications.
  2. Persistent Dual/Triple Reconnection: Maintains active GATT connections concurrently.
  3. Real-Time 3-Node Soft Late Fusion Consensus:
     Fuses predictions instantly upon notification and evaluates consensus keywords.

Usage:
  /home/acar/kws_env/bin/python3 multi_device_fusion.py
===============================================================================
"""

import asyncio
import time
import sys
import struct
import math
import numpy as np
from bleak import BleakScanner, BleakClient

# GATT Service & Characteristic UUIDs matching ble_server.c
CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

# Universal Overlap 25-Class MoE Vocabularies (ZERO 1-Node Words):
# Node 1: 25 Classes
DEV1_LABELS = [
    "backward", "bird", "dog", "eight", "five", "follow", "forward", "four", 
    "learn", "left", "nine", "no", "on", "one", "right", "seven", 
    "six", "three", "tree", "two", "up", "visual", "yes", "zero", "nothing"
]

# Node 2: 25 Classes
DEV2_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "go", 
    "happy", "house", "learn", "left", "marvin", "nine", "no", "off", 
    "one", "right", "sheila", "stop", "three", "tree", "wow", "yes", "nothing"
]

# Node 3: 25 Classes
DEV3_LABELS = [
    "bed", "cat", "down", "five", "follow", "forward", "four", "go", 
    "happy", "house", "marvin", "off", "on", "seven", "sheila", "six", 
    "stop", "three", "tree", "two", "up", "visual", "wow", "zero", "nothing"
]

DEVICE_VOCABULARIES = {
    1: DEV1_LABELS,
    2: DEV2_LABELS,
    3: DEV3_LABELS
}

# Master vocabulary (35 words + nothing)
CLASS_LABELS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", 
    "forward", "four", "go", "happy", "house", "learn", "left", "marvin", "nine", 
    "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", 
    "tree", "two", "up", "visual", "wow", "yes", "zero", "nothing"
]

# Late Fusion Trigger Threshold (72% sensitive threshold for distant/quiet speech)
FUSED_TRIGGER_THRESHOLD = 72.0

# Strict Single-Node High-Confidence Threshold (85% to eliminate single-node false alarms)
SINGLE_NODE_TRIGGER_THRESHOLD = 85.0

# Required consecutive consensus frames for trigger confirmation
REQUIRED_DEBOUNCE_COUNT = 1

# Debounce tracking variables
current_candidate_kw = None
candidate_debounce_count = 0
last_fired_fused_kw = None
last_print_time = 0.0

# Refractory Lockout Window (250 ms agile cooldown after trigger)
REFRACTORY_LOCKOUT_SEC = 0.250
last_trigger_time = 0.0

# Device heartbeat timeout in seconds
DEVICE_TIMEOUT_SEC = 3.0

# Peak-Hold Window duration in seconds (holds detection for 300 ms to bridge clock skew)
PEAK_HOLD_SEC = 0.30

# Maximum inter-device acoustic event alignment tolerance in seconds (250 ms)
MAX_INTER_DEVICE_SKEW_SEC = 0.250

# State dictionary for tracking live predictions, clock windows, and peak-hold across 3 devices
device_state = {
    "XIAO_SENSE_BLE_1": {
        "dev_id": 1,
        "last_seen": 0.0,
        "probs": [0.0, 0.0, 0.0, 0.0, 100.0],
        "top_class": "nothing",
        "top_prob": 100.0,
        "hold_kw": "nothing",
        "hold_prob": 0.0,
        "hold_time": 0.0,
        "clock_history": [],  # Sliding window of (T_device_norm, T_laptop_sec)
        "boot_offset": None,  # Initial offset to normalize hardware uptimes
        "alpha": 1.0,         # OLS Slope (Clock frequency ratio)
        "beta": 0.0,          # OLS Intercept (Clock offset)
        "master_time": 0.0,   # Synchronized Master Laptop Time
        "ppm_drift": 0.0      # Crystal drift in parts-per-million
    },
    "XIAO_SENSE_BLE_2": {
        "dev_id": 2,
        "last_seen": 0.0,
        "probs": [0.0, 0.0, 0.0, 0.0, 100.0],
        "top_class": "nothing",
        "top_prob": 100.0,
        "hold_kw": "nothing",
        "hold_prob": 0.0,
        "hold_time": 0.0,
        "clock_history": [],
        "boot_offset": None,
        "alpha": 1.0,
        "beta": 0.0,
        "master_time": 0.0,
        "ppm_drift": 0.0
    },
    "XIAO_SENSE_BLE_3": {
        "dev_id": 3,
        "last_seen": 0.0,
        "probs": [0.0, 0.0, 0.0, 0.0, 100.0],
        "top_class": "nothing",
        "top_prob": 100.0,
        "hold_kw": "nothing",
        "hold_prob": 0.0,
        "hold_time": 0.0,
        "clock_history": [],
        "boot_offset": None,
        "alpha": 1.0,
        "beta": 0.0,
        "master_time": 0.0,
        "ppm_drift": 0.0
    }
}


def parse_gatt_payload(raw_data):
    """
    Parses 10-byte binary BleKwsPacket struct or legacy ASCII GATT notification:
      10-Byte Struct (<BBBBHI):
        - dev_id      (uint8)  : Device ID (1, 2, 3)
        - top_class   (uint8)  : Class index (0=go, 1=stop, 2=left, 3=right, 4=nothing)
        - conf        (uint8)  : Confidence (0..100)
        - rms         (uint8)  : RMS intensity (0..255)
        - noise_x1k   (uint16) : Linear noise * 1000
        - ts_ms       (uint32) : Microcontroller hardware uptime in ms
    """
    try:
        if isinstance(raw_data, (bytes, bytearray)) and len(raw_data) == 10:
            dev_id, top_class_idx, conf, rms, noise_x1k, ts_ms = struct.unpack("<BBBBHI", raw_data)
            dev_labels = DEVICE_VOCABULARIES.get(dev_id, CLASS_LABELS)
            if 0 <= top_class_idx < len(dev_labels):
                top_kw = dev_labels[top_class_idx]
            else:
                top_kw = "nothing"
            top_conf = float(conf)
            rms_f = float(rms)
            noise_f = float(noise_x1k) / 1000.0
            probs = [0.0] * len(dev_labels)
            probs[top_class_idx] = top_conf
            return dev_id, top_kw, top_conf, rms_f, noise_f, ts_ms, probs
        else:
            payload_str = raw_data.decode("utf-8") if isinstance(raw_data, (bytes, bytearray)) else str(raw_data)
            parts = payload_str.split("|")
            dev_id = int(parts[0].split(":")[1])
            top_kw = parts[1].split(":")[1]
            top_conf = float(parts[2].split(":")[1])
            rms_f = 0.0
            noise_f = 0.0
            ts_ms = None
            probs_idx = 3

            if len(parts) >= 6 and "RMS:" in parts[3]:
                rms_f = float(parts[3].split(":")[1])
                noise_f = float(parts[4].split(":")[1])
                probs_idx = 5
                if len(parts) >= 7 and "TS:" in parts[5]:
                    ts_ms = int(parts[5].split(":")[1])
                    probs_idx = 6

            prob_strs = parts[probs_idx].split(":")[1].split(",")
            probs = [float(p) for p in prob_strs]
            return dev_id, top_kw, top_conf, rms_f, noise_f, ts_ms, probs
    except Exception:
        return None, None, None, None, None, None, None


def update_ols_clock_sync(dev_key, ts_ms, laptop_time):
    """
    Fits Adaptive Sliding Window Ordinary Least Squares (OLS) Linear Regression:
      Y = alpha * X + beta
    Where X = Microcontroller Hardware Time (sec), Y = Laptop Master Time (sec).
    Uses Initial Boot Offset Normalization so timestamps align starting on sample #1!
    """
    state = device_state[dev_key]
    if ts_ms is None:
        state["master_time"] = laptop_time
        return

    # Convert hardware timestamp (ms) to seconds
    t_dev = ts_ms / 1000.0

    # Initial Boot Offset Normalization
    if state["boot_offset"] is None:
        state["boot_offset"] = laptop_time - t_dev

    t_dev_norm = t_dev + state["boot_offset"]

    history = state["clock_history"]
    history.append((t_dev_norm, laptop_time))

    if len(history) > 20:
        history.pop(0)

    N = len(history)
    if N >= 3:
        sum_x = sum(x for x, y in history)
        sum_y = sum(y for x, y in history)
        mean_x = sum_x / N
        mean_y = sum_y / N

        num = sum((x - mean_x) * (y - mean_y) for x, y in history)
        den = sum((x - mean_x) ** 2 for x, y in history)

        if den > 1e-9:
            raw_alpha = num / den
            # Physical Quartz Crystal Clamp: Limit drift to +/- 500 ppm ([0.9995, 1.0005])
            # Prevents Bluetooth packet arrival jitter spikes from distorting the slope
            alpha = max(0.9995, min(1.0005, raw_alpha))
            beta = mean_y - alpha * mean_x
            state["alpha"] = alpha
            state["beta"] = beta
            state["ppm_drift"] = (alpha - 1.0) * 1e6
            state["master_time"] = alpha * t_dev_norm + beta
        else:
            state["master_time"] = t_dev_norm
    else:
        state["master_time"] = t_dev_norm


def update_peak_hold(dev_key, top_class, top_prob, rms, noise, ts_ms, probs, current_time):
    """
    Updates device state, Adaptive OLS clock sync, RMS, Hendriks noise floor, and 300 ms Peak-Hold buffer.
    """
    state = device_state[dev_key]
    state["last_seen"] = current_time
    state["top_class"] = top_class
    state["top_prob"] = top_prob
    state["rms"] = rms
    state["noise"] = noise
    state["probs"] = probs

    # Update Adaptive OLS Clock Sync
    update_ols_clock_sync(dev_key, ts_ms, current_time)

    # Pure Mathematical SNR: Confidence / (Linear Noise Floor + 0.001)
    snr = (top_prob + 1.0) / (noise + 0.001)
    state["snr"] = snr

    if top_class != "nothing" and top_prob >= 60.0:
        state["hold_kw"] = top_class
        state["hold_prob"] = top_prob
        state["hold_time"] = current_time
    else:
        if (current_time - state["hold_time"]) > PEAK_HOLD_SEC:
            state["hold_kw"] = "nothing"
            state["hold_prob"] = 0.0


def process_strict_consensus_fusion():
    """
    Evaluates SNR-Weighted Late Fusion Consensus across 3 Devices with Immediate Firing & Refractory Cooldown.
    """
    global current_candidate_kw, candidate_debounce_count, last_print_time, last_fired_fused_kw, last_trigger_time
    current_time = time.time()

    # Refractory Lockout Gate: Suppress echoes / mouth-release noise within 500ms of any trigger
    if (current_time - last_trigger_time) < REFRACTORY_LOCKOUT_SEC:
        return

    # Find all currently active devices (received data within DEVICE_TIMEOUT_SEC)
    active_devices = {}
    for dev_name, state in device_state.items():
        if (current_time - state["last_seen"]) < DEVICE_TIMEOUT_SEC:
            active_devices[dev_name] = state

    if not active_devices:
        return

    # Check for keyword matches among active devices using 300ms Peak-Hold windows
    node_preds = []
    for dev_name, state in active_devices.items():
        kw = state.get("hold_kw", state["top_class"])
        prob = state.get("hold_prob", state["top_prob"])
        dev_id = state.get("dev_id", 3)
        dev_vocab = DEVICE_VOCABULARIES.get(dev_id, CLASS_LABELS)
        rms = state.get("rms", 0.0)
        noise = state.get("noise", 0.001)
        m_time = state.get("master_time", current_time)
        ppm = state.get("ppm_drift", 0.0)

        # Calibrated Physical Acoustic SNR (Speech Energy vs Microphone Noise Floor):
        speech_power = (float(rms))**2
        noise_power = max(20.0, float(noise) * 1000.0)
        snr_db = 10.0 * math.log10(speech_power / noise_power) if rms > 0.0 else 0.0

        # Calibrated 4-Bit Integer Sigmoidal Weight (Center: 10 dB, T = 3.5, Scale: 1 to 15):
        sig_w = float(1.0 / (1.0 + math.exp(-(snr_db - 10.0) / 3.5)))
        weight_4bit = int(max(1, min(15, round(sig_w * 15.0))))
        if rms >= 240.0:
            weight_4bit = max(1, weight_4bit // 4)

        nothing_idx = dev_vocab.index("nothing") if "nothing" in dev_vocab else -1
        raw_probs = state.get("probs", [])
        nothing_prob = float(raw_probs[nothing_idx]) * 100.0 if (nothing_idx >= 0 and raw_probs and len(raw_probs) > nothing_idx) else 0.0

        # Normalized Shannon Predictive Entropy
        if raw_probs and len(raw_probs) > 0:
            p_arr = np.array(raw_probs, dtype=np.float32) / 100.0
            entropy = -float(np.sum(p_arr * np.log(p_arr + 1e-9)))
            norm_entropy = float(entropy / np.log(len(dev_vocab)))
        else:
            norm_entropy = 0.0

        node_preds.append({
            "dev_name": dev_name,
            "dev_id": dev_id,
            "kw": kw,
            "conf": prob,
            "weight_4bit": weight_4bit,
            "rms": rms,
            "noise": noise,
            "nothing_prob": nothing_prob,
            "entropy": norm_entropy,
            "m_time": m_time,
            "ppm": ppm,
            "vocab": dev_vocab
        })

    candidate_this_frame = None
    candidate_conf = 0.0
    matching_nodes = []
    trigger_tier = "CONSENSUS"

    # Dynamic Noise-Adaptive Multi-Node Threshold (Base: 60.0% + 14.0 * noise):
    avg_room_noise = sum(state.get("noise", 0.001) for state in active_devices.values()) / (len(active_devices) + 1e-6)
    dynamic_threshold = min(75.0, max(56.0, 60.0 + (14.0 * avg_room_noise)))
    applied_threshold = dynamic_threshold

    # Cross-Node Sibling Quorum Squelch: If >= 2 nodes hear voice energy (RMS >= 6.0) and classify as 'nothing',
    # it is non-keyword speech -> block all single-node triggers!
    silence_quorum = sum(1 for s in node_preds if s["rms"] >= 6.0 and s["nothing_prob"] >= 55.0) >= 2

    proposed_kws = set(p["kw"] for p in node_preds if p["kw"] != "nothing" and p["conf"] >= 45.0)
    vetoed_kws = set()

    # Universal Graph-Theoretic Arbiter Veto (Pure Graph Theory - Zero Hardcoding):
    # If a node 'p' has full vocabulary coverage over words {W_spec, W_cand} and confidently chooses W_spec (>= 70%),
    # any W_cand proposed by sibling nodes that LACK W_spec is an aliasing artifact -> VETO W_cand!
    for p in node_preds:
        spec_kw = p["kw"]
        spec_conf = p["conf"]
        spec_vocab = p["vocab"]

        if spec_kw != "nothing" and spec_conf >= 70.0:
            for other_kw in proposed_kws:
                if other_kw != spec_kw and other_kw in spec_vocab:
                    # Check if proposing nodes for other_kw lack spec_kw in their neural network
                    proposing_nodes_for_other = [n for n in node_preds if n["kw"] == other_kw]
                    lacking_spec_count = sum(1 for n in proposing_nodes_for_other if spec_kw not in n["vocab"])
                    if lacking_spec_count > 0:
                        vetoed_kws.add(other_kw)

    for kw in proposed_kws:
        # Apply veto if this keyword was blocked by an acoustic arbiter node
        if kw in vetoed_kws:
            continue

        capable_nodes = [dev_id for dev_id, vocab in DEVICE_VOCABULARIES.items() if kw in vocab[:-1]]
        rep_level = len(capable_nodes)
        agreeing_nodes = [p for p in node_preds if p["kw"] == kw and p["dev_id"] in capable_nodes]

        if rep_level >= 2:
            # Shared Words (Tiers 1 & 2): STRICT CONSENSUS ONLY (Requires >= 2 Nodes to Agree)
            if len(agreeing_nodes) >= 2:
                # Inter-device acoustic alignment gate
                master_times = [p["m_time"] for p in agreeing_nodes]
                if (max(master_times) - min(master_times)) > MAX_INTER_DEVICE_SKEW_SEC:
                    continue

                total_w = sum(p["weight_4bit"] for p in agreeing_nodes) + 1e-6
                fused_conf = sum(p["conf"] * p["weight_4bit"] for p in agreeing_nodes) / total_w

                # Consensus priority bonus (+20) ensures 2-node agreements always defeat single-node artifacts
                if fused_conf >= dynamic_threshold and (fused_conf + 20.0) > candidate_conf:
                    candidate_this_frame = kw
                    candidate_conf = fused_conf + 20.0
                    matching_nodes = agreeing_nodes
                    trigger_tier = f"{len(agreeing_nodes)}-NODE SNR CONSENSUS"
                    applied_threshold = dynamic_threshold
        else:
            # 1-Node Specialist Word (Tier 3) - Squelched if sibling quorum says non-keyword silence
            if len(agreeing_nodes) >= 1 and not silence_quorum:
                p = agreeing_nodes[0]
                snr_w = p["weight_4bit"]  # 1 to 15
                rms_val = p["rms"]

                # Physical Acoustic Proximity Scaling:
                if snr_w >= 8 or rms_val >= 35.0:
                    th = 75.0  # High SNR near-field speech -> high recall
                elif snr_w >= 4 or rms_val >= 20.0:
                    th = 82.0  # Moderate mid-field speech
                else:
                    th = 90.0  # Weak / far-field speech -> strict noise rejection

                th = max(dynamic_threshold, th)

                # Sibling OOD cross-validation (sibling nodes hearing voice energy classify it as 'nothing')
                sibling_ood = any(
                    s.get("nothing_prob", 0.0) >= 35.0 and s["rms"] >= 6.0 
                    for s in node_preds if s["dev_id"] != p["dev_id"]
                )

                if (p["conf"] >= th and p["entropy"] <= 0.30) and p["conf"] > candidate_conf:
                    candidate_this_frame = kw
                    candidate_conf = p["conf"]
                    matching_nodes = [p]
                    trigger_tier = f"SPECIALIST 1-NODE GATE (SNR-Scaled >= {th:.0f}%)" + (" (OOD VERIFIED)" if sibling_ood else "")
                    applied_threshold = th

    # Asymmetric Debounce: Instant 1-frame firing for consensus, 2-frame persistence for single-node
    if candidate_this_frame:
        if candidate_this_frame == current_candidate_kw:
            candidate_debounce_count += 1
        else:
            current_candidate_kw = candidate_this_frame
            candidate_debounce_count = 1

        required_debounce = 1 if "CONSENSUS" in trigger_tier else 2
        if candidate_debounce_count >= required_debounce:
            if candidate_this_frame != last_fired_fused_kw:
                last_fired_fused_kw = candidate_this_frame
                last_trigger_time = current_time  # Start 250ms refractory lockout
                master_times = [n["m_time"] for n in matching_nodes]
                max_skew_ms = (max(master_times) - min(master_times)) * 1000.0 if len(master_times) > 1 else 0.0
                display_conf = min(100.0, candidate_conf - 20.0) if "CONSENSUS" in trigger_tier else candidate_conf
                print(f"\n*** [FUSED TRIGGER: {candidate_this_frame.upper()} ({display_conf:.1f}% >= {applied_threshold:.0f}% | {trigger_tier} | OLS Skew: {max_skew_ms:.1f}ms)] ***")
                for p in matching_nodes:
                    master_ts_ms = int((p["m_time"] % 1000.0) * 1000.0)
                    print(f"    ├── Dev {p['dev_id']} ({p['dev_name']}): {candidate_this_frame.upper()} ({p['conf']:.1f}%) | Audio RMS: {p['rms']:.1f} | Linear Noise: {p['noise']:.3f} | 4-Bit Weight: {p['weight_4bit']}/15 | OLS TS: {master_ts_ms}ms | XTAL Drift: {p['ppm']:+.1f}ppm")
                # Clear peak-hold buffer across all active nodes to prevent residual echo triggers
                for dev_k in active_devices:
                    device_state[dev_k]["hold_kw"] = "nothing"
                    device_state[dev_k]["hold_prob"] = 0.0
                print()
    else:
        current_candidate_kw = None
        candidate_debounce_count = 0
        last_fired_fused_kw = None


fusion_timer_task = None

async def _delayed_fusion_eval():
    await asyncio.sleep(0.120)  # 120ms aggregation window
    process_strict_consensus_fusion()

def trigger_fusion_window():
    global fusion_timer_task
    try:
        loop = asyncio.get_running_loop()
        if fusion_timer_task is None or fusion_timer_task.done():
            fusion_timer_task = loop.create_task(_delayed_fusion_eval())
    except Exception:
        process_strict_consensus_fusion()


def notification_handler_dev1(sender, data):
    try:
        dev_id, top_kw, top_conf, rms, noise, ts_ms, probs = parse_gatt_payload(data)
        if dev_id == 1 and probs:
            update_peak_hold("XIAO_SENSE_BLE_1", top_kw, top_conf, rms, noise, ts_ms, probs, time.time())
            trigger_fusion_window()
    except Exception as e:
        print(f"[DEV1 GATT DECODE ERROR]: {e}")


def notification_handler_dev2(sender, data):
    try:
        dev_id, top_kw, top_conf, rms, noise, ts_ms, probs = parse_gatt_payload(data)
        if dev_id == 2 and probs:
            update_peak_hold("XIAO_SENSE_BLE_2", top_kw, top_conf, rms, noise, ts_ms, probs, time.time())
            trigger_fusion_window()
    except Exception as e:
        print(f"[DEV2 GATT DECODE ERROR]: {e}")


def notification_handler_dev3(sender, data):
    try:
        dev_id, top_kw, top_conf, rms, noise, ts_ms, probs = parse_gatt_payload(data)
        if dev_id == 3 and probs:
            update_peak_hold("XIAO_SENSE_BLE_3", top_kw, top_conf, rms, noise, ts_ms, probs, time.time())
            trigger_fusion_window()
    except Exception as e:
        print(f"[DEV3 GATT DECODE ERROR]: {e}")


ble_connect_lock = asyncio.Lock()


async def connect_and_listen(device_name, handler):
    """
    Persistent GATT connection task with lock to prevent BlueZ Operation In Progress collisions.
    """
    while True:
        try:
            async with ble_connect_lock:
                print(f"Scanning for {device_name}...")
                device = await BleakScanner.find_device_by_name(device_name, timeout=10.0)
                if not device:
                    print(f"[{device_name}] Device not found during scan. Retrying...")
                    await asyncio.sleep(1.5)
                    continue

                print(f"Connecting to {device_name} ({device.address})...")
                client = BleakClient(device, timeout=15.0)
                await client.connect()
                print(f"[SUCCESS] Connected to {device_name}! Waiting for GATT MTU setup...")
                await asyncio.sleep(0.3)

                # Retry loop for GATT notification subscription
                subscribed = False
                for attempt in range(3):
                    try:
                        await client.start_notify(CHAR_UUID, handler)
                        subscribed = True
                        print(f"[SUCCESS] Subscribed to GATT notifications on {device_name}!")
                        break
                    except Exception as sub_err:
                        print(f"[{device_name}] Notification subscription attempt {attempt + 1}/3 failed ({sub_err}). Retrying...")
                        await asyncio.sleep(0.5)

                if not subscribed:
                    raise RuntimeError(f"Failed to subscribe to GATT notifications on {device_name}")

            try:
                ping_counter = 0
                while client.is_connected:
                    await asyncio.sleep(1.0)
                    ping_counter += 1
                    if ping_counter >= 3:
                        ping_counter = 0
                        # Active GATT keep-alive ping to prevent Linux OS / BlueZ idle disconnects
                        try:
                            await client.read_gatt_char(CHAR_UUID)
                        except Exception:
                            pass
            finally:
                if client.is_connected:
                    await client.disconnect()
                print(f"[WARNING] Disconnected from {device_name}. Reconnecting...")
        except Exception as e:
            print(f"[{device_name} Connection Issue]: {e}. Retrying in 3 seconds...")
            await asyncio.sleep(3.0)


async def main():
    print("===============================================================================")
    print("  TRIPLE BLE GATT CONNECTED LATE FUSION RECEIVER (3 DEVICES)")
    print("===============================================================================")
    print("Connecting to XIAO_SENSE_BLE_1, XIAO_SENSE_BLE_2, and XIAO_SENSE_BLE_3 over GATT...\n")

    # Run triple GATT connection tasks concurrently
    task1 = asyncio.create_task(connect_and_listen("XIAO_SENSE_BLE_1", notification_handler_dev1))
    task2 = asyncio.create_task(connect_and_listen("XIAO_SENSE_BLE_2", notification_handler_dev2))
    task3 = asyncio.create_task(connect_and_listen("XIAO_SENSE_BLE_3", notification_handler_dev3))

    await asyncio.gather(task1, task2, task3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTriple GATT fusion receiver stopped cleanly.")
        sys.exit(0)
