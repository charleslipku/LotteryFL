"""Sub-byte INT4 quantization utilities for pruned lottery ticket subnetworks."""

import numpy as np
import torch


def quantize_masked_array(
    weights: np.ndarray,
    mask: np.ndarray,
    block_size: int = 64,
) -> dict:
    """Quantize non-zero pruned weights into packed 4-bit representations.

    Only active parameters where mask != 0 are quantized and packed, compounding
    the bandwidth savings of lottery pruning with 4-bit wire compression.
    """
    flat_weights = weights.flatten()
    flat_mask = mask.flatten()

    active_indices = np.nonzero(flat_mask)[0]
    active_weights = flat_weights[active_indices]

    num_active = len(active_weights)
    if num_active == 0:
        return {
            "packed": np.empty(0, dtype=np.uint8),
            "scales": np.empty(0, dtype=np.float32),
            "shape": weights.shape,
            "num_active": 0,
            "total_params": len(flat_weights),
            "active_indices": active_indices,
            "block_size": block_size,
        }

    pad_len = (block_size - (num_active % block_size)) % block_size
    if pad_len > 0:
        padded_active = np.pad(active_weights, (0, pad_len), mode="constant", constant_values=0.0)
    else:
        padded_active = active_weights

    blocks = padded_active.reshape(-1, block_size)
    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 7.0
    safe_scales = np.where(scales == 0.0, 1.0, scales)

    scaled_blocks = blocks / safe_scales
    quantized_blocks = np.clip(np.round(scaled_blocks), -7, 7)

    unsigned_vals = (quantized_blocks + 7).astype(np.uint8).flatten()
    if len(unsigned_vals) % 2 != 0:
        unsigned_vals = np.pad(unsigned_vals, (0, 1), mode="constant", constant_values=0)

    low = unsigned_vals[0::2] & 0x0F
    high = (unsigned_vals[1::2] & 0x0F) << 4
    packed = low | high

    return {
        "packed": packed,
        "scales": scales.flatten().astype(np.float32),
        "shape": weights.shape,
        "num_active": num_active,
        "total_params": len(flat_weights),
        "active_indices": active_indices,
        "block_size": block_size,
    }


def dequantize_masked_array(quantized_data: dict) -> np.ndarray:
    """Restore weight array from packed 4-bit active subnetwork parameters."""
    shape = quantized_data["shape"]
    num_active = quantized_data["num_active"]
    if num_active == 0:
        return np.zeros(shape, dtype=np.float32)

    packed = quantized_data["packed"]
    scales = quantized_data["scales"]
    block_size = quantized_data["block_size"]
    active_indices = quantized_data["active_indices"]

    low = (packed & 0x0F).astype(np.int8) - 7
    high = ((packed >> 4) & 0x0F).astype(np.int8) - 7

    unpacked = np.empty(len(packed) * 2, dtype=np.float32)
    unpacked[0::2] = low
    unpacked[1::2] = high

    total_padded = len(scales) * block_size
    unpacked = unpacked[:total_padded]

    blocks = unpacked.reshape(-1, block_size)
    rescaled_blocks = blocks * scales[:, np.newaxis]
    restored_active = rescaled_blocks.flatten()[:num_active]

    flat_restored = np.zeros(quantized_data["total_params"], dtype=np.float32)
    flat_restored[active_indices] = restored_active
    return flat_restored.reshape(shape)


def quantize_lottery_model(model_state: dict, masks: list, block_size: int = 64) -> dict:
    """Compress subnetwork weights in model state dict using lottery masks."""
    step = 0
    compressed: dict = {}
    for name, tensor in model_state.items():
        if "weight" in name and step < len(masks) and masks[step] is not None:
            weights = tensor.cpu().numpy()
            compressed[name] = quantize_masked_array(weights, masks[step], block_size=block_size)
            step += 1
        else:
            compressed[name] = tensor.cpu().numpy()
    return compressed


def dequantize_lottery_model(compressed_model: dict) -> dict:
    """Restore model weights from compressed lottery model state."""
    restored: dict = {}
    for name, data in compressed_model.items():
        if isinstance(data, dict) and "packed" in data:
            restored[name] = torch.from_numpy(dequantize_masked_array(data))
        elif isinstance(data, np.ndarray):
            restored[name] = torch.from_numpy(data)
        else:
            restored[name] = data
    return restored
