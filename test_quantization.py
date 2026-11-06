"""Unit tests for LotteryFL subnetwork INT4 quantization."""

import unittest

import numpy as np
import torch
from torch import nn

from src.quantization import (
    dequantize_lottery_model,
    dequantize_masked_array,
    quantize_lottery_model,
    quantize_masked_array,
)


class TestLotteryQuantization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_masked_roundtrip_fidelity(self):
        weights = np.random.randn(64, 64).astype(np.float32)
        mask = (np.random.rand(64, 64) > 0.5).astype(np.float32)  # ~50% pruned

        quantized = quantize_masked_array(weights, mask, block_size=64)
        reconstructed = dequantize_masked_array(quantized)

        active = mask != 0
        self.assertTrue((reconstructed[~active] == 0.0).all())

        orig_active = weights[active]
        recon_active = reconstructed[active]
        cos_sim = np.dot(orig_active, recon_active) / (np.linalg.norm(orig_active) * np.linalg.norm(recon_active))
        self.assertGreater(cos_sim, 0.95)

    def test_empty_mask(self):
        weights = np.random.randn(32, 32).astype(np.float32)
        mask = np.zeros_like(weights)

        quantized = quantize_masked_array(weights, mask, block_size=64)
        reconstructed = dequantize_masked_array(quantized)

        self.assertEqual(quantized["num_active"], 0)
        self.assertTrue((reconstructed == 0.0).all())

    def test_unaligned_active_counts(self):
        weights = np.random.randn(100).astype(np.float32)
        mask = np.zeros(100, dtype=np.float32)
        mask[:53] = 1.0  # prime number of active parameters

        quantized = quantize_masked_array(weights, mask, block_size=64)
        reconstructed = dequantize_masked_array(quantized)

        self.assertEqual(quantized["num_active"], 53)
        self.assertTrue((reconstructed[53:] == 0.0).all())

    def test_lottery_model_roundtrip(self):
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        state_dict = model.state_dict()
        masks = [
            (np.random.rand(32, 64) > 0.4).astype(np.float32),
            (np.random.rand(10, 32) > 0.4).astype(np.float32),
        ]

        compressed = quantize_lottery_model(state_dict, masks, block_size=64)
        restored = dequantize_lottery_model(compressed)

        for key in state_dict:
            self.assertEqual(restored[key].shape, state_dict[key].shape)


if __name__ == "__main__":
    unittest.main()
