# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for USD texture export format normalization."""

from __future__ import annotations

import unittest

import numpy as np

from newton._src.viewer.viewer_usd import _ensure_texture_rgba_uint8


class TestViewerUsdTextureFormat(unittest.TestCase):
    def test_rgb_uint8_promoted_to_rgba(self):
        rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        rgba = _ensure_texture_rgba_uint8(rgb)

        self.assertEqual(rgba.shape, (1, 2, 4))
        self.assertEqual(rgba.dtype, np.uint8)
        self.assertTrue(rgba.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(rgba[..., :3], rgb)
        np.testing.assert_array_equal(rgba[..., 3], 255)

    def test_rgb_promotion_does_not_mutate_source(self):
        rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
        original = rgb.copy()
        _ = _ensure_texture_rgba_uint8(rgb)
        np.testing.assert_array_equal(rgb, original)

    def test_rgba_uint8_unchanged(self):
        rgba = np.array([[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8)
        out = _ensure_texture_rgba_uint8(rgba)
        np.testing.assert_array_equal(out, rgba)
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_float_texture_preserved(self):
        floats = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
        out = _ensure_texture_rgba_uint8(floats)
        np.testing.assert_array_equal(out, floats)


if __name__ == "__main__":
    unittest.main()
