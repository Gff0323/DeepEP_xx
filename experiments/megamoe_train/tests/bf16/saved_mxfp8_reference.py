"""Standalone saved-MXFP8 CPU reference for BF16 training.

Quantization helpers retained verbatim from train@6d8f833
(test/megamoe_w8a8_test/mega_moe_w8a8_golden.py). No W8A8 operator dependency.
"""

import math
import numpy as np


def float32_to_bf16_bits(data: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(data, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_round(data: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(data))


def make_fp8_e4m3fn_tables():
    decode = np.empty(256, dtype=np.float32)
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exponent = (raw >> 3) & 0xF
        mantissa = raw & 0x7
        if exponent == 0:
            value = mantissa * (2.0**-9)
        elif exponent == 0xF and mantissa == 0x7:
            value = math.nan
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        decode[raw] = sign * value
    positive_raw = np.array(
        [raw for raw in range(0x7F) if np.isfinite(decode[raw])],
        dtype=np.uint8,
    )
    return decode, positive_raw, decode[positive_raw]


FP8_DECODE, FP8_POSITIVE_RAW, FP8_POSITIVE_VALUES = make_fp8_e4m3fn_tables()


def encode_fp8_e4m3fn(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    absolute = np.minimum(np.abs(values), FP8_POSITIVE_VALUES[-1])
    upper_index = np.searchsorted(FP8_POSITIVE_VALUES, absolute, side="left")
    upper_index = np.clip(upper_index, 0, FP8_POSITIVE_VALUES.size - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    lower_diff = absolute - FP8_POSITIVE_VALUES[lower_index]
    upper_diff = FP8_POSITIVE_VALUES[upper_index] - absolute
    choose_upper = upper_diff < lower_diff
    tie = upper_diff == lower_diff
    upper_even = (FP8_POSITIVE_RAW[upper_index] & 1) == 0
    choose_upper |= tie & upper_even
    selected = np.where(choose_upper, upper_index, lower_index)
    raw = FP8_POSITIVE_RAW[selected].copy()
    raw |= np.where(np.signbit(values), 0x80, 0).astype(np.uint8)
    raw = np.where(np.isnan(values), 0x7F, raw).astype(np.uint8)
    return raw


def decode_fp8_e4m3fn(raw: np.ndarray) -> np.ndarray:
    return FP8_DECODE[np.asarray(raw, dtype=np.uint8)]


def mxfp8_quantize(data: np.ndarray, axis: int):
    """Return E4M3 raw data, dequantized FP32 data, and unpacked E8M0 scales."""
    source = np.asarray(data, dtype=np.float32)
    axis = axis % source.ndim
    moved = np.moveaxis(source, axis, -1)
    original_length = moved.shape[-1]
    padded_length = ((original_length + 31) // 32) * 32
    if padded_length != original_length:
        moved = np.pad(
            moved,
            [(0, 0)] * (moved.ndim - 1) + [(0, padded_length - original_length)],
        )
    blocks = moved.reshape(*moved.shape[:-1], padded_length // 32, 32)
    max_abs = np.max(np.abs(blocks), axis=-1)
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log2(max_abs)).astype(np.float32) - 8.0
    exponent = np.where(max_abs == 0, -127.0, exponent)
    exponent = np.clip(exponent, -127.0, 127.0).astype(np.int16)
    scale = np.exp2(exponent.astype(np.float32))
    normalized = blocks / scale[..., None]
    raw_blocks = encode_fp8_e4m3fn(normalized)
    dequantized_blocks = decode_fp8_e4m3fn(raw_blocks) * scale[..., None]

    raw_moved = raw_blocks.reshape(*moved.shape)[..., :original_length]
    dequantized_moved = dequantized_blocks.reshape(*moved.shape)[..., :original_length]
    raw = np.moveaxis(raw_moved, -1, axis)
    dequantized = np.moveaxis(dequantized_moved, -1, axis)
    scale_raw = (exponent + 127).astype(np.uint8)
    scale_raw = np.moveaxis(scale_raw, -1, axis)
    scale_value = np.moveaxis(scale, -1, axis)
    return raw, dequantized, scale_raw, scale_value
