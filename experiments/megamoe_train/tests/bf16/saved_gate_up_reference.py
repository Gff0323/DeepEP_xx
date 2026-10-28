"""CPU reference for saved MXFP8, using the existing W8A8 quantization rule."""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def w8a8_reference():
    import saved_mxfp8_reference

    return saved_mxfp8_reference


def compare_mxfp8_blocks(raw, scale_raw, bf16_reference, *, context=""):
    """Compare sampled whole groups against CPU GMM1 + W8A8 quantization.

    CPU/device GMM1 may round BF16 differently. Allow one FP8 normal-bin ULP
    (plus the BF16 tolerance), and allow adjacent scales only near an exponent
    boundary. This is a storage smoke check, not a gradient-accuracy test.
    """
    import numpy as np

    ref = w8a8_reference()
    raw = np.asarray(raw, dtype=np.uint8)
    scale_raw = np.asarray(scale_raw, dtype=np.uint8)
    bf16_reference = np.asarray(bf16_reference, dtype=np.float32)
    if raw.shape != bf16_reference.shape or scale_raw.shape != raw.shape[:-1] or raw.shape[-1] != 32:
        raise AssertionError(
            f"{context} invalid MXFP8 block shapes: raw={raw.shape} "
            f"scales={scale_raw.shape} reference={bf16_reference.shape}"
        )

    def fail(reason, mask, actual=None):
        index = tuple(int(i) for i in np.argwhere(mask)[0])
        group = index[:-1]
        details = (
            f"{context} saved MXFP8 {reason}: sample_index={index} "
            f"raw=0x{int(raw[index]):02x} scale=0x{int(scale_raw[group]):02x} "
            f"bf16_reference={float(bf16_reference[index])}"
        )
        if actual is not None:
            details += (
                f" actual={float(actual[index])} expected={float(expected[index])} "
                f"expected_scale=0x{int(expected_scales[group]):02x}"
            )
        details += "\nraw_group_hex=" + " ".join(f"{int(v):02x}" for v in raw[group])
        if actual is not None:
            details += "\nexpected_group_hex=" + " ".join(f"{int(v):02x}" for v in expected_raw[group])
        raise AssertionError(details)

    if not np.all(np.isfinite(bf16_reference)):
        fail("non-finite CPU GMM1 reference", ~np.isfinite(bf16_reference))
    expected_raw, expected, expected_scales, scale = ref.mxfp8_quantize(bf16_reference, -1)
    expected_scales = expected_scales.squeeze(-1)
    scale = scale.squeeze(-1)
    if np.any(scale_raw == 255):
        fail("non-finite E8M0 scale", np.broadcast_to((scale_raw == 255)[..., None], raw.shape))
    decoded_scale = np.exp2(scale_raw.astype(np.float32) - 127)
    with np.errstate(over="ignore", invalid="ignore"):
        actual = ref.decode_fp8_e4m3fn(raw) * decoded_scale[..., None]
    nan_codes = (raw & 0x7F) == 0x7F
    if np.any(nan_codes):
        fail("NaN FP8 encoding (0x7f/0xff)", nan_codes, actual)
    if not np.all(np.isfinite(actual)):
        fail("non-finite dequantized value", ~np.isfinite(actual), actual)
    magnitude = np.max(np.abs(bf16_reference), axis=-1)
    boundary = np.exp2(expected_scales.astype(np.float32) - 127 + 8)
    near_boundary = (np.abs(magnitude - boundary) <= 0.02 * boundary) | (
        np.abs(magnitude - 2 * boundary) <= 0.04 * boundary
    )
    delta = np.abs(scale_raw.astype(np.int16) - expected_scales.astype(np.int16))
    bad_scale = (delta != 0) & ~((delta == 1) & near_boundary)
    if np.any(bad_scale):
        fail("scale disagrees with sampled GMM1 groups", np.broadcast_to(bad_scale[..., None], raw.shape), actual)
    tolerance = 0.135 * np.abs(expected) + 0.01 + scale[..., None] * (2.0**-9)
    bad_value = np.abs(actual - expected) > tolerance
    if np.any(bad_value):
        fail(f"values mismatch (max_abs_error={np.max(np.abs(actual - expected))})", bad_value, actual)


def pack_clip_masks(gate, up, limit):
    """Raw BF16 values (represented as FP32), never decoded MXFP8 values.

    Each uint32 word stores 32 columns, bit 0 first. Pad invalid tail bits
    with zero. Equality to the FP32 clipping boundary is not clipped.
    """
    gate, up = np.asarray(gate, dtype=np.float32), np.asarray(up, dtype=np.float32)
    if gate.shape != up.shape or gate.ndim != 2:
        raise ValueError("gate and up must have matching [rows, columns] shapes")
    limit = np.float32(limit)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("clip mask reference requires a finite positive FP32 limit")

    def pack(bits):
        packed = np.packbits(np.pad(bits, ((0, 0), (0, (-bits.shape[1]) % 32))), axis=-1, bitorder="little")
        return packed.copy().view("<u4")

    return tuple(pack(bits) for bits in (gate > limit, up < -limit, up > limit))
