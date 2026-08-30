"""Physical storage accounting for supported weight and activation formats."""

from __future__ import annotations

from dataclasses import dataclass

from llm_vis.ir.models import DType, WeightFormat

INT4_GROUP_SIZE = 128
INT4_SCALE_BYTES = 2
INT4_FORMAT_NAME = "int4-groupwise-symmetric-g128-fp16-scale-v1"


@dataclass(frozen=True)
class WeightStorage:
    """Storage split into packed payload, metadata, and explicit padding."""

    format: WeightFormat
    logical_values: int
    packed_weight_bytes: int
    scale_metadata_bytes: int
    zero_point_metadata_bytes: int
    padding_values: int
    group_size: int | None
    format_name: str

    @property
    def metadata_bytes(self) -> int:
        return self.scale_metadata_bytes + self.zero_point_metadata_bytes

    @property
    def total_bytes(self) -> int:
        return self.packed_weight_bytes + self.metadata_bytes


def dtype_nbytes(dtype: DType | str) -> int:
    """Return supported scalar storage bytes."""

    normalized = DType(dtype)
    sizes = {
        DType.FLOAT16: 2,
        DType.BFLOAT16: 2,
        DType.FLOAT32: 4,
        DType.INT8: 1,
        DType.UINT8: 1,
    }
    try:
        return sizes[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype for cost accounting: {normalized.value}") from exc


def weight_storage(
    logical_values: int,
    weight_format: WeightFormat | str,
    *,
    group_size: int = INT4_GROUP_SIZE,
    scale_bytes: int = INT4_SCALE_BYTES,
) -> WeightStorage:
    """Return physical bytes for float weights or the frozen groupwise INT4 format.

    INT4 groups are zero-padded to ``group_size`` values. Each group stores packed
    nibbles plus one FP16 scale and no zero point (symmetric quantization).
    """

    if isinstance(logical_values, bool) or not isinstance(logical_values, int):
        raise TypeError("logical_values must be an integer")
    if logical_values < 0:
        raise ValueError("logical_values must be non-negative")
    normalized = WeightFormat(weight_format)
    if normalized in {WeightFormat.BFLOAT16, WeightFormat.FLOAT16}:
        bytes_per_value = 2
        return WeightStorage(
            format=normalized,
            logical_values=logical_values,
            packed_weight_bytes=logical_values * bytes_per_value,
            scale_metadata_bytes=0,
            zero_point_metadata_bytes=0,
            padding_values=0,
            group_size=None,
            format_name=normalized.value,
        )
    if normalized != WeightFormat.INT4:
        raise ValueError(f"Unsupported weight format: {normalized.value}")
    if group_size <= 0 or group_size % 2:
        raise ValueError("INT4 group_size must be a positive even integer")
    if scale_bytes <= 0:
        raise ValueError("scale_bytes must be positive")

    groups = (logical_values + group_size - 1) // group_size
    padded_values = groups * group_size
    return WeightStorage(
        format=normalized,
        logical_values=logical_values,
        packed_weight_bytes=padded_values // 2,
        scale_metadata_bytes=groups * scale_bytes,
        zero_point_metadata_bytes=0,
        padding_values=padded_values - logical_values,
        group_size=group_size,
        format_name=(
            INT4_FORMAT_NAME
            if group_size == INT4_GROUP_SIZE and scale_bytes == INT4_SCALE_BYTES
            else f"int4-groupwise-symmetric-g{group_size}-scale{scale_bytes}-v1"
        ),
    )
