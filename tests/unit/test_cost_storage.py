from __future__ import annotations

import pytest

from llm_vis.cost import INT4_FORMAT_NAME, dtype_nbytes, weight_storage


@pytest.mark.parametrize("weight_format", ["bfloat16", "float16"])
def test_float_weight_storage_has_no_metadata(weight_format: str) -> None:
    storage = weight_storage(129, weight_format)
    assert storage.packed_weight_bytes == 258
    assert storage.metadata_bytes == 0
    assert storage.padding_values == 0
    assert storage.total_bytes == 258


def test_frozen_int4_format_counts_packing_scale_and_padding() -> None:
    storage = weight_storage(129, "int4")
    assert storage.format_name == INT4_FORMAT_NAME
    assert storage.group_size == 128
    assert storage.padding_values == 127
    assert storage.packed_weight_bytes == 128
    assert storage.scale_metadata_bytes == 4
    assert storage.zero_point_metadata_bytes == 0
    assert storage.total_bytes == 132
    assert storage.total_bytes > (129 + 1) // 2


def test_int4_group_boundaries_exhaustively() -> None:
    for values in range(0, 513):
        storage = weight_storage(values, "int4")
        groups = (values + 127) // 128
        assert storage.packed_weight_bytes == groups * 64
        assert storage.scale_metadata_bytes == groups * 2
        assert storage.total_bytes == groups * 66
        assert storage.padding_values == groups * 128 - values


def test_supported_dtype_sizes_and_invalid_formats() -> None:
    assert dtype_nbytes("bfloat16") == 2
    assert dtype_nbytes("float16") == 2
    assert dtype_nbytes("float32") == 4
    with pytest.raises(ValueError, match="Unsupported dtype"):
        dtype_nbytes("complex64")
    with pytest.raises(ValueError, match="Unsupported weight format"):
        weight_storage(10, "float32")
    with pytest.raises(ValueError, match="positive even"):
        weight_storage(10, "int4", group_size=127)
