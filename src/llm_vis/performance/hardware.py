# ruff: noqa: UP045
"""User-supplied hardware profiles for theoretical roofline lower bounds."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from llm_vis.ir import DType

PositiveNumber = Annotated[Union[StrictInt, StrictFloat], Field(gt=0)]


class HardwareProfileBase(BaseModel):
    """Strict base model shared by the hardware profile wire objects."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class HardwareProvenanceKind(str, Enum):
    """How the user obtained the stated peak values."""

    VENDOR_SPEC = "vendor_spec"
    MEASURED = "measured"
    USER_SUPPLIED = "user_supplied"
    SYNTHETIC = "synthetic"


class HardwareProvenance(HardwareProfileBase):
    """Evidence accompanying a user-provided hardware profile."""

    kind: HardwareProvenanceKind
    source: str = Field(min_length=1)
    uri: Optional[str] = Field(default=None, min_length=1)
    notes: List[str] = Field(default_factory=list)


class HardwareProfile(HardwareProfileBase):
    """Dtype-specific theoretical peaks supplied by the caller.

    LLM-Vis deliberately ships no default device profile.  Optional peak fields
    preserve an explicitly unknown capability instead of substituting a guessed
    vendor or device value.
    """

    schema_version: Literal["0.1"] = "0.1"
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    peak_flops_per_second: Optional[PositiveNumber] = None
    memory_bandwidth_bytes_per_second: Optional[PositiveNumber] = None
    dtype: DType
    provenance: HardwareProvenance

    @field_validator("dtype")
    @classmethod
    def require_concrete_dtype(cls, value: DType) -> DType:
        if value == DType.UNKNOWN:
            raise ValueError("hardware profile dtype must be concrete")
        return value


def hardware_profile_schema() -> Dict[str, Any]:
    """Return a standalone Draft 2020-12 JSON Schema for ``HardwareProfile``."""

    schema = HardwareProfile.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://llm-vis.local/schemas/hardware-profile-0.1.json"
    return schema
