from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from distillkit.compression.bitpack import pack_to_bytes, unpack_from_bytes
    from distillkit.compression.compressor import LogprobCompressor
    from distillkit.compression.config import (
        DistributionQuantizationConfig,
        LegacyLogitCompressionConfig,
        QuantizationBin,
    )
    from distillkit.compression.densify import densify
    from distillkit.compression.legacy import (
        LogitCompressor as LegacyLogitCompressor,
    )

_EXPORTS = {
    "pack_to_bytes": ("distillkit.compression.bitpack", "pack_to_bytes"),
    "unpack_from_bytes": ("distillkit.compression.bitpack", "unpack_from_bytes"),
    "QuantizationBin": ("distillkit.compression.config", "QuantizationBin"),
    "DistributionQuantizationConfig": (
        "distillkit.compression.config",
        "DistributionQuantizationConfig",
    ),
    "LogprobCompressor": ("distillkit.compression.compressor", "LogprobCompressor"),
    "densify": ("distillkit.compression.densify", "densify"),
    "LegacyLogitCompressor": (
        "distillkit.compression.legacy",
        "LogitCompressor",
    ),
    "LegacyLogitCompressionConfig": (
        "distillkit.compression.config",
        "LegacyLogitCompressionConfig",
    ),
}

__all__ = [
    "pack_to_bytes",
    "unpack_from_bytes",
    "QuantizationBin",
    "DistributionQuantizationConfig",
    "LogprobCompressor",
    "densify",
    "LegacyLogitCompressor",
    "LegacyLogitCompressionConfig",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
