import importlib


def test_configuration_import_avoids_circular_dependency():
    configuration = importlib.import_module("distillkit.configuration")

    assert configuration.MissingProbabilityHandling.ZERO.value == "zero"


def test_compression_package_exports_are_lazy_and_still_available():
    compression = importlib.import_module("distillkit.compression")
    config = importlib.import_module("distillkit.compression.config")

    assert (
        compression.DistributionQuantizationConfig
        is config.DistributionQuantizationConfig
    )
    assert (
        compression.LegacyLogitCompressionConfig is config.LegacyLogitCompressionConfig
    )
