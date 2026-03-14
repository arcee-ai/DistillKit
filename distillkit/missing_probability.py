from enum import Enum


class MissingProbabilityHandling(Enum):
    ZERO = "zero"
    SYMMETRIC_UNIFORM = "symmetric_uniform"
