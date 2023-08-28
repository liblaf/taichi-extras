from enum import StrEnum

from taichi import Vector

TOLERANCE: float = 1e-6

GRAVITY: Vector = Vector([0.0, 0.0, 0.0])
MASS_DENSITY: float = 1000.0
# TIME_STEP: float = 1.0 / 30.0
TIME_STEP: float = 1.0 / 30.0 * 10.0

# https://en.wikipedia.org/wiki/Young%27s_modulus
YOUNG_MODULUS: float = 1e5
POISSON_RATIO: float = 0.0
SHEAR_MODULUS: float = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))

# fixed
# FIXED_STIFFNESS: float = 1e8
FIXED_STIFFNESS: float = 1e1


class Method(StrEnum):
    CG = "CG"  # Conjugate Gradient
    Sparse = "Sparse"
