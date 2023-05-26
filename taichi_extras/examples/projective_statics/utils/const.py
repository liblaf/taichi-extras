import taichi as ti

GRAVITY: ti.Vector = ti.Vector([0.0, 0.0, 0.0])
MASS_DENSITY: float = 1000.0
SHEAR_MODULUS: float = 1.0
TIME_STEP: float = 1.0 / 30.0 / 100


# https://en.wikipedia.org/wiki/Young%27s_modulus
YOUNG_MODULUS: float = 1e4
POISSON_RATIO: float = 0.0
SHEAR_MODULUS: float = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
