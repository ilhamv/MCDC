import math
import numba as nb

from mcdc.constant import SQRT_E_TO_SPEED


@nb.njit
def get_particle_speed(particle, mcdc):
    if mcdc["setting"]["mode_MG"]:
        material_ID = particle['material_ID']
        g = particle["g"]
        return mcdc["materials"][material_ID]["speed"][g]
    else:
        E = particle["E"]
        return math.sqrt(E) * SQRT_E_TO_SPEED


