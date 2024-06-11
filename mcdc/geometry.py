import math
import numba as nb

import mcdc.physics as physics

from mcdc.algorithm import binary_search
from mcdc.constant import (
    BC_REFLECTIVE,
    BC_VACUUM,
    INF,
    REGION_HALFSPACE,
    REGION_INTERSECTION,
    REGION_COMPLEMENT,
    REGION_UNION,
    REGION_ALL,
    SHIFT,
)


# ======================================================================================
# Particle local coordinate
# ======================================================================================


@nb.njit
def reset_local_coordinate(particle):
    """
    Reset local coordinate so that it is equivalent to the global coordinate
    """
    particle["x_local"] = particle["x"]
    particle["y_local"] = particle["y"]
    particle["z_local"] = particle["z"]

    particle["ux_local"] = particle["ux"]
    particle["uy_local"] = particle["uy"]
    particle["uz_local"] = particle["uz"]


@nb.njit
def translate_local_coordinate(particle, translation):
    """
    Translate local coordinate wrt the given translation
    """
    particle["x_local"] -= translation[0]
    particle["y_local"] -= translation[1]
    particle["z_local"] -= translation[2]


@nb.njit
def rotate_local_coordinate(particle, rotation):
    """
    Rotate both local coordinate and direction wrt the given rotation
    """
    x = particle["x_local"]
    y = particle["y_local"]
    z = particle["z_local"]

    particle["x_local"] = x * rotation[0] + y * rotation[1] + z * rotation[2]
    particle["y_local"] = x * rotation[3] + y * rotation[4] + z * rotation[5]
    particle["z_local"] = x * rotation[6] + y * rotation[7] + z * rotation[8]

    ux = particle["ux_local"]
    uy = particle["uy_local"]
    uz = particle["uz_local"]

    particle["ux_local"] = ux * rotation[0] + uy * rotation[1] + uz * rotation[2]
    particle["uy_local"] = ux * rotation[3] + uy * rotation[4] + uz * rotation[5]
    particle["uz_local"] = ux * rotation[6] + uy * rotation[7] + uz * rotation[8]


# ======================================================================================
# Particle locator
# ======================================================================================


@nb.njit
def get_cell(particle, universe_ID, mcdc):
    """
    Find and return particle cell ID in the given universe
    """
    universe = mcdc["universes"][universe_ID]
    N_cell = universe['N_cell']
    for i in range(N_cell):
        cell_ID = universe["cell_IDs"][i]
        cell = mcdc["cells"][cell_ID]
        if check_cell(particle, cell, mcdc):
            return cell["ID"]

    # Particle is not found
    print("A particle is lost at (", particle["x"], particle["y"], particle["z"], ")")

    particle["alive"] = False
    return -1


@nb.njit
def check_cell(particle, cell, mcdc):
    """
    Check if a particle is in the given cell
    """
    region = mcdc["regions"][cell["region_ID"]]
    return check_region(particle, region, mcdc)


@nb.njit
def check_region(particle, region, mcdc):
    """
    Check if a particle is in the given region
    """
    if region["type"] == REGION_HALFSPACE:
        """
        In halfspace region
            A: ID of the reference surface, and
            B: Toggle indicating if the region is in the positive side of the surface.
        """
        surface_ID = region["A"]
        positive_side = region["B"]

        surface = mcdc["surfaces"][surface_ID]
        side = surface_evaluate(particle, surface)

        if positive_side:
            if side > 0.0:
                return True
        elif side < 0.0:
            return True

        return False

    elif region["type"] == REGION_INTERSECTION:
        """
        In intersection region, A and B are the IDs of the intersecting regions.
        """
        region_A = mcdc["regions"][region["A"]]
        region_B = mcdc["regions"][region["B"]]

        check_A = check_region(particle, region_A, mcdc)
        check_B = check_region(particle, region_B, mcdc)

        if check_A and check_B:
            return True
        else:
            return False

    elif region["type"] == REGION_COMPLEMENT:
        """
        In complement region, A is the ID of the complemented region
        """
        region_A = mcdc["regions"][region["A"]]
        if not check_region(particle, region_A, mcdc):
            return True
        else:
            return False

    elif region["type"] == REGION_UNION:
        """
        In union region, A and B are the IDs of the unionized regions.
        """
        region_A = mcdc["regions"][region["A"]]
        region_B = mcdc["regions"][region["B"]]

        if check_region(particle, region_A, mcdc):
            return True

        if check_region(particle, region_B, mcdc):
            return True

        return False

    elif region["type"] == REGION_ALL:
        return True


# =============================================================================
# Surface operations
# =============================================================================
# Quadric surface: Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J(t) = 0
#   J(t) = J0_i + J1_i*t for t in [t_{i-1}, t_i), t_0 = 0

# TODO: Rename and rorganize into different types

@nb.njit
def surface_evaluate(P, surface):
    x = P["x_local"]
    y = P["y_local"]
    z = P["z_local"]
    t = P["t"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    # Get time indices
    idx = 0
    if surface["N_slice"] > 1:
        idx = binary_search(t, surface["t"][: surface["N_slice"] + 1])

    # Get constant
    J0 = surface["J"][idx][0]
    J1 = surface["J"][idx][1]
    J = J0 + J1 * (t - surface["t"][idx])

    result = G * x + H * y + I_ * z + J

    if surface["linear"]:
        return result

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    return (
        result + A * x * x + B * y * y + C * z * z + D * x * y + E * x * z + F * y * z
    )


@nb.njit
def surface_bc(P, surface):
    if surface["BC"] == BC_VACUUM:
        P["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        surface_reflect(P, surface)


@nb.njit
def surface_reflect(P, surface):
    # TODO: consider rotated universe
    ux = P["ux_local"]
    uy = P["uy_local"]
    uz = P["uz_local"]
    nx, ny, nz = surface_normal(P, surface)
    # 2.0*surface_normal_component(...)
    c = 2.0 * (nx * ux + ny * uy + nz * uz)

    P["ux_local"] = ux - c * nx
    P["uy_local"] = uy - c * ny
    P["uz_local"] = uz - c * nz

    # Also update global coordinate
    P["ux"] -= c * nx
    P["uy"] -= c * ny
    P["uz"] -= c * nz


@nb.njit
def surface_shift(P, surface, mcdc):
    ux = P["ux_local"]
    uy = P["uy_local"]
    uz = P["uz_local"]

    # Get surface normal
    nx, ny, nz = surface_normal(P, surface)

    # The shift
    shift_x = nx * SHIFT
    shift_y = ny * SHIFT
    shift_z = nz * SHIFT

    # Get dot product to determine shift sign
    if surface["linear"]:
        # Get time indices
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(P["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = physics.get_particle_speed(P, mcdc)
        dot = ux * nx + uy * ny + uz * nz + J1 / v
    else:
        dot = ux * nx + uy * ny + uz * nz

    if dot > 0.0:
        P["x"] += shift_x
        P["y"] += shift_y
        P["z"] += shift_z
        P["x_local"] += shift_x
        P["y_local"] += shift_y
        P["z_local"] += shift_z
    else:
        P["x"] -= shift_x
        P["y"] -= shift_y
        P["z"] -= shift_z
        P["x_local"] -= shift_x
        P["y_local"] -= shift_y
        P["z_local"] -= shift_z


@nb.njit
def surface_normal(P, surface):
    if surface["linear"]:
        return surface["nx"], surface["ny"], surface["nz"]

    x = P["x_local"]
    y = P["y_local"]
    z = P["z_local"]

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]
    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    dx = 2 * A * x + D * y + E * z + G
    dy = 2 * B * y + D * x + F * z + H
    dz = 2 * C * z + E * x + F * y + I_

    norm = (dx**2 + dy**2 + dz**2) ** 0.5
    return dx / norm, dy / norm, dz / norm


@nb.njit
def surface_normal_component(P, surface):
    ux = P["ux_local"]
    uy = P["uy_local"]
    uz = P["uz_local"]
    nx, ny, nz = surface_normal(P, surface)
    return nx * ux + ny * uy + nz * uz


@nb.njit
def surface_distance(P, surface, mcdc):
    ux = P["ux_local"]
    uy = P["uy_local"]
    uz = P["uz_local"]

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    surface_move = False
    if surface["linear"]:
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(P["t"], surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = physics.get_particle_speed(P, mcdc)

        t_max = surface["t"][idx + 1]
        d_max = (t_max - P["t"]) * v

        div = G * ux + H * uy + I_ * uz + J1 / v
        if div == 0.0:
            return INF, surface_move

        distance = -surface_evaluate(P, surface) / div

        # Go beyond current movement slice?
        if distance > d_max:
            distance = d_max
            surface_move = True
        elif distance < 0 and idx < surface["N_slice"] - 1:
            distance = d_max
            surface_move = True

        # Moving away from the surface
        if distance < 0.0:
            return INF, surface_move
        else:
            return distance, surface_move

    x = P["x_local"]
    y = P["y_local"]
    z = P["z_local"]

    A = surface["A"]
    B = surface["B"]
    C = surface["C"]
    D = surface["D"]
    E = surface["E"]
    F = surface["F"]

    # Quadratic equation constants
    a = (
        A * ux * ux
        + B * uy * uy
        + C * uz * uz
        + D * ux * uy
        + E * ux * uz
        + F * uy * uz
    )
    b = (
        2 * (A * x * ux + B * y * uy + C * z * uz)
        + D * (x * uy + y * ux)
        + E * (x * uz + z * ux)
        + F * (y * uz + z * uy)
        + G * ux
        + H * uy
        + I_ * uz
    )
    c = surface_evaluate(P, surface)

    determinant = b * b - 4.0 * a * c

    # Roots are complex  : no intersection
    # Roots are identical: tangent
    # ==> return huge number
    if determinant <= 0.0:
        return INF, surface_move
    else:
        # Get the roots
        denom = 2.0 * a
        sqrt = math.sqrt(determinant)
        root_1 = (-b + sqrt) / denom
        root_2 = (-b - sqrt) / denom

        # Negative roots, moving away from the surface
        if root_1 < 0.0:
            root_1 = INF
        if root_2 < 0.0:
            root_2 = INF

        # Return the smaller root
        return min(root_1, root_2), surface_move
