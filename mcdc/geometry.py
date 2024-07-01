import math
import numba as nb

import mcdc.physics as physics

from mcdc.algorithm import binary_search
from mcdc.constant import (
    BC_REFLECTIVE,
    BC_VACUUM,
    INF,
    PI,
    REGION_HALFSPACE,
    REGION_INTERSECTION,
    REGION_COMPLEMENT,
    REGION_UNION,
    REGION_ALL,
    SURFACE_COINCIDENT_TOLERANCE,
)


# ======================================================================================
# Particle local coordinate
# ======================================================================================


@nb.njit
def get_local_coordinate(particle):
    # Global coordinate
    x = particle['x']
    y = particle['y']
    z = particle['z']

    ux = particle['ux']
    uy = particle['uy']
    uz = particle['uz']

    # Translate
    if particle['translated']:
        translation = particle['translation']
        x -= translation[0]
        y -= translation[1]
        z -= translation[2]

    if not particle['rotated']:
        return x, y, z, ux, uy, uz

    # Rotation matrix
    rotation = particle['rotation']
    xx, xy, xz, yz, yy, yz, zx, zy, zz = rotation_matrix(rotation)

    # Rotate
    x_local = x * xx + y * xy + z * xz
    y_local = x * yx + y * yy + z * yz
    z_local = x * zx + y * zy + z * zz

    ux_local = ux * xx + uy * xy + uz * xz
    uy_local = ux * yx + uy * yy + uz * yz
    uz_local = ux * zx + uy * zy + uz * zz

    return x_local, y_local, z_local, ux_local, uy_local, uz_local


@nb.njit
def get_local_position(particle):
    # Global coordinate
    x = particle['x']
    y = particle['y']
    z = particle['z']

    # Translate
    if particle['translated']:
        translation = particle['translation']
        x -= translation[0]
        y -= translation[1]
        z -= translation[2]

    if not particle['rotated']:
        return x, y, z

    # Rotation matrix
    rotation = particle['rotation']
    xx, xy, xz, yz, yy, yz, zx, zy, zz = rotation_matrix(rotation)

    # Rotate
    x_local = x * xx + y * xy + z * xz
    y_local = x * yx + y * yy + z * yz
    z_local = x * zx + y * zy + z * zz

    return x_local, y_local, z_local


@nb.njit
def rotation_matrix(rotation):
    phi = rotation[0] * PI / 180.0
    theta = rotation[1] * PI / 180.0
    psi = rotation[2] * PI / 180.0

    xx = math.cos(theta) * math.cos(psi)
    xy = -math.cos(phi) * math.sin(psi) + math.sin(phi) * math.sin(theta) * math.cos(psi)
    xz = math.sin(phi) * math.sin(psi) + math.cos(phi) * math.sin(theta) * math.cos(psi)

    yx = math.cos(theta) * math.sin(psi)
    yy = math.cos(phi) * math.cos(psi) + math.sin(phi) * math.sin(theta) * math.sin(psi)
    yz = -math.sin(phi) * math.cos(psi) + math.cos(phi) * math.sin(theta) * math.sin(psi)

    zx = -math.sin(theta)
    zy = math.sin(phi) * math.cos(theta)
    zz = math.cos(phi) * math.cos(theta)

    return xx, xy, xz, yz, yy, yz, zx, zy, zz


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

        # Local coordinate
        x, y, z, ux, uy, uz = get_local_coordinate(particle)
        t = particle['t']
        result = evaluate_surface(x, y, z, t, surface)

        # Check if coincident
        if abs(result) < SURFACE_COINCIDENT_TOLERANCE:
            nx, ny, nz = get_surface_normal(x, y, z, surface)
            result = nx * ux + ny * uy + nz * uz

        if positive_side:
            if result > 0.0:
                return True
        elif result < 0.0:
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
# TODO: Make moving plane its own type to minimize intrusion

@nb.njit
def evaluate_surface(x, y, z, t, surface):
    """Evaluate the surface equation"""
    # Linear part
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

    # Quadratic part
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
def apply_boundary_condition(particle, surface):
    if surface["BC"] == BC_VACUUM:
        particle["alive"] = False
    elif surface["BC"] == BC_REFLECTIVE:
        # Get coordinate
        x = particle['x']
        y = particle['y']
        z = particle['z']
        ux = particle["ux"]
        uy = particle["uy"]
        uz = particle["uz"]
        # TODO: Consider local coordinate?

        # Get surface normal
        nx, ny, nz = get_surface_normal(x, y, z, surface)

        # Reflect direction
        c = 2.0 * (nx * ux + ny * uy + nz * uz)
        particle["ux"] -= c * nx
        particle["uy"] -= c * ny
        particle["uz"] -= c * nz


@nb.njit
def get_surface_normal(x, y, z, surface):
    if surface["linear"]:
        return surface["nx"], surface["ny"], surface["nz"]

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
def get_surface_distance(particle, surface, mcdc):
    # Get local coordinate
    x, y, z, ux, uy, uz = get_local_coordinate(particle)
    t = particle['t']

    G = surface["G"]
    H = surface["H"]
    I_ = surface["I"]

    surface_move = False
    if surface["linear"]:
        idx = 0
        if surface["N_slice"] > 1:
            idx = binary_search(t, surface["t"][: surface["N_slice"] + 1])
        J1 = surface["J"][idx][1]
        v = physics.get_particle_speed(particle, mcdc)

        t_max = surface["t"][idx + 1]
        d_max = (t_max - t) * v

        div = G * ux + H * uy + I_ * uz + J1 / v
        if div == 0.0:
            return INF, surface_move

        # Evaluate surface equation
        result = evaluate_surface(x, y, z, t, surface)

        # Check if coincident
        if abs(result) < SURFACE_COINCIDENT_TOLERANCE:
            return INF, False
        distance = -result / div

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
    c = evaluate_surface(x, y, z, t, surface)

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
