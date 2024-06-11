import h5py
import numpy as np

import mcdc

# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")


# Materials
def set_mat(mat):
    return mcdc.material(
        capture=mat["capture"][:],
        scatter=mat["scatter"][:],
        fission=mat["fission"][:],
        nu_p=mat["nu_p"][:],
        nu_d=mat["nu_d"][:],
        chi_p=mat["chi_p"][:],
        chi_d=mat["chi_d"][:],
        speed=mat["speed"],
        decay=mat["decay"],
    )


fuel = set_mat(lib["uo2"])
casing = set_mat(lib["gt"])
water = set_mat(lib["mod"])

# =============================================================================
# Shooting star
# =============================================================================

# Surfaces
cy_z = mcdc.surface("cylinder-z", center=[0.0, 0.0], radius=1.0)
cy_x = mcdc.surface("cylinder-x", center=[0.0, 0.0], radius=1.0)

pz_0 = mcdc.surface("plane-z", z=-2.5)
pz_1 = mcdc.surface("plane-z", z=2.5)

px_0 = mcdc.surface("plane-x", x=-2.5)
px_1 = mcdc.surface("plane-x", x=2.5)

cylinder1 = -cy_z & +pz_0 & -pz_1
cylinder2 = -cy_x & +px_0 & -px_1
shooting_star = cylinder1 | cylinder2

fuel_cell = mcdc.cell(shooting_star, fuel)

# =============================================================================
# Casing and water outside
# =============================================================================

sp = mcdc.surface("sphere", center = [0.0, 0.0, 0.0], radius=3.0)

casing_cell = mcdc.cell(-sp & ~shooting_star, casing)
water_cell = mcdc.cell(+sp, water)

# =============================================================================
# The universe
# =============================================================================

univ = mcdc.universe([fuel_cell, casing_cell, water_cell])

# =============================================================================
# The bathub
# =============================================================================

# Create the bathtub surfaces
plane_left = mcdc.surface('plane-x', x=-10.0, bc='reflective')
plane_right = mcdc.surface('plane-x', x=10.0, bc='reflective')
plane_bottom = mcdc.surface('plane-z', z=-5.0, bc='reflective')
plane_top = mcdc.surface('plane-z', z=5.0, bc='reflective')
plane_front = mcdc.surface('plane-y', y=-5.0, bc='reflective')
plane_back = mcdc.surface('plane-y', y=5.0, bc='reflective')

# Create the bathtub region
bathtub = +plane_left & -plane_right & +plane_bottom & -plane_top & +plane_front & -plane_back

# Make a divider that splits the bathtub into two container cells
divider = mcdc.surface('plane-x', x=0.0)

# Container cells
left_cell = mcdc.cell(bathtub & -divider, univ, translation=[-5.0, 0.0, 0.0])
right_cell = mcdc.cell(bathtub & +divider, univ, translation=[5.0, 0.0, 0.0])#, rotation=[0.0, 45.0, 0.0])

# Root universe
mcdc.universe([left_cell, right_cell], root=True)

# =============================================================================
# Set source
# =============================================================================
# Uniform in energy

source = mcdc.source(x=[-10.0, 10.0], y=[-5.0, 5.0], z=[-5.0, 5.0])

# =============================================================================
# Set tally and parameter, and then run mcdc
# =============================================================================

# Tally
mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(-10.0, 10.0, 101),
    z=np.linspace(-5.0, 5.0, 51),
    g="all",
)

# Setting
mcdc.setting(N_particle=1000, census_bank_buff=3.0)
mcdc.eigenmode(N_inactive=2, N_active=3, gyration_radius="all")
mcdc.population_control()

# Run
colors = {
    fuel: "red",
    casing: "gray",
    water: "blue"
}
mcdc.visualize('xz', x=[-10.0, 10.0], z=[-5.0, 5.0], pixels=(200, 100), colors=colors)
#mcdc.run()
