import h5py
import numpy as np
import mcdc

'''
KobayashiG7-TD
- Based on the Kobayashi dog-leg benchmark problem
  [PNE 2001, https://doi.org/10.1016/S0149-1970(01)00007-5]
- Modified by replacing the shield and void materials with water and
  low-density water
- UO2 fuel is added in the second turn of the void channel.
- 7-group library from C5G7-TD benchmark is used
  [NED 2017, http://dx.doi.org/10.1016/j.nucengdes.2017.02.008]
'''


# =============================================================================
# Materials
# =============================================================================

# Load material data
lib = h5py.File("c5g7_xs.h5", "r")

# Material setter
def set_mat(mat, density=1.0):
    return mcdc.material(
        capture=mat["capture"][:]*density,
        scatter=mat["scatter"][:]*density,
        fission=mat["fission"][:]*density,
        nu_p=mat["nu_p"][:],
        nu_d=mat["nu_d"][:],
        chi_p=mat["chi_p"][:],
        chi_d=mat["chi_d"][:],
        speed=mat["speed"][:],
        decay=mat["decay"][:],
    )

# Set materials
mat_shield = set_mat(lib["mod"]) # Water
mat_void = set_mat(lib["mod"], density=1E-3) # Low-density water
mat_fuel = set_mat(lib["uo2"]) # Fuel


# =============================================================================
# Cells
# =============================================================================

# Set surfaces
sx1 = mcdc.surface("plane-x", x=0.0, bc="reflective")
sx2 = mcdc.surface("plane-x", x=10.0)
sx3 = mcdc.surface("plane-x", x=30.0)
sx4 = mcdc.surface("plane-x", x=40.0)
sx5 = mcdc.surface("plane-x", x=60.0, bc="vacuum")
sy1 = mcdc.surface("plane-y", y=0.0, bc="reflective")
sy2 = mcdc.surface("plane-y", y=10.0)
sy3 = mcdc.surface("plane-y", y=50.0)
sy4 = mcdc.surface("plane-y", y=60.0)
sy5 = mcdc.surface("plane-y", y=100.0, bc="vacuum")
sz1 = mcdc.surface("plane-z", z=0.0, bc="reflective")
sz2 = mcdc.surface("plane-z", z=10.0)
sz3 = mcdc.surface("plane-z", z=30.0)
sz4 = mcdc.surface("plane-z", z=40.0)
sz5 = mcdc.surface("plane-z", z=60.0, bc="vacuum")

# Set cells
# Source
mcdc.cell([+sx1, -sx2, +sy1, -sy2, +sz1, -sz2], mat_void)
# Void straight channels
mcdc.cell([+sx1, -sx2, +sy2, -sy3, +sz1, -sz2], mat_void)
mcdc.cell([+sx2, -sx3, +sy3, -sy4, +sz1, -sz2], mat_void)
mcdc.cell([+sx3, -sx4, +sy3, -sy4, +sz2, -sz3], mat_void)
mcdc.cell([+sx3, -sx4, +sy4, -sy5, +sz3, -sz4], mat_void)
# Void channel turns (fuel in the second turn)
mcdc.cell([+sx1, -sx2, +sy3, -sy4, +sz1, -sz2], mat_void)
mcdc.cell([+sx3, -sx4, +sy3, -sy4, +sz1, -sz2], mat_fuel)
mcdc.cell([+sx3, -sx4, +sy3, -sy4, +sz3, -sz4], mat_void)
# Shield surrounding the channel
mcdc.cell([+sx1, -sx3, +sy1, -sy5, +sz2, -sz5], mat_shield)
mcdc.cell([+sx2, -sx5, +sy1, -sy3, +sz1, -sz2], mat_shield)
mcdc.cell([+sx3, -sx5, +sy1, -sy3, +sz2, -sz5], mat_shield)
mcdc.cell([+sx3, -sx5, +sy4, -sy5, +sz1, -sz3], mat_shield)
mcdc.cell([+sx4, -sx5, +sy4, -sy5, +sz3, -sz5], mat_shield)
mcdc.cell([+sx4, -sx5, +sy3, -sy4, +sz1, -sz5], mat_shield)
mcdc.cell([+sx3, -sx4, +sy3, -sy5, +sz4, -sz5], mat_shield)
mcdc.cell([+sx1, -sx3, +sy4, -sy5, +sz1, -sz2], mat_shield)

# =============================================================================
# Set source
# =============================================================================
# Isotropic, highest-energy group, uniform box (10x10x10) source

energy = np.zeros(7)
energy[0] = 1.0
mcdc.source(
    x=[0.0, 10.0], y=[0.0, 10.0], z=[0.0, 10.0], isotropic=True, energy=energy
)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: 5D (x, y, z, t, g)
mcdc.tally(
    scores=["flux"],
    x=np.linspace(0.0, 60.0, 31),
    y=np.linspace(0.0, 100.0, 51),
    z=np.linspace(0.0, 60.0, 31),
    t=np.logspace(-8, 2, 51),
    g=np.array([-0.5, 3.5, 6.5]) # fast (0, 1, 2, 3) and thermal (4, 5, 6) groups
)

# Setting
#   Run in batches (at least 2) to avoid tally bottleneck
#   Need active bank buffer due to significant fission production
mcdc.setting(N_particle=10, N_batch=10, active_bank_buff=10000)

# Run
mcdc.run()
