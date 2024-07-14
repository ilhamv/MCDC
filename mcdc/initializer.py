import os, math

import numpy as np

from mpi4py import MPI

import mcdc.type_ as type_

from mcdc.constant import (
    BC_NONE,
    BC_REFLECTIVE,
    BC_VACUUM,
    FILL_LATTICE,
    FILL_MATERIAL,
    FILL_UNIVERSE,
    REGION_COMPLEMENT,
    REGION_HALFSPACE,
    REGION_INTERSECTION,
    REGION_UNION,
)


# ======================================================================================
# Simulation objects
# ======================================================================================


def set_nuclides(mcdc, input_deck):
    N_nuclide = len(input_deck.nuclides)
    mode_CE = input_deck.setting["mode_CE"]
    mode_MG = input_deck.setting["mode_MG"]

    for i in range(N_nuclide):
        # General data
        for name in ["ID", "fissionable", "sensitivity", "sensitivity_ID", "dsm_Np"]:
            copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # MG data
        if mode_MG:
            for name in [
                "G",
                "J",
                "speed",
                "decay",
                "total",
                "capture",
                "scatter",
                "fission",
                "nu_s",
                "nu_f",
                "nu_p",
                "nu_d",
                "chi_s",
                "chi_p",
                "chi_d",
            ]:
                copy_field(mcdc["nuclides"][i], input_deck.nuclides[i], name)

        # CE data (load data from XS library)
        dir_name = os.getenv("MCDC_XSLIB")
        if mode_CE:
            nuc_name = input_deck.nuclides[i]["name"]
            with h5py.File(dir_name + "/" + nuc_name + ".h5", "r") as f:
                # Atomic weight ratio
                mcdc["nuclides"][i]["A"] = f["A"][()]
                # Energy grids
                for name in [
                    "E_xs",
                    "E_nu_p",
                    "E_nu_d",
                    "E_chi_p",
                    "E_chi_d1",
                    "E_chi_d2",
                    "E_chi_d3",
                    "E_chi_d4",
                    "E_chi_d5",
                    "E_chi_d6",
                ]:
                    mcdc["nuclides"][i]["N" + name] = len(f[name][:])
                    mcdc["nuclides"][i][name][: len(f[name][:])] = f[name][:]

                # XS
                for name in ["capture", "scatter", "fission"]:
                    mcdc["nuclides"][i]["ce_" + name][: len(f[name][:])] = f[name][:]
                    mcdc["nuclides"][i]["ce_total"][: len(f[name][:])] += f[name][:]

                # Fission production
                mcdc["nuclides"][i]["ce_nu_p"][: len(f["nu_p"][:])] = f["nu_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_nu_d"][j][: len(f["nu_d"][j, :])] = f[
                        "nu_d"
                    ][j, :]

                # Fission spectrum
                mcdc["nuclides"][i]["ce_chi_p"][: len(f["chi_p"][:])] = f["chi_p"][:]
                for j in range(6):
                    mcdc["nuclides"][i]["ce_chi_d%i" % (j + 1)][
                        : len(f["chi_d%i" % (j + 1)][:])
                    ] = f["chi_d%i" % (j + 1)][:]

                # Decay
                mcdc["nuclides"][i]["ce_decay"][: len(f["decay_rate"][:])] = f[
                    "decay_rate"
                ][:]


def set_materials(mcdc, input_deck):
    N_material = len(mcdc['materials'])

    for i in range(N_material):
        for name in type_.material.names:
            if name in ["nuclide_IDs", "nuclide_densities"]:
                mcdc["materials"][i][name][: mcdc["materials"][i]["N_nuclide"]] = (
                    getattr(input_deck.materials[i], name)
                )
            else:
                copy_field(mcdc["materials"][i], input_deck.materials[i], name)


def set_surfaces(mcdc, input_deck):
    N_surface = len(input_deck.surfaces)
    for i in range(N_surface):
        for name in type_.surface.names:
            if name not in ["J", "t", "BC"]:
                copy_field(mcdc["surfaces"][i], input_deck.surfaces[i], name)

        # Boundary condition
        if input_deck.surfaces[i].boundary_type == "interface":
            mcdc["surfaces"][i]["BC"] = BC_NONE
        elif input_deck.surfaces[i].boundary_type == "vacuum":
            mcdc["surfaces"][i]["BC"] = BC_VACUUM
        elif input_deck.surfaces[i].boundary_type == "reflective":
            mcdc["surfaces"][i]["BC"] = BC_REFLECTIVE

        # Variables with possible different sizes
        for name in ["J", "t"]:
            N = len(getattr(input_deck.surfaces[i], name))
            mcdc["surfaces"][i][name][:N] = getattr(input_deck.surfaces[i], name)


def set_regions(mcdc, input_deck):
    N_region = len(input_deck.regions)
    for i in range(N_region):
        for name in type_.region.names:
            if name not in ["type"]:
                copy_field(mcdc["regions"][i], input_deck.regions[i], name)

        # Type
        if input_deck.regions[i].type == "halfspace":
            mcdc["regions"][i]["type"] = REGION_HALFSPACE
        elif input_deck.regions[i].type == "intersection":
            mcdc["regions"][i]["type"] = REGION_INTERSECTION
        elif input_deck.regions[i].type == "union":
            mcdc["regions"][i]["type"] = REGION_UNION
        elif input_deck.regions[i].type == "complement":
            mcdc["regions"][i]["type"] = REGION_COMPLEMENT
        elif input_deck.regions[i].type == "all":
            mcdc["regions"][i]["type"] = REGION_ALL


def set_cells(mcdc, input_deck):
    N_cell = len(mcdc['cells'])

    for i in range(N_cell):
        # Simple copies
        for name in type_.cell.names:
            if name not in ["fill_type", "surface_IDs"]:
                copy_field(mcdc["cells"][i], input_deck.cells[i], name)

        # Fill type
        if input_deck.cells[i].fill_type == "material":
            mcdc["cells"][i]["fill_type"] = FILL_MATERIAL
        elif input_deck.cells[i].fill_type == "universe":
            mcdc["cells"][i]["fill_type"] = FILL_UNIVERSE
        elif input_deck.cells[i].fill_type == "lattice":
            mcdc["cells"][i]["fill_type"] = FILL_LATTICE

        # Fields with different sizes
        for name in ["surface_IDs"]:
            N = mcdc["cells"][i]["N_surface"]
            mcdc["cells"][i][name][:N] = getattr(input_deck.cells[i], name)


def set_universes(mcdc, input_deck):
    N_universe = len(input_deck.universes)
    for i in range(N_universe):
        for name in type_.universe.names:
            if name not in ["cell_IDs"]:
                mcdc["universes"][i][name] = getattr(input_deck.universes[i], name)

        # Variables with possible different sizes
        for name in ["cell_IDs"]:
            N = mcdc["universes"][i]["N_cell"]
            mcdc["universes"][i][name][:N] = getattr(input_deck.universes[i], name)


def set_lattices(mcdc, input_deck):
    N_lattice = len(input_deck.lattices)
    for i in range(N_lattice):
        # Mesh
        for name in type_.mesh_uniform.names:
            mcdc["lattices"][i]["mesh"][name] = input_deck.lattices[i].mesh[name]

        # Universe IDs
        Nx = mcdc["lattices"][i]["mesh"]["Nx"]
        Ny = mcdc["lattices"][i]["mesh"]["Ny"]
        Nz = mcdc["lattices"][i]["mesh"]["Nz"]
        mcdc["lattices"][i]["universe_IDs"][:Nx, :Ny, :Nz] = input_deck.lattices[
            i
        ].universe_IDs


def set_sources(mcdc, input_deck):
    N_source = len(input_deck.sources)
    for i in range(N_source):
        for name in type_.source.names:
            copy_field(mcdc["sources"][i], input_deck.sources[i], name)

    # Normalize source probabilities
    tot = 1e-16
    for S in mcdc["sources"]:
        tot += S["prob"]
    for S in mcdc["sources"]:
        S["prob"] /= tot


def set_tally(mcdc, input_deck):
    for name in type_.tally.names:
        if name not in ["score", "mesh"]:
            copy_field(mcdc["tally"], input_deck.tally, name)
    # Set mesh
    for name in type_.mesh_names:
        copy_field(mcdc["tally"]["mesh"], input_deck.tally["mesh"], name)


# ======================================================================================
# Simulation settings
# ======================================================================================


def set_setting(mcdc, input_deck):
    for name in type_.setting.names:
        copy_field(mcdc["setting"], input_deck.setting, name)

    # Check if time boundary is above the final tally mesh time grid
    if mcdc["setting"]["time_boundary"] > mcdc["tally"]["mesh"]["t"][-1]:
        mcdc["setting"]["time_boundary"] = mcdc["tally"]["mesh"]["t"][-1]

    if input_deck.technique["iQMC"]:
        if len(mcdc["technique"]["iqmc"]["mesh"]["t"]) - 1 > 1:
            if (
                mcdc["setting"]["time_boundary"]
                > input_deck.technique["iqmc"]["mesh"]["t"][-1]
            ):
                mcdc["setting"]["time_boundary"] = input_deck.technique["iqmc"]["mesh"][
                    "t"
                ][-1]


def set_particle_banks(mcdc, input_deck):
    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"

    # IC generator banks
    if mcdc["technique"]["IC_generator"]:
        mcdc["technique"]["IC_bank_neutron_local"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor_local"]["tag"] = "precursor"
        mcdc["technique"]["IC_bank_neutron"]["tag"] = "neutron"
        mcdc["technique"]["IC_bank_precursor"]["tag"] = "precursor"


def set_eigenvalue(mcdc, input_deck):
    # Initial guess
    mcdc["k_eff"] = mcdc["setting"]["k_init"]

    # Activate tally scoring for fixed-source
    if not mcdc["setting"]["mode_eigenvalue"]:
        mcdc["cycle_active"] = True

    # All active eigenvalue cycle?
    elif mcdc["setting"]["N_inactive"] == 0:
        mcdc["cycle_active"] = True


def set_mpi(mcdc, input_deck):
    # MPI parameters
    mcdc["mpi_size"] = MPI.COMM_WORLD.Get_size()
    mcdc["mpi_rank"] = MPI.COMM_WORLD.Get_rank()
    mcdc["mpi_master"] = mcdc["mpi_rank"] == 0

    # Distribute work to MPI ranks
    if mcdc["technique"]["domain_decomposition"]:
        distribute_work_dd(mcdc["setting"]["N_particle"], mcdc)
    else:
        distribute_work(mcdc["setting"]["N_particle"], mcdc)


# ======================================================================================
# Simulation settings - Occasionals
# ======================================================================================


def set_domain_decomposition(mcdc, input_deck):
    # Set domain mesh
    if input_deck.technique["domain_decomposition"]:
        for name in ["x", "y", "z", "t", "mu", "azi"]:
            copy_field(
                mcdc["technique"]["dd_mesh"], input_deck.technique["dd_mesh"], name
            )
        # Set exchange rate
        for name in ["dd_exchange_rate", "dd_repro"]:
            copy_field(mcdc["technique"], input_deck.technique, name)
        # Set domain index
        copy_field(mcdc, input_deck.technique, "dd_idx")
        for name in ["xp", "xn", "yp", "yn", "zp", "zn"]:
            copy_field(mcdc["technique"], input_deck.technique, f"dd_{name}_neigh")
        copy_field(mcdc["technique"], input_deck.technique, "dd_work_ratio")





def set_uncertainty_quantification(mcdc, input_deck):
    if mcdc["technique"]["uq"]:
        # Assumes that all tallies will also be uq tallies
        for name in type_.uq_tally.names:
            if name != "score":
                copy_field(mcdc["technique"]["uq_tally"], input_deck.tally, name)

        M = len(input_deck.uq_deltas["materials"])
        for i in range(M):
            idm = input_deck.uq_deltas["materials"][i].ID
            mcdc["technique"]["uq_"]["materials"][i]["info"]["ID"] = idm
            mcdc["technique"]["uq_"]["materials"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["materials"][i].distribution
            )
            for name in input_deck.uq_deltas["materials"][i].flags:
                mcdc["technique"]["uq_"]["materials"][i]["flags"][name] = True
                mcdc["technique"]["uq_"]["materials"][i]["delta"][name] = getattr(
                    input_deck.uq_deltas["materials"][i], name
                )
            flags = mcdc["technique"]["uq_"]["materials"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
                flags["speed"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True
            if mcdc["materials"][idm]["N_nuclide"] > 1:
                for name in type_.uq_mat.names:
                    mcdc["technique"]["uq_"]["materials"][i]["mean"][name] = (
                        input_deck.materials[idm][name]
                    )

        N = len(input_deck.uq_deltas["nuclides"])
        for i in range(N):
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["distribution"] = (
                input_deck.uq_deltas["nuclides"][i].distribution
            )
            idn = input_deck.uq_deltas["nuclides"][i].ID
            mcdc["technique"]["uq_"]["nuclides"][i]["info"]["ID"] = idn
            for name in type_.uq_nuc.names:
                if name == "scatter":
                    G = input_deck.nuclides[idn].G
                    chi_s = input_deck.nuclides[idn].chi_s
                    scatter = input_deck.nuclides[idn].scatter
                    scatter_matrix = np.zeros((G, G))
                    for g in range(G):
                        scatter_matrix[g, :] = chi_s[g, :] * scatter[g]

                    mcdc["technique"]["uq_"]["nuclides"][i]["mean"][
                        name
                    ] = scatter_matrix
                else:
                    copy_field(
                        mcdc["technique"]["uq_"]["nuclides"][i]["mean"],
                        input_deck.nuclides[idn],
                        name,
                    )

            for name in input_deck.uq_deltas["nuclides"][i].flags:
                if "padding" in name:
                    continue
                mcdc["technique"]["uq_"]["nuclides"][i]["flags"][name] = True
                copy_field(
                    mcdc["technique"]["uq_"]["nuclides"][i]["delta"],
                    input_deck.uq_deltas["nuclides"][i],
                    name,
                )
            flags = mcdc["technique"]["uq_"]["nuclides"][i]["flags"]
            if flags["capture"] or flags["scatter"] or flags["fission"]:
                flags["total"] = True
            if flags["nu_p"] or flags["nu_d"]:
                flags["nu_f"] = True



def set_sensitivity_quantification(mcdc, input_deck):
    # Threshold
    mcdc["technique"]["dsm_order"] = input_deck.technique["dsm_order"]


def set_source_files(mcdc, input_deck):
    if mcdc["setting"]["source_file"]:
        with h5py.File(mcdc["setting"]["source_file_name"], "r") as f:
            # Get source particle size
            N_particle = f["particles_size"][()]

            # Redistribute work
            distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["particles"][start:end]
            mcdc["bank_source"]["size"] = N_local

    if mcdc["setting"]["IC_file"]:
        with h5py.File(mcdc["setting"]["IC_file_name"], "r") as f:
            # =================================================================
            # Set neutron source
            # =================================================================

            # Get source particle size
            N_particle = f["IC/neutrons_size"][()]

            # Redistribute work
            distribute_work(N_particle, mcdc)
            N_local = mcdc["mpi_work_size"]
            start = mcdc["mpi_work_start"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_source"]["particles"][:N_local] = f["IC/neutrons"][start:end]
            mcdc["bank_source"]["size"] = N_local

            # =================================================================
            # Set precursor source
            # =================================================================

            # Get source particle size
            N_precursor = f["IC/precursors_size"][()]

            # Redistribute work
            distribute_work(N_precursor, mcdc, True)  # precursor = True
            N_local = mcdc["mpi_work_size_precursor"]
            start = mcdc["mpi_work_start_precursor"]
            end = start + N_local

            # Add particles to source bank
            mcdc["bank_precursor"]["precursors"][:N_local] = f["IC/precursors"][
                start:end
            ]
            mcdc["bank_precursor"]["size"] = N_local

            # Set precursor strength
            if N_precursor > 0 and N_particle > 0:
                mcdc["precursor_strength"] = mcdc["bank_precursor"]["precursors"][0][
                    "w"
                ]


def set_IC_generator(mcdc, input_deck):
    for name in [
        "IC_N_neutron",
        "IC_N_precursor",
        "IC_neutron_density",
        "IC_neutron_density_max",
        "IC_precursor_density",
        "IC_precursor_density_max",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)


# ======================================================================================
# Techniques
# ======================================================================================


def set_technique_flags(mcdc, input_deck):
    for name in [
        "weighted_emission",
        "implicit_capture",
        "population_control",
        "weight_window",
        "domain_decomposition",
        "weight_roulette",
        "iQMC",
        "IC_generator",
        "branchless_collision",
        "uq",
    ]:
        copy_field(mcdc["technique"], input_deck.technique, name)


def set_population_control(mcdc, input_deck):
    # Population control technique (PCT)
    mcdc["technique"]["pct"] = input_deck.technique["pct"]
    mcdc["technique"]["pc_factor"] = input_deck.technique["pc_factor"]


def set_weight_roulette(mcdc, input_deck):
    # Threshold
    mcdc["technique"]["wr_threshold"] = input_deck.technique["wr_threshold"]

    # Survival probability
    mcdc["technique"]["wr_survive"] = input_deck.technique["wr_survive"]


def set_weight_window(mcdc, input_deck):
    # WW mesh
    for name in type_.mesh_names[:-1]:
        copy_field(mcdc["technique"]["ww_mesh"], input_deck.technique["ww_mesh"], name)

    # WW windows
    mcdc["technique"]["ww"] = input_deck.technique["ww"]
    mcdc["technique"]["ww_width"] = input_deck.technique["ww_width"]


# ======================================================================================
# Specials
# ======================================================================================


def set_iQMC(mcdc, input_deck):
    for name in type_.technique["iqmc"].names:
        if name not in [
            "mesh",
            "res",
            "lds",
            "sweep_counter",
            "total_source",
            "material_idx",
            "w_min",
            "score_list",
            "score",
        ]:
            copy_field(mcdc["technique"]["iqmc"], input_deck.technique["iqmc"], name)

    if input_deck.technique["iQMC"]:
        # pass in mesh
        iqmc = mcdc["technique"]["iqmc"]
        for name in ["x", "y", "z", "t"]:
            copy_field(iqmc["mesh"], input_deck.technique["iqmc"]["mesh"], name)
        # pass in score list
        for name, value in input_deck.technique["iqmc"]["score_list"].items():
            iqmc["score_list"][name] = value
        # pass in initial tallies
        for name, value in input_deck.technique["iqmc"]["score"].items():
            mcdc["technique"]["iqmc"]["score"][name] = value
        # minimum particle weight
        iqmc["w_min"] = 1e-13


# =============================================================================
# Utilities
# =============================================================================


# For handling discrepancies between input and program types
def copy_field(dst, src, name):
    if "padding" in name:
        return

    if isinstance(src, dict):
        data = src[name]
    else:
        data = getattr(src, name)

    if isinstance(dst[name], np.ndarray):
        if isinstance(data, np.ndarray) and dst[name].shape != data.shape:
            for dim in data.shape:
                if dim == 0:
                    return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {src[name].shape}"
            )
        elif isinstance(data, list) and dst[name].shape[0] != len(data):
            if len(src[name]) == 0:
                return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {len(src[name])}"
            )

    dst[name] = data


def distribute_work(N, mcdc, precursor=False):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = math.floor(N / size)

    # Starting index (based on even distribution)
    work_start = work_size * rank

    # Count reminder
    rem = N % size

    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1
        work_start += rank
    else:
        work_start += rem

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total


def distribute_work_dd(N, mcdc, precursor=False):
    # Total # of work
    work_size_total = N

    if not mcdc["technique"]["dd_repro"]:
        work_size, work_start = domain_work(mcdc, mcdc["dd_idx"], N)
    else:
        work_start = 0
        work_size = work_size_total

    if not precursor:
        mcdc["mpi_work_start"] = work_start
        mcdc["mpi_work_size"] = work_size
        mcdc["mpi_work_size_total"] = work_size_total
    else:
        mcdc["mpi_work_start_precursor"] = work_start
        mcdc["mpi_work_size_precursor"] = work_size
        mcdc["mpi_work_size_total_precursor"] = work_size_total
