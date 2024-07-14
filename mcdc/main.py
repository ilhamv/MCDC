import h5py, numba

import numpy as np

from mpi4py import MPI

import mcdc.adapt as adapt
import mcdc.code_factory as code_factory
import mcdc.global_ as global_
import mcdc.initializer as initializer
import mcdc.prep as prep
import mcdc.rng as rng
import mcdc.simulation as simulation
import mcdc.type_ as type_

#from mcdc.iqmc.iqmc_loop import iqmc_simulation
from mcdc.print_ import (
    print_banner,
    print_error,
    print_header_eigenvalue,
    print_msg,
    print_runtime,
    print_warning,
)


def run():
    # Start timer
    total_start = MPI.Wtime()

    # ==================================================================================
    # Preparation
    # ==================================================================================

    # Get input deck
    input_deck = global_.input_deck

    # Get command line options
    cmd_option = prep.parse_cmd_options()

    # Override input deck with command line options
    prep.override_input_deck(input_deck, cmd_option)

    # Finalize input deck
    prep.finalize_input_deck(input_deck)

    # Set Numba environment variables
    prep.set_numba_env_variables(numba, input_deck)

    # ==================================================================================
    # Code factory
    # ==================================================================================

    # Make types
    type_.make_type_particle(input_deck)
    type_.make_type_particle_record(input_deck)
    type_.make_type_nuclide(input_deck)
    type_.make_type_material(input_deck)
    type_.make_type_surface(input_deck)
    type_.make_type_region()
    type_.make_type_cell(input_deck)
    type_.make_type_universe(input_deck)
    type_.make_type_lattice(input_deck)
    type_.make_type_source(input_deck)
    type_.make_type_tally(input_deck)
    type_.make_type_setting(input_deck)
    type_.make_type_uq_tally(input_deck)
    type_.make_type_uq(input_deck)
    type_.make_type_domain_decomp(input_deck)
    type_.make_type_dd_turnstile_event(input_deck)
    type_.make_type_technique(input_deck)
    type_.make_type_global(input_deck)
    #
    type_.make_type_translate(input_deck)
    type_.make_type_group_array(input_deck)
    type_.make_type_j_array(input_deck)

    # Create the global variable container
    mcdc = np.zeros(1, dtype=type_.global_)[0]

    # Set Numba-JIT decorator to all functions in selected modules
    code_factory.numba_jit(rng, input_deck, numba)

    # Set random number generator
    code_factory.set_rng(rng, input_deck)

    # ==============
    # Initialization
    # ==============

    # Simulation objects
    initializer.set_nuclides(mcdc, input_deck)
    initializer.set_materials(mcdc, input_deck)
    initializer.set_surfaces(mcdc, input_deck)
    initializer.set_regions(mcdc, input_deck)
    initializer.set_cells(mcdc, input_deck)
    initializer.set_universes(mcdc, input_deck)
    initializer.set_lattices(mcdc, input_deck)
    initializer.set_sources(mcdc, input_deck)
    initializer.set_tally(mcdc, input_deck)

    # Simulation settings
    initializer.set_setting(mcdc, input_deck)
    initializer.set_particle_banks(mcdc, input_deck)
    initializer.set_eigenvalue(mcdc, input_deck)
    initializer.set_mpi(mcdc, input_deck)

    # Occasional settings
    initializer.set_domain_decomposition(mcdc, input_deck)
    initializer.set_uncertainty_quantification(mcdc, input_deck)
    initializer.set_sensitivity_quantification(mcdc, input_deck)
    initializer.set_source_files(mcdc, input_deck)
    initializer.set_IC_generator(mcdc, input_deck)

    # Techniques (for variance and runtime reductions)
    initializer.set_technique_flags(mcdc, input_deck)
    initializer.set_population_control(mcdc, input_deck)
    initializer.set_weight_roulette(mcdc, input_deck)
    initializer.set_weight_window(mcdc, input_deck)

    # Specials
    initializer.set_iQMC(mcdc, input_deck)

    # ==========================================
    # Platform Targeting, Adapters, Toggles, etc
    # ==========================================

    target = input_deck.setting['hardware_target']
    if target == "gpu":
        if not adapt.HAS_HARMONIZE:
            print_error(
                "No module named 'harmonize' - GPU functionality not available. "
            )
        adapt.gpu_forward_declare()

    adapt.set_toggle("iQMC", input_deck.technique["iQMC"])
    adapt.set_toggle("domain_decomp", input_deck.technique["domain_decomposition"])
    adapt.eval_toggle()
    adapt.target_for(target)

    if target == "gpu":
        build_gpu_progs()

    adapt.nopython_mode(input_deck.setting['numba_jit'])

    loop.setup_gpu(mcdc)

   # Hit timer
    mcdc["runtime_preparation"] = MPI.Wtime() - total_start

    # ==================================================================================
    # Run simulatthe simulation
    # ==================================================================================

    # Print banner, hardware configuration, and header
    print_banner(mcdc)

    print_msg(" Now running TNT...")
    if mcdc["setting"]["mode_eigenvalue"]:
        print_header_eigenvalue(mcdc)

    # Run simulation
    simulation_start = MPI.Wtime()
    if mcdc["technique"]["iQMC"]:
        iqmc_simulation(mcdc)
    elif mcdc["setting"]["mode_eigenvalue"]:
        loop_eigenvalue(mcdc)
    else:
        simulation.fixed_source(mcdc)
    mcdc["runtime_simulation"] = MPI.Wtime() - simulation_start

    # Output: generate hdf5 output files
    output_start = MPI.Wtime()
    generate_hdf5(mcdc)
    mcdc["runtime_output"] = MPI.Wtime() - output_start

    # Stop timer
    MPI.COMM_WORLD.Barrier()
    mcdc["runtime_total"] = MPI.Wtime() - total_start

    # Closout
    closeout(mcdc)



def cardlist_to_h5group(dictlist, input_group, name):
    main_group = input_group.create_group(name + "s")
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % getattr(item, "ID"))
        card_to_h5group(item, group)


def card_to_h5group(card, group):
    for name in [
        a
        for a in dir(card)
        if not a.startswith("__") and not callable(getattr(card, a)) and a != "tag"
    ]:
        value = getattr(card, name)
        if type(value) == dict:
            dict_to_h5group(value, group.create_group(name))
        elif value is None:
            next
        else:
            group[name] = value


def dictlist_to_h5group(dictlist, input_group, name):
    main_group = input_group.create_group(name + "s")
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        elif v is None:
            next
        else:
            group[k] = v


def generate_hdf5(mcdc):
    if mcdc["mpi_master"]:
        if mcdc["setting"]["progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "w") as f:
            # Input deck
            if mcdc["setting"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                cardlist_to_h5group(input_deck.nuclides, input_group, "nuclide")
                cardlist_to_h5group(input_deck.materials, input_group, "material")
                cardlist_to_h5group(input_deck.surfaces, input_group, "surface")
                cardlist_to_h5group(input_deck.cells, input_group, "cell")
                cardlist_to_h5group(input_deck.universes, input_group, "universe")
                cardlist_to_h5group(input_deck.lattices, input_group, "lattice")
                cardlist_to_h5group(input_deck.sources, input_group, "source")
                dict_to_h5group(input_deck.tally, input_group.create_group("tally"))
                dict_to_h5group(input_deck.setting, input_group.create_group("setting"))
                dict_to_h5group(
                    input_deck.technique, input_group.create_group("technique")
                )

            # Tally
            T = mcdc["tally"]
            f.create_dataset("tally/grid/t", data=T["mesh"]["t"])
            f.create_dataset("tally/grid/x", data=T["mesh"]["x"])
            f.create_dataset("tally/grid/y", data=T["mesh"]["y"])
            f.create_dataset("tally/grid/z", data=T["mesh"]["z"])
            f.create_dataset("tally/grid/mu", data=T["mesh"]["mu"])
            f.create_dataset("tally/grid/azi", data=T["mesh"]["azi"])
            f.create_dataset("tally/grid/g", data=T["mesh"]["g"])

            # Scores
            for name in T["score"].dtype.names:
                if mcdc["tally"][name]:
                    name_h5 = name.replace("_", "-")
                    f.create_dataset(
                        "tally/" + name_h5 + "/mean",
                        data=np.squeeze(T["score"][name]["mean"]),
                    )
                    f.create_dataset(
                        "tally/" + name_h5 + "/sdev",
                        data=np.squeeze(T["score"][name]["sdev"]),
                    )
                    if mcdc["technique"]["uq_tally"][name]:
                        mc_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_var"
                        ]
                        tot_var = mcdc["technique"]["uq_tally"]["score"][name][
                            "batch_bin"
                        ]
                        f.create_dataset(
                            "tally/" + name_h5 + "/uq_var",
                            data=np.squeeze(tot_var - mc_var),
                        )

            # Eigenvalues
            if mcdc["setting"]["mode_eigenvalue"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                else:
                    N_cycle = mcdc["setting"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["setting"]["gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # dump iQMC mesh
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc"]["mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc"]["mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc"]["mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc"]["mesh"]["z"])
                # dump x,y,z scalar flux across all groups
                f.create_dataset(
                    "iqmc/tally/flux", data=np.squeeze(T["iqmc"]["score"]["flux"])
                )
                f.create_dataset(
                    "iqmc/tally/fission_source",
                    data=T["iqmc"]["score"]["fission-source"],
                )
                f.create_dataset(
                    "iqmc/tally/fission_power", data=T["iqmc"]["score"]["fission-power"]
                )
                f.create_dataset("iqmc/tally/source_constant", data=T["iqmc"]["source"])
                f.create_dataset(
                    "iqmc/tally/source_x", data=T["iqmc"]["score"]["tilt-x"]
                )
                f.create_dataset(
                    "iqmc/tally/source_y", data=T["iqmc"]["score"]["tilt-y"]
                )
                f.create_dataset(
                    "iqmc/tally/source_z", data=T["iqmc"]["score"]["tilt-z"]
                )
                # iteration data
                f.create_dataset("iqmc/itteration_count", data=T["iqmc"]["itt"])
                f.create_dataset("iqmc/final_residual", data=T["iqmc"]["res"])
                f.create_dataset("iqmc/sweep_count", data=T["iqmc"]["sweep_counter"])
                if mcdc["setting"]["mode_eigenvalue"]:
                    f.create_dataset(
                        "iqmc/outter_itteration_count", data=T["iqmc"]["itt_outter"]
                    )
                    f.create_dataset(
                        "iqmc/outter_final_residual", data=T["iqmc"]["res_outter"]
                    )

            # IC generator
            if mcdc["technique"]["IC_generator"]:
                Nn = mcdc["technique"]["IC_bank_neutron"]["size"]
                Np = mcdc["technique"]["IC_bank_precursor"]["size"]
                f.create_dataset(
                    "IC/neutrons",
                    data=mcdc["technique"]["IC_bank_neutron"]["particles"][:Nn],
                )
                f.create_dataset(
                    "IC/precursors",
                    data=mcdc["technique"]["IC_bank_precursor"]["precursors"][:Np],
                )
                f.create_dataset("IC/neutrons_size", data=Nn)
                f.create_dataset("IC/precursors_size", data=Np)
                f.create_dataset(
                    "IC/fission", data=mcdc["technique"]["IC_fission"] / Nn
                )

    # Save particle?
    if mcdc["setting"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def closeout(mcdc):

    loop.teardown_gpu(mcdc)

    # Runtime
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

    print_runtime(mcdc)
    input_deck.reset()
