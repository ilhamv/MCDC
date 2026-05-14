import harmonize

from numba import njit

###

import mcdc.code_factory.gpu.program_builder as gpu_module
import mcdc.config as config
import mcdc.transport.particle_bank as particle_bank_module

from mcdc.constant import GPU_STORAGE_SEPARATE, GPU_STRATEGY_ASYNC
from mcdc.transport.simulation import source_closeout

caching = config.caching


@njit(cache=caching)
def source_loop(seed, simulation, data):
    # For async execution
    iter_count = 655360000
    # For event-based execution
    batch_size = 64

    settings = simulation["settings"]

    full_work_size = simulation["mpi_work_size"]

    if settings["gpu_strategy"] == GPU_STRATEGY_ASYNC:
        phase_size = 1000000000
    else:
        phase_size = 1000000
    phase_count = (full_work_size + phase_size - 1) // phase_size

    for phase in range(phase_count):

        simulation["mpi_work_iter"][0] = phase_size * phase
        simulation["mpi_work_size"] = min(phase_size * (phase + 1), full_work_size)
        simulation["source_seed"] = seed

        # Store the global state to the GPU
        if settings["gpu_storage"] == GPU_STORAGE_SEPARATE:
            harmonize.memcpy_host_to_device(
                simulation["gpu_meta"]["state_pointer"], simulation
            )
            harmonize.memcpy_host_to_device(
                simulation["gpu_meta"]["state_pointer"], data
            )

        # Execute the program, and continue to do so until it is done
        block_count = gpu_module.BLOCK_COUNT

        if settings["gpu_strategy"] == GPU_STRATEGY_ASYNC:
            gpu_module.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, iter_count
            )
            while not gpu_module.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_module.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, iter_count
                )
        else:
            gpu_module.exec_program(
                simulation["gpu_meta"]["program_pointer"], block_count, batch_size
            )
            while not gpu_module.complete(simulation["gpu_meta"]["program_pointer"]):
                gpu_module.exec_program(
                    simulation["gpu_meta"]["program_pointer"], block_count, batch_size
                )
        gpu_module.clear_flags(simulation["gpu_meta"]["program_pointer"])

        # Recover the original program state
        src_load_constant(mcdc, mcdc["gpu_state_pointer"])
        src_load_data(data, mcdc["gpu_state_pointer"])
        src_clear_flags(mcdc["source_program_pointer"])

    mcdc["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(mcdc["bank_active"], 0)

    # =====================================================================
    # Closeout (Moved out of the typical particle loop)
    # =====================================================================

    source_closeout(mcdc, 1, 1, data)

    if mcdc["technique"]["domain_decomposition"]:
        source_dd_resolution(data, mcdc)


def build_gpu_progs(input_deck, args):

    STRAT = args.gpu_strategy

    src_spec = gpu_sources_spec()

    adapt.harm.RuntimeSpec.bind_specs()

    rank = MPI.COMM_WORLD.Get_rank()
    device_id = rank % args.gpu_share_stride

    if MPI.COMM_WORLD.Get_size() > 1:
        MPI.COMM_WORLD.Barrier()

    adapt.harm.RuntimeSpec.load_specs()

    if STRAT == "async":
        args.gpu_arena_size = args.gpu_arena_size // 32
        src_fns = src_spec.async_functions()
        pre_fns = pre_spec.async_functions()
    else:
        src_fns = src_spec.event_functions()
        pre_fns = pre_spec.event_functions()

    ARENA_SIZE = args.gpu_arena_size
    BLOCK_COUNT = args.gpu_block_count

    global alloc_state, free_state
    alloc_state = src_fns["alloc_state"]
    free_state = src_fns["free_state"]

    global src_alloc_program, src_free_program
    global src_load_global, src_store_global, src_load_data, src_store_data, src_store_pointer_data
    global src_init_program, src_exec_program, src_complete, src_clear_flags
    src_alloc_program = src_fns["alloc_program"]
    src_free_program = src_fns["free_program"]
    src_load_global = src_fns["load_state_device_global"]
    src_store_global = src_fns["store_state_device_global"]
    src_store_pointer_global = src_fns["store_pointer_state_device_global"]
    src_load_data = src_fns["load_state_device_data"]
    src_store_data = src_fns["store_state_device_data"]
    src_store_pointer_data = src_fns["store_pointer_state_device_data"]
    src_init_program = src_fns["init_program"]
    src_exec_program = src_fns["exec_program"]
    src_complete = src_fns["complete"]
    src_clear_flags = src_fns["clear_flags"]
    src_set_device = src_fns["set_device"]

    global pre_alloc_program, pre_free_program
    global pre_load_global, pre_store_global, pre_load_data, pre_store_data
    global pre_init_program, pre_exec_program, pre_complete, pre_clear_flags
    pre_alloc_state = pre_fns["alloc_state"]
    pre_free_state = pre_fns["free_state"]
    pre_alloc_program = pre_fns["alloc_program"]
    pre_free_program = pre_fns["free_program"]
    pre_load_global = pre_fns["load_state_device_global"]
    pre_store_global = pre_fns["store_state_device_global"]
    pre_load_data = pre_fns["load_state_device_data"]
    pre_store_data = pre_fns["store_state_device_data"]
    pre_init_program = pre_fns["init_program"]
    pre_exec_program = pre_fns["exec_program"]
    pre_complete = pre_fns["complete"]
    pre_clear_flags = pre_fns["clear_flags"]

    @njit
    def real_setup_gpu(mcdc_array, data_tally):
        mcdc = mcdc_array[0]

        print("STATE POINTER {mcdc['gpu_meta']['state_pointer']}")
        print("GLOBAL POINTER {mcdc['gpu_meta']['global_pointer']}")
        print("TALLY POINTER {mcdc['gpu_meta']['tally_pointer']}")
        src_set_device(device_id)
        arena_size = ARENA_SIZE
        mcdc["gpu_meta"]["state_pointer"] = adapt.cast_voidptr_to_uintp(alloc_state())
        # src_store_global(mcdc["gpu_meta"]["state_pointer"], mcdc_array[0])
        if config.gpu_state_storage == "separate":
            harmonize.memcpy_device_to_host(
                simulation, simulation["gpu_meta"]["state_pointer"]
            )
            harmonize.memcpy_device_to_host(
                data, simulation["gpu_meta"]["state_pointer"]
            )

        gpu_module.clear_flags(simulation["gpu_meta"]["program_pointer"])

    simulation["mpi_work_size"] = full_work_size

    particle_bank_module.set_bank_size(simulation["bank_active"], 0)

    source_closeout(simulation, 1, 1, data)
