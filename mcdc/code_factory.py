# Get Harmonize
# https://github.com/CEMeNT-PSAAP/harmonize
try:
    import harmonize as harm
    HAS_HARMONIZE = True
except:
    HAS_HARMONIZE = False

from mcdc.print_ import print_error


# ======================================================================================
# Main functions
# ======================================================================================
"""Functions directly called by main.py"""


def numba_jit(module, input_deck, numba):
    """Adapt all functions in the module."""
    # Add JIT decorator
    cache = input_deck.setting['numba_cache']
    for name in [x for x in dir(module) if not x.startswith("__")]:
        function = getattr(module, name)
        if callable(function):
            setattr(module, name, numba.njit(function, cache=cache))


def set_rng(rng, input_deck):
    python_mode = not input_deck.setting['numba_jit']

    # Set the wrapping addition and multiplication (see `rng.py`)
    if python_mode:
        rng.add = rng.add_python
        rng.multiply = rng.multiply_python
    else:
        rng.add = rng.add_numba
        rng.multiply = rng.multiply_numba


# ======================================================================================
# GPU Type / Extern Functions Forward Declarations
# =============================================================================


SIMPLE_ASYNC = True

none_type = None
mcdc_type = None
state_spec = None
device_gpu = None
group_gpu = None
thread_gpu = None
particle_gpu = None
prep_gpu = None
step_async = None
find_cell_async = None


def gpu_forward_declare():
    if not code_factory.HAS_HARMONIZE:
        print_error(
            "No module named 'harmonize' - GPU functionality not available."
        )

    global none_type, mcdc_type, state_spec
    global device_gpu, group_gpu, thread_gpu
    global particle_gpu, particle_record_gpu
    global step_async, find_cell_async

    none_type = numba.from_dtype(np.dtype([]))
    mcdc_type = numba.from_dtype(type_.global_)
    state_spec = (mcdc_type, none_type, none_type)
    device_gpu, group_gpu, thread_gpu = harm.RuntimeSpec.access_fns(state_spec)
    particle_gpu = numba.from_dtype(type_.particle)
    particle_record_gpu = numba.from_dtype(type_.particle_record)

    def step(prog: numba.uintp, P: particle_gpu):
        pass

    def find_cell(prog: numba.uintp, P: particle_gpu):
        pass

    step_async, find_cell_async = harm.RuntimeSpec.async_dispatch(step, find_cell)

# ======================================================================================
# Utilities
# ======================================================================================

def unknown_target(target):
    print_error(f"ERROR: Unrecognized target '{target}'")
