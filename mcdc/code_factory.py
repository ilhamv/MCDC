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
