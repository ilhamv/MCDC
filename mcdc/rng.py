"""
Random Number Generator (RNG)

Currently, MC/DC uses LCG with hash-based seeding
[https://doi.org/10.48550/arXiv.2403.06362]
"""
import numba as nb
import numpy as np


from mcdc.constant import (
    RNG_G,
    RNG_C,
    RNG_MOD,
    RNG_MOD_MASK
)


# ======================================================================================
# Wrapping addition and multiplication
# ======================================================================================
"""
These are needed because
  (1) Python and Numba modes behave differently when overflow occurs and
  (2) Python recasts uint parameter into int.

Check function `set_rng` in `code_factory.py` for the determination of the functions
"""
# TODO: better idea?


add = None
multiply = None


def add_numba(a, b):
    return a + b


def multiply_numba(a, b):
    return a * b


def add_python(a, b):
    a = nb.uint64(a)
    b = nb.uint64(b)
    with np.errstate(all="ignore"):
        return a + b


def multiply_python(a, b):
    a = nb.uint64(a)
    b = nb.uint64(b)
    with np.errstate(all="ignore"):
        return a * b


# ======================================================================================
# Interfacing functions
# ======================================================================================


def lcg(seed):
    seed = nb.uint64(seed)
    return add(multiply(RNG_G, seed), RNG_C) & RNG_MOD_MASK


def split_seed(key, seed):
    """murmur_hash64a"""
    multiplier = nb.uint64(0xC6A4A7935BD1E995)
    length = nb.uint64(8)
    rotator = nb.uint64(47)
    key = nb.uint64(key)
    seed = nb.uint64(seed)

    hash_value = nb.uint64(seed) ^ multiply(length, multiplier)

    key = multiply(key, multiplier)
    key ^= key >> rotator
    key = multiply(key, multiplier)
    hash_value ^= key
    hash_value = multiply(hash_value, multiplier)

    hash_value ^= hash_value >> rotator
    hash_value = multiply(hash_value, multiplier)
    hash_value ^= hash_value >> rotator
    return hash_value


def random(state):
    state["rng_seed"] = lcg(state["rng_seed"])
    return state["rng_seed"] / RNG_MOD


def random_from_seed(seed):
    return lcg(seed) / RNG_MOD


def random_array(seed, shape, size):
    xi = np.zeros(size)
    for i in range(size):
        xi_seed = split_seed(i, seed)
        xi[i] = random_from_seed(xi_seed)
    xi = xi.reshape(shape)
    return xi
