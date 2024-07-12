import argparse

from mpi4py import MPI

from mcdc.card import UniverseCard


# ======================================================================================
# Main functions
# ======================================================================================
"""Functions directly called by main.py"""


def parse_cmd_options():
    """
    Parse command-line options which will override input deck values.
    """
    parser = argparse.ArgumentParser(description="MC/DC: Monte Carlo Dynamic Code")

    # Run mode
    parser.add_argument(
        "--mode",
        type=str,
        help="Run mode",
        choices=["python", "numba", "numba-debug"],
        default="python",
    )

    # Hardware target
    parser.add_argument(
        "--target",
        type=str,
        help="Hardware target",
        choices=["cpu", "gpu"],
        default="cpu",
    )

    # Number of particles
    parser.add_argument(
        "--N_particle",
        type=int,
        help="Number of particles",
    )

    # Output file name
    parser.add_argument(
        "--output",
        type=str,
        help="Output file name",
    )

    # Progress bar toggles
    parser.add_argument("--progress_bar", default=True, action="store_true")
    parser.add_argument("--no-progress_bar", dest="progress_bar", action="store_false")

    # JIT compile caching toggles
    parser.add_argument("--cache", default=True, action="store_true")
    parser.add_argument("--no-cache", dest="cache", action="store_false")

    args, unargs = parser.parse_known_args()

    return args


def override_input_deck(input_deck, cmd_option):
    """Override input deck values based on captured command-line options."""
    # Simulation parameters
    if cmd_option.N_particle is not None:
        input_deck.setting["N_particle"] = cmd_option.N_particle

    # Output settings
    if cmd_option.output is not None:
        input_deck.setting["output_name"] = cmd_option.output
    if cmd_option.progress_bar is not None:
        input_deck.setting["progress_bar"] = cmd_option.progress_bar

    # Numba-related flags
    mode = cmd_option.mode
    if mode == "python":
        input_deck.setting["numba_jit"] = False
    elif mode == "numba":
        input_deck.setting["numba_jit"] = True
    elif mode == "numba-debug":
        input_deck.setting["numba_jit"] = True
        input_deck.setting["numba_debug"] = True
    input_deck.setting["numba_cache"] = cmd_option.cache

    # Hardware target
    target = cmd_option.target
    if target is not None:
        input_deck.setting["hardware_target"] = target


def finalize_input_deck(input_deck):
    # If not defined, create root universe that contains all the defined cells
    if input_deck.universes[0] == None:
        N_cell = len(input_deck.cells)
        root_universe = UniverseCard(N_cell)
        root_universe.ID = 0
        for i, cell in enumerate(input_deck.cells):
            root_universe.cell_IDs[i] = cell.ID
        input_deck.universes[0] = root_universe

    # Finalize inputs related to domain decomposition
    finalize_dd_input(input_deck)


def set_numba_env_variables(numba, input_deck):
    """Set Numba environment variables"""

    if input_deck.setting["numba_jit"]:
        numba.config.DISABLE_JIT = False
    else:
        numba.config.DISABLE_JIT = True

    if input_deck.setting["numba_cache"]:
        numba.config.NUMBA_DEBUG_CACHE = 1
    else:
        numba.config.NUMBA_DEBUG_CACHE = 0

    if input_deck.setting["numba_debug"]:
        msg = (
            "\n >> Entering numba debug mode"
            + "\n >> will result in slower code and longer compile times"
            + "\n >> to configure debug options see prep.py"
        )
        print_warning(msg)

        # Turn on debugging options
        numba.config.DEBUG = 1

        # Enable errors from sub-packages to be printed
        numba.config.NUMBA_FULL_TRACEBACKS = 1

        # Check bounds errors of vectors
        numba.config.NUMBA_BOUNDSCHECK = 1

        # Print error messages for dark background terminals
        numba.config.NUMBA_COLOR_SCHEME = "dark_bg"

        # Numba run time (NRT) statistics counter
        numba.config.NUMBA_DEBUG_NRT = 1

        # Print out debugging information about type inference.
        numba.config.NUMBA_DEBUG_TYPEINFER = 1

        # Enable profiler use
        numba.config.NUMBA_ENABLE_PROFILING = 1

        # Print out a control flow diagram
        numba.config.NUMBA_DUMP_CFG = 1

        # Form unoptimized code from compilers
        numba.config.NUMBA_OPT = 0

        numba.config.NUMBA_DEBUGINFO = 1

        # Allow for inspection of numba variables after end of compilation
        numba.config.NUMBA_EXTEND_VARIABLE_LIFETIMES = 1


# ======================================================================================
# Domain decomposition
# ======================================================================================


def finalize_dd_input(input_deck):
    work_ratio = input_deck.technique["dd_work_ratio"]

    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1

    input_deck.setting["bank_active_buff"] = 1000
    if input_deck.technique["dd_exchange_rate"] == None:
        input_deck.technique["dd_exchange_rate"] = 100

    if work_ratio is None:
        work_ratio = np.ones(d_Nx * d_Ny * d_Nz)
        input_deck.technique["dd_work_ratio"] = work_ratio

    if (
        input_deck.technique["domain_decomposition"]
        and np.sum(work_ratio) != MPI.COMM_WORLD.Get_size()
    ):
        print_msg(
            "Domain work ratio not equal to number of processors, %i != %i "
            % (np.sum(work_ratio), MPI.COMM_WORLD.Get_size())
        )
        exit()

    if input_deck.technique["domain_decomposition"]:
        # Assigning domain index
        i = 0
        rank_info = []
        for n in range(d_Nx * d_Ny * d_Nz):
            ranks = []
            for r in range(int(work_ratio[n])):
                ranks.append(i)
                if MPI.COMM_WORLD.Get_rank() == i:
                    d_idx = n
                i += 1
            rank_info.append(ranks)
        input_deck.technique["dd_idx"] = d_idx
        xn, xp, yn, yp, zn, zp = get_neighbors(d_idx, 0, d_Nx, d_Ny, d_Nz)
    else:
        input_deck.technique["dd_idx"] = 0
        input_deck.technique["dd_xp_neigh"] = []
        input_deck.technique["dd_xn_neigh"] = []
        input_deck.technique["dd_yp_neigh"] = []
        input_deck.technique["dd_yn_neigh"] = []
        input_deck.technique["dd_zp_neigh"] = []
        input_deck.technique["dd_zn_neigh"] = []
        return

    if xp is not None:
        input_deck.technique["dd_xp_neigh"] = rank_info[xp]
    else:
        input_deck.technique["dd_xp_neigh"] = []
    if xn is not None:
        input_deck.technique["dd_xn_neigh"] = rank_info[xn]
    else:
        input_deck.technique["dd_xn_neigh"] = []

    if yp is not None:
        input_deck.technique["dd_yp_neigh"] = rank_info[yp]
    else:
        input_deck.technique["dd_yp_neigh"] = []
    if yn is not None:
        input_deck.technique["dd_yn_neigh"] = rank_info[yn]
    else:
        input_deck.technique["dd_yn_neigh"] = []

    if zp is not None:
        input_deck.technique["dd_zp_neigh"] = rank_info[zp]
    else:
        input_deck.technique["dd_zp_neigh"] = []
    if zn is not None:
        input_deck.technique["dd_zn_neigh"] = rank_info[zn]
    else:
        input_deck.technique["dd_zn_neigh"] = []


def get_neighbors(N, w, nx, ny, nz):
    i, j, k = get_indexes(N, nx, ny)
    if i > 0:
        xn = get_domain_idx(i - 1, j, k, nx, ny)
    else:
        xn = None
    if i < (nx - 1):
        xp = get_domain_idx(i + 1, j, k, nx, ny)
    else:
        xp = None
    if j > 0:
        yn = get_domain_idx(i, j - 1, k, nx, ny)
    else:
        yn = None
    if j < (ny - 1):
        yp = get_domain_idx(i, j + 1, k, nx, ny)
    else:
        yp = None
    if k > 0:
        zn = get_domain_idx(i, j, k - 1, nx, ny)
    else:
        zn = None
    if k < (nz - 1):
        zp = get_domain_idx(i, j, k + 1, nx, ny)
    else:
        zp = None
    return xn, xp, yn, yp, zn, zp


def get_domain_idx(i, j, k, ni, nj):
    N = i + j * ni + k * ni * nj
    return N


def get_indexes(N, nx, ny):
    k = int(N / (nx * ny))
    j = int((N - nx * ny * k) / nx)
    i = int(N - nx * ny * k - nx * j)
    return i, j, k
