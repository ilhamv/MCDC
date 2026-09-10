import math
import numpy as np

from numba import literal_unroll, njit, objmode
from mpi4py import MPI

####

import mcdc.mcdc_set as mcdc_set
import mcdc.output as output_module
import mcdc.transport.particle_bank as particle_bank_module

from mcdc.constant import (
    GYRATION_RADIUS_ALL,
    GYRATION_RADIUS_INFINITE_X,
    GYRATION_RADIUS_INFINITE_Y,
    GYRATION_RADIUS_INFINITE_Z,
    GYRATION_RADIUS_ONLY_X,
    GYRATION_RADIUS_ONLY_Y,
    GYRATION_RADIUS_ONLY_Z,
)
from mcdc.print_ import print_structure

# ======================================================================================
# Reduce tally bins
# ======================================================================================


@njit
def reduce(simulation, data):
    """Normalize and reduce all tally scores to the master rank."""
    for tally in simulation["tallies"]:
        _reduce(tally, simulation, data)


@njit
def _reduce(tally, simulation, data):
    """Normalize one tally's scores and sum them across MPI ranks."""
    N = tally["bin_length"]
    start = tally["bin_offset"]
    end = start + N

    # Normalize
    N_particle = simulation["settings"]["N_particle"]
    for i in range(N):
        data[start + i] /= N_particle

    # MPI Reduce
    master = simulation["mpi_master"]
    with objmode():
        if master:
            MPI.COMM_WORLD.Reduce(MPI.IN_PLACE, data[start:end], MPI.SUM, 0)
        else:
            MPI.COMM_WORLD.Reduce(data[start:end], None, MPI.SUM, 0)


# ======================================================================================
# Accumulate tally bins
# ======================================================================================


@njit
def accumulate(simulation, data):
    """Accumulate sample moments and reset scores for all tallies."""
    settings = simulation["settings"]
    local_moments = (
        not settings["neutron_eigenvalue_mode"]
        and settings["N_batch"] == 1
        and not settings["use_census_based_tally"]
    )
    for tally in simulation["tallies"]:
        if simulation["mpi_master"] or local_moments:
            _accumulate(tally, data)

        # Reset score bin
        start = tally["bin_offset"]
        for i in range(tally["bin_length"]):
            data[start + i] = 0.0


@njit
def _accumulate(tally, data):
    """Add one tally's scores to its sums and sums of squares."""
    N_bin = tally["bin_length"]
    offset_bin = tally["bin_offset"]
    offset_sum = tally["bin_sum_offset"]
    offset_sum_square = tally["bin_sum_square_offset"]

    # Note: Separate loops are employed to avoid cache miss due to potentially
    #       large N_bin

    # Sum of score
    for i in range(N_bin):
        score = data[offset_bin + i]
        data[offset_sum + i] += score

    # Sum of score squared
    for i in range(N_bin):
        score = data[offset_bin + i]
        data[offset_sum_square + i] += score * score


# ======================================================================================
# Finalize
# ======================================================================================


@njit
def finalize(simulation, data):
    """Finalize all tally statistics and store their combined effective variance."""
    relative_variance = 0.0
    nonzero_bins = 0

    for tally in simulation["tallies"]:
        subtotal, count = _finalize(tally, simulation, data)
        relative_variance += subtotal
        nonzero_bins += count

    if nonzero_bins == 0:
        simulation["effective_variance"] = np.nan
    else:
        simulation["effective_variance"] = relative_variance / nonzero_bins


@njit
def finalize_census(simulation, data):
    """Combine census batches and store effective variance."""
    simulation["effective_variance"] = np.nan

    # Census tallies use batches as independent samples.
    N_sample = simulation["settings"]["N_batch"]
    if not simulation["mpi_master"] or N_sample < 2:
        return

    relative_variance = 0.0
    nonzero_bins = 0

    for tally in simulation["tallies"]:
        for score in range(tally["scores_length"]):
            for census in range(simulation["settings"]["N_census"] - 1):
                # The sum and the sum of squares
                N_bin = tally["bin_length"] // tally["scores_length"]
                sum_ = np.zeros(N_bin)
                sum_sq = np.zeros(N_bin)
                for batch in range(N_sample):
                    with objmode(values="float64[:]"):
                        values = output_module.read_census_score(
                            simulation, data, tally, score, batch, census
                        )
                    for i in range(N_bin):
                        sum_[i] += values[i]
                        sum_sq[i] += values[i] * values[i]

                # Calculate and store statistics
                for i in range(N_bin):
                    # Convert sum into mean
                    sum_[i] = sum_[i] / N_sample

                    # Convert sum of squares into standard error
                    radicand = (sum_sq[i] - N_sample * sum_[i] ** 2) / (N_sample - 1)
                    radicand = radicand / N_sample

                    # Clamp negative variance caused by round-off error.
                    radicand = max(radicand, 0.0)
                    sum_sq[i] = math.sqrt(radicand)

                    # Accumulate squared relative errors for nonzero means.
                    if sum_[i] != 0.0:
                        relative_error = sum_sq[i] / sum_[i]
                        relative_variance += relative_error * relative_error
                        nonzero_bins += 1

    if nonzero_bins > 0:
        simulation["effective_variance"] = relative_variance / nonzero_bins


@njit
def _finalize(tally, simulation, data):
    """Finalize one tally's statistics and return its relative-variance sum and count."""
    N_bin = tally["bin_length"]

    # The sum and the sum of squares
    sum_offset = tally["bin_sum_offset"]
    sum_end = sum_offset + N_bin
    sum_ = data[sum_offset:sum_end]
    sum_sq_offset = tally["bin_sum_square_offset"]
    sum_sq_end = sum_sq_offset + N_bin
    sum_sq = data[sum_sq_offset:sum_sq_end]

    # Determine number of samples
    N_batch = simulation["settings"]["N_batch"]
    N_active = simulation["settings"]["N_active"]
    N_particle = simulation["settings"]["N_particle"]
    if simulation["settings"]["neutron_eigenvalue_mode"]:
        N_sample = N_active
    elif N_batch > 1:
        N_sample = N_batch
    else:
        # History-based sampling
        N_sample = N_particle

        # MPI Reduce
        with objmode():
            if simulation["mpi_master"]:
                MPI.COMM_WORLD.Reduce(MPI.IN_PLACE, sum_, MPI.SUM, 0)
                MPI.COMM_WORLD.Reduce(MPI.IN_PLACE, sum_sq, MPI.SUM, 0)
            else:
                MPI.COMM_WORLD.Reduce(sum_, None, MPI.SUM, 0)
                MPI.COMM_WORLD.Reduce(sum_sq, None, MPI.SUM, 0)

    # All ranks must finish any reductions before workers can return.
    if not simulation["mpi_master"]:
        return 0.0, 0

    # Calculate and store statistics
    relative_variance = 0.0
    nonzero_bins = 0

    for i in range(N_bin):
        # Convert sum into mean
        sum_[i] = sum_[i] / N_sample

        # Convert sum of squares into standard error
        radicand = (sum_sq[i] - N_sample * sum_[i] ** 2) / (N_sample - 1)
        radicand = radicand / N_sample

        # Clamp negative variance caused by round-off error.
        radicand = max(radicand, 0.0)
        sum_sq[i] = math.sqrt(radicand)

        # Accumulate squared relative errors for nonzero means.
        if sum_[i] != 0.0:
            relative_error = sum_sq[i] / sum_[i]
            relative_variance += relative_error * relative_error
            nonzero_bins += 1

    return relative_variance, nonzero_bins


# ======================================================================================
# Reset sum bins
# ======================================================================================


@njit
def reset_sum_bins(simulation, data):
    """Reset accumulated sample moments for all tallies."""
    for tally in simulation["tallies"]:
        _reset_sum_bins(tally, data)


@njit
def _reset_sum_bins(tally, data):
    """Reset one tally's sums and sums of squares."""
    N_bin = tally["bin_length"]
    offset_sum = tally["bin_sum_offset"]
    offset_sum_square = tally["bin_sum_square_offset"]

    for i in range(N_bin):
        data[offset_sum + i] = 0.0
        data[offset_sum_square + i] = 0.0


# ======================================================================================
# Eigenvalue
# ======================================================================================


@njit
def eigenvalue_cycle(simulation, data):
    """Close out one eigenvalue cycle and update global statistics and diagnostics."""
    idx_cycle = simulation["idx_cycle"]
    N_particle = simulation["settings"]["N_particle"]

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_n = np.zeros(1, np.float64)
    buff_nmax = np.zeros(1, np.float64)
    buff_C = np.zeros(1, np.float64)
    buff_Cmax = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(
            np.array(simulation["eigenvalue_tally_nuSigmaF"]), buff_nuSigmaF, MPI.SUM
        )
        if simulation["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array(simulation["eigenvalue_tally_n"]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(
                np.array([simulation["n_max"]]), buff_nmax, MPI.MAX
            )
            MPI.COMM_WORLD.Allreduce(
                np.array(simulation["eigenvalue_tally_C"]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(
                np.array([simulation["C_max"]]), buff_Cmax, MPI.MAX
            )

    # Update and store k_eff
    simulation["k_eff"] = buff_nuSigmaF[0] / N_particle
    mcdc_set.simulation.k_cycle(idx_cycle, simulation, data, value=simulation["k_eff"])

    # Normalize other eigenvalue/global tallies
    tally_n = buff_n[0] / N_particle
    tally_C = buff_C[0] / N_particle

    # Maximum densities
    simulation["n_max"] = buff_nmax[0]
    simulation["C_max"] = buff_Cmax[0]

    # Accumulate running average
    if simulation["cycle_active"]:
        simulation["k_avg"] += simulation["k_eff"]
        simulation["k_sdv"] += simulation["k_eff"] * simulation["k_eff"]
        simulation["n_avg"] += tally_n
        simulation["n_sdv"] += tally_n * tally_n
        simulation["C_avg"] += tally_C
        simulation["C_sdv"] += tally_C * tally_C

        N = 1 + simulation["idx_cycle"] - simulation["settings"]["N_inactive"]
        simulation["k_avg_running"] = simulation["k_avg"] / N
        if N == 1:
            simulation["k_sdv_running"] = 0.0
        else:
            simulation["k_sdv_running"] = math.sqrt(
                (simulation["k_sdv"] / N - simulation["k_avg_running"] ** 2) / (N - 1)
            )

    # Reset accumulators
    simulation["eigenvalue_tally_nuSigmaF"][0] = 0.0
    simulation["eigenvalue_tally_n"][0] = 0.0
    simulation["eigenvalue_tally_C"][0] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if simulation["settings"]["use_gyration_radius"]:
        # Center of mass
        N_local = particle_bank_module.get_bank_size(simulation["bank_census"])
        total_local = np.zeros(4, np.float64)  # [x,y,z,W]
        total = np.zeros(4, np.float64)
        for i in range(N_local):
            P = simulation["bank_census"]["particle_data"][i]
            total_local[0] += P["x"] * P["w"]
            total_local[1] += P["y"] * P["w"]
            total_local[2] += P["z"] * P["w"]
            total_local[3] += P["w"]
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W = total[3]
        com_x = total[0] / W
        com_y = total[1] / W
        com_z = total[2] / W

        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms = np.zeros(1, np.float64)
        gr_type = simulation["settings"]["gyration_radius_type"]
        if gr_type == GYRATION_RADIUS_ALL:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += (
                    (P["x"] - com_x) ** 2
                    + (P["y"] - com_y) ** 2
                    + (P["z"] - com_z) ** 2
                ) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_X:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Y:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Z:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_X:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Y:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Z:
            for i in range(N_local):
                P = simulation["bank_census"]["particle_data"][i]
                rms_local[0] += ((P["z"] - com_z) ** 2) * P["w"]

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0] / W)

        # Gyration radius
        mcdc_set.simulation.gyration_radius(idx_cycle, simulation, data, value=rms)


@njit
def eigenvalue_simulation(simulation):
    """Finalize neutron and precursor density statistics over active cycles."""
    N = simulation["settings"]["N_active"]
    simulation["n_avg"] /= N
    simulation["C_avg"] /= N
    if N > 1:
        simulation["n_sdv"] = math.sqrt(
            (simulation["n_sdv"] / N - simulation["n_avg"] ** 2) / (N - 1)
        )
        simulation["C_sdv"] = math.sqrt(
            (simulation["C_sdv"] / N - simulation["C_avg"] ** 2) / (N - 1)
        )
    else:
        simulation["n_sdv"] = 0.0
        simulation["C_sdv"] = 0.0
