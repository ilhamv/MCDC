def fixed_source(mcdc):
    # Loop over batches
    for idx_batch in range(mcdc["setting"]["N_batch"]):
        mcdc["idx_batch"] = idx_batch
        seed_batch = kernel.split_seed(idx_batch, mcdc["setting"]["rng_seed"])

        # Print multi-batch header
        if mcdc["setting"]["N_batch"] > 1:
            with objmode():
                print_header_batch(mcdc)
            if mcdc["technique"]["uq"]:
                seed_uq = kernel.split_seed(seed_batch, SEED_SPLIT_UQ)
                kernel.uq_reset(mcdc, seed_uq)

        # Loop over time censuses
        for idx_census in range(mcdc["setting"]["N_census"]):
            mcdc["idx_census"] = idx_census
            seed_census = kernel.split_seed(seed_batch, SEED_SPLIT_CENSUS)

            # Loop over source particles
            seed_source = kernel.split_seed(seed_census, SEED_SPLIT_SOURCE)
            loop_source(seed_source, mcdc)

            # Loop over source precursors
            if kernel.get_bank_size(mcdc["bank_precursor"]) > 0:
                seed_source_precursor = kernel.split_seed(
                    seed_census, SEED_SPLIT_SOURCE_PRECURSOR
                )
                loop_source_precursor(seed_source_precursor, mcdc)

            # Time census closeout
            if idx_census < mcdc["setting"]["N_census"] - 1:
                # TODO: Output tally (optional)

                # Manage particle banks: population control and work rebalance
                seed_bank = kernel.split_seed(seed_census, SEED_SPLIT_BANK)
                kernel.manage_particle_banks(seed_bank, mcdc)

        # Multi-batch closeout
        if mcdc["setting"]["N_batch"] > 1:
            # Reset banks
            kernel.set_bank_size(mcdc["bank_source"], 0)
            kernel.set_bank_size(mcdc["bank_census"], 0)
            kernel.set_bank_size(mcdc["bank_active"], 0)

            # Tally history closeout
            kernel.tally_reduce_bin(mcdc)
            kernel.tally_closeout_history(mcdc)
            # Uq closeout
            if mcdc["technique"]["uq"]:
                kernel.uq_tally_closeout_batch(mcdc)

    # Tally closeout
    if mcdc["technique"]["uq"]:
        kernel.uq_tally_closeout(mcdc)
    else:
        kernel.tally_closeout(mcdc)


def adapt_simulations(input_deck):
    global fixed_source

    cache = input_deck.setting['numba_cache']

    if input_deck.setting['numba_jit']:
        fixed_source = nb.njit(fixed_source, cache=cache)
