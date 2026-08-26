# MC/DC Data Library Generator

Converts ACE-format nuclear data files into MC/DC's per-nuclide HDF5 format
for continuous-energy neutron transport.

## Prerequisites

```bash
pip install h5py numpy tqdm
```

## Install ACEtk from source

[ACEtk](https://github.com/njoy/ACEtk)

You need a collection of ACE files (e.g., from NJOY or an ENDF/B distribution).

## Environment Variables

| Variable      | Description                                           |
|---------------|-------------------------------------------------------|
| `MCDC_ACELIB` | Path to the directory containing your ACE files.      |
| `MCDC_LIB`    | Path to the output directory for MC/DC HDF5 files.    |

## Usage

```bash
export MCDC_ACELIB=/path/to/ace/files
export MCDC_LIB=/path/to/mcdc/library

python generate.py              # Convert only missing nuclides
python generate.py --rewrite    # Regenerate all files
python generate.py --verbose    # Print detailed per-nuclide info
```

## What it Does

For each ACE file in `$MCDC_ACELIB`, the generator:

1. Uses ACEtk's parsed table metadata to identify the nuclide (Z, A, isomeric
   state), then reads the table suffix and physical temperature from the ACE
   header. The ACE thermal energy is converted from MeV to kelvin.
2. Extracts pointwise cross sections (elastic, capture, inelastic, fission) and the energy grid.
3. Extracts angular distributions (tabulated cosine PDFs) and energy distributions
   (level scattering, evaporation, Maxwellian, Kalbach-Mann, N-body, tabulated) per reaction channel.
4. For fissionable nuclides, extracts prompt/delayed ν(E), precursor fractions, decay constants, and energy spectra.
5. Writes a single HDF5 file per nuclide and header-temperature combination.

## Output HDF5 Schema

```
<Nuclide>-<Temperature>K.h5
├── attributes
│   ├── source_zaid
│   ├── source_suffix
│   └── source_temperature      (MeV)
├── nuclide_name              (string)
├── temperature               (float, K)
├── atomic_weight_ratio       (float)
├── fissionable               (bool)
└── neutron_reactions/
    ├── xs_energy_grid         (1-D array, MeV)
    ├── elastic_scattering/
    │   └── MT-002/
    │       ├── xs             (1-D array, barns)
    │       ├── cosine/
    │       └── energy/
    ├── capture/
    │   └── MT-102/ ...
    ├── inelastic_scattering/
    │   └── MT-051/ ...
    └── fission/
        └── MT-018/
            ├── xs
            ├── cosine/
            ├── energy/
            ├── nu_total/
            ├── nu_prompt/
            ├── nu_delayed/
            └── delayed_neutron/ ...
```

## See Also

- [Continuous Energy Theory Guide](../../../docs/source/theory/continuous_energy.rst)
- [Installation — CE Library Configuration](../../../docs/source/user_guide/getting_started/installation.rst)
