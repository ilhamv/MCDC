"""Helpers for converting ACE continuous-energy neutron data to MC/DC HDF5.

The functions in this module translate ACEtk objects into the intermediate
HDF5 schema consumed by :mod:`mcdc.object_.nuclide` and
:mod:`mcdc.object_.neutron_reaction`.  Energies are stored in MeV, cross
sections in barns, and probability densities retain their ACE units unless a
function explicitly documents otherwise.

This module is a format converter, not a sampler.  It must therefore preserve
all information needed by MC/DC's runtime samplers, including interpolation
rules, discrete probability mass, reference frames, and correlated variables.
"""

from typing import Any

import ACEtk
import h5py
import numpy as np

from constant import ACE_TEMPERATURE_LIB81, Z_TO_SYMBOL


class DataLibraryGenerationError(RuntimeError):
    """Raised when ACE data cannot be represented by the MC/DC HDF5 schema."""


def decode_name(header: Any) -> tuple[str, str, int, int, int, float]:
    """Return MC/DC file and nuclide metadata decoded from an ACE header.

    Parameters
    ----------
    header
        ACEtk header object whose ``zaid`` includes the table suffix.

    Returns
    -------
    tuple
        ``(file_name, nuclide_name, Z, A, isomer, temperature_K)``.

    Notes
    -----
    The current implementation assumes the LANL ENDF/B-VIII.1 suffix-to-
    temperature mapping defined below.  It is not a general ACE-table decoder.
    """

    Z, A, S, T = decode_ace_name(header.zaid)
    symbol = Z_TO_SYMBOL[Z]
    nuclide_name = f"{symbol}{A}" if S == 0 else f"{symbol}{A}m{S}"
    mcdc_name = f"{nuclide_name}-{T}K.h5"
    return mcdc_name, nuclide_name, Z, A, S, T


def decode_ace_name(name: str) -> tuple[int, int, int, float]:
    """Decode a LANL ENDF/B-VIII.1 ACE table identifier.

    The current convention is assumed to be

    ``ZAID = 1000*Z + A`` for a ground state, and
    ``ZAID = 1000*Z + A + 300 + 100*S`` for isomer ``S >= 1``.

    Parameters
    ----------
    name
        ACE table identifier such as ``"14028.10c"``.

    Returns
    -------
    tuple[int, int, int, float]
        Atomic number, mass number, isomer index, and temperature in kelvin.

    Notes
    -----
    FIXME: The inverse isomer calculation below truncates mass numbers greater
    than 99.  Decode the identity from authoritative ACE metadata or implement
    the complete ZAID convention before supporting metastable tables.

    TODO: Read the table temperature from the ACE header.  A suffix identifies
    a library table and is not a universal temperature definition.
    """
    zaid, extension = name.split(".")

    zaid = int(zaid)
    Z = zaid // 1000
    remainder = zaid % 1000

    if remainder < 300:
        # ground state
        A = remainder
        S = 0
    else:
        # excited state
        offset = remainder - 300
        S = offset // 100
        A = offset % 100

    T = ACE_TEMPERATURE_LIB81[extension]

    return Z, A, S, T


def get_zaid(nuclide_name: str) -> tuple[int, int]:
    """Return ``(Z, A)`` parsed from a ground-state nuclide name.

    Examples include ``"Si28"`` and ``"U235"``.  Metastable suffixes are not
    currently supported.

    FIXME: ``Z_MAP`` is undefined; this helper is presently unusable.  Replace
    it with ``SYMBOL_TO_Z`` and add focused parsing tests before calling it.
    """

    nuclide_name = nuclide_name.strip().capitalize()

    # Find where the letters end and digits begin
    symbol = ""
    mass = 0
    for i, ch in enumerate(nuclide_name):
        if ch.isdigit():
            symbol = nuclide_name[:i]
            mass = int(nuclide_name[i:])
            break
    else:
        raise ValueError(f"No mass number found in '{nuclide_name}'")

    if symbol not in Z_MAP.keys():
        raise ValueError(f"Unknown element symbol '{symbol}'")

    Z = Z_MAP[symbol]
    A = mass
    return Z, A


def get_ace_name(Z: int, A: int, T: float, S: int | None = None) -> str:
    """Construct an ACE table identifier from nuclide metadata.

    FIXME: ``ACE_EXTENSION_LIB81`` is undefined and the returned identifier
    omits the separator before the suffix.  Reconcile this helper with
    ``TEMPERATURE_TO_ACELIB81`` and test ground-state and metastable names.
    """

    ID = Z * 1000 + A
    if S is not None:
        ID += 300 + S * 100
    extension = ACE_EXTENSION_LIB81[T]
    return f"{ID}{extension}"


def extract_interpolation_data(
    interpolation_data: Any, tag: str
) -> tuple[list[str], Any]:
    """Convert ACE interpolation codes to MC/DC interpolation names.

    Parameters
    ----------
    interpolation_data
        ACEtk interpolation-region metadata.
    tag
        Human-readable context included in error messages.

    Returns
    -------
    tuple[list[str], sequence]
        Interpolation names and their one-based ACE region boundaries.
    """

    interpolations: list[str] = []
    for interpolation in interpolation_data.interpolants:
        if interpolation == 1:
            interpolations.append("histogram")
        elif interpolation == 2:
            interpolations.append("linear")
        elif interpolation == 3:
            interpolations.append("semilog-x")
        elif interpolation == 4:
            interpolations.append("semilog-y")
        elif interpolation == 5:
            interpolations.append("log")
        else:
            raise DataLibraryGenerationError(f"Unsupported interpolation type in {tag}")
    interpolation_boundaries = interpolation_data.boundaries[:]
    return interpolations, interpolation_boundaries


def load_fission_multiplicity(data: Any, h5_group: h5py.Group) -> None:
    """Write polynomial or tabulated fission multiplicity data to HDF5.

    Parameters
    ----------
    data
        ACEtk fission-multiplicity object.
    h5_group
        Destination group for the MC/DC multiplicity schema.

    Notes
    -----
    TODO: Preserve non-linear interpolation regions and support any valid ACE
    multiplicity/yield representations required by the LANL library.
    """

    # Polynomial
    if data.type == 1:
        h5_group.attrs["type"] = "polynomial"

        C = np.array(data.coefficients)
        dataset = h5_group.create_dataset("coefficient", data=C)
        dataset.attrs["unit-base"] = "MeV"

    # Tabulated
    elif data.type == 2:
        h5_group.attrs["type"] = "tabulated"

        if not data.interpolation_data.is_linear_linear:
            raise DataLibraryGenerationError(
                "Non linear-linear tabulated multiplicity is not supported"
            )

        energy = np.array(data.energies)

        h5_group.create_dataset("value", data=data.multiplicities)
        dataset = h5_group.create_dataset("energy", data=energy)
        dataset.attrs["unit"] = "MeV"

    # Yield - unsupported
    else:
        raise DataLibraryGenerationError(f"Unsupported multiplicity type: {data.type}")


def load_cosine_distribution(data: Any, h5_group: h5py.Group) -> None:
    """Write an ACE angular-cosine distribution to HDF5.

    Fully isotropic and energy-correlated angular laws are represented by a
    type attribute.  Explicit angular data are flattened into incident-energy,
    offset, cosine, and PDF arrays.

    TODO: Preserve per-table interpolation codes and support ACE equiprobable
    angular bins.  The current path accepts only tabulated, linearly
    interpolated cosine distributions.
    """

    if isinstance(data, ACEtk.continuous.FullyIsotropicDistribution):
        h5_group.attrs["type"] = "isotropic"

    elif isinstance(data, ACEtk.continuous.DistributionGivenElsewhere):
        h5_group.attrs["type"] = "energy-correlated"

    else:
        h5_group.attrs["type"] = "tabulated"

        # Check distribution support: all tabulated
        NE = data.number_incident_energies
        for i in range(NE):
            idx = i + 1
            if data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated:
                raise DataLibraryGenerationError(
                    "Angular distribution is not all-tabulated"
                )

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated distributions
        interpolation = np.zeros(NE, dtype=int)
        offset = np.zeros(NE, dtype=int)
        cosine = []
        pdf = []
        for i, distribution in enumerate(data.distributions):
            interpolation[i] = distribution.interpolation
            offset[i] = len(cosine)
            cosine.extend(distribution.cosines)
            pdf.extend(distribution.pdf)
        cosine = np.array(cosine)
        pdf = np.array(pdf)
        h5_group.create_dataset("offset", data=offset)
        h5_group.create_dataset("value", data=cosine)
        h5_group.create_dataset("pdf", data=pdf)

        if not all(interpolation == 2):
            raise DataLibraryGenerationError(
                "Angular distribution is not linearly-iterpolable"
            )


def load_energy_distribution(data: Any, h5_group: h5py.Group) -> None:
    """Write one ACE outgoing-energy law to the MC/DC HDF5 schema.

    Supported branches currently include level scattering, evaporation,
    Maxwellian, tabulated energy, Kalbach--Mann, correlated energy--angle, and
    a reduced N-body representation.  Unsupported ACE laws terminate the
    generator by raising :class:`DataLibraryGenerationError`.

    Parameters
    ----------
    data
        ACEtk outgoing-energy distribution object.
    h5_group
        Destination group beneath an MC/DC reaction.

    Notes
    -----
    TODO: Add an explicit capability table and support the remaining ACE laws
    used by the LANL library, including Watt and other transfer laws.
    """

    if isinstance(data, ACEtk.continuous.LevelScatteringDistribution):
        h5_group.attrs["type"] = "level-scattering"

        C1 = np.array(data.C1)
        C1 = h5_group.create_dataset("C1", data=C1)
        C1.attrs["unit"] = "MeV"

        h5_group.create_dataset("C2", data=data.C2)

    elif isinstance(data, ACEtk.continuous.EvaporationSpectrum):
        h5_group.attrs["type"] = "evaporation"

        interpolations, interpolation_boundaries = extract_interpolation_data(
            data.interpolation_data, "Evaporation spectrum temperature"
        )

        energy = np.array(data.energies)
        temperature = np.array(data.temperatures)
        restriction_energy = np.array(data.restriction_energy)

        h5_group.create_dataset("temperature_interpolations", data=interpolations)
        h5_group.create_dataset(
            "interpolation_boundaries", data=interpolation_boundaries
        )
        dataset = h5_group.create_dataset("temperature_energy_grid", data=energy)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("temperature", data=temperature)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("restriction_energy", data=restriction_energy)
        dataset.attrs["unit"] = "MeV"

    elif isinstance(data, ACEtk.continuous.SimpleMaxwellianFissionSpectrum):
        h5_group.attrs["type"] = "maxwellian"

        interpolations, interpolation_boundaries = extract_interpolation_data(
            data.interpolation_data, "Maxwellian spectrum temperature"
        )

        energy = np.array(data.energies)
        temperature = np.array(data.temperatures)
        restriction_energy = np.array(data.restriction_energy)

        # FIXME: The runtime loader expects "temperature_interpolations"
        # (plural).  Rename this dataset together with a schema regression test.
        h5_group.create_dataset("temperature_interpolation", data=interpolations)
        h5_group.create_dataset(
            "interpolation_boundaries", data=interpolation_boundaries
        )
        dataset = h5_group.create_dataset("temperature_energy_grid", data=energy)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("temperature", data=temperature)
        dataset.attrs["unit"] = "MeV"
        dataset = h5_group.create_dataset("restriction_energy", data=restriction_energy)
        dataset.attrs["unit"] = "MeV"

    elif isinstance(data, ACEtk.continuous.OutgoingEnergyDistributionData):
        h5_group.attrs["type"] = "tabulated"

        # This checks interpolation between incident-energy tables only.  Each
        # outgoing-energy table also carries its own interpolation code and can
        # begin with discrete lines.
        #
        # FIXME: Preserve each inner interpolation code, discrete probability
        # mass, and ACE CDF.  The current HDF5 consumer reconstructs every table
        # as a continuous, piecewise-linear PDF, which changes valid ACE data.
        if not data.interpolation_data.is_linear_linear:
            raise DataLibraryGenerationError(
                "Non-linearly-interpolated energy distribution is not supported"
            )

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated disstributions
        NE = data.number_incident_energies
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        for i in range(NE):
            distribution = data.distribution(i + 1)
            offset[i] = len(energy_out)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("value", data=energy_out)
        dataset.attrs["unit"] = ["MeV"]
        h5_group.create_dataset("pdf", data=pdf)

    elif isinstance(data, ACEtk.continuous.KalbachMannDistributionData):
        h5_group.attrs["type"] = "kalbach-mann"

        # TODO: Preserve and validate the interpolation type of every outgoing-
        # energy table, not only interpolation across incident energy.
        if not data.interpolation_data.is_linear_linear:
            raise DataLibraryGenerationError(
                "Non-linearly-interpolated kalbach-mann is not supported"
            )

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies)
        energy = h5_group.create_dataset("energy", data=energy)
        energy.attrs["unit"] = "MeV"

        # Tabulated distributions
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        precompound_factor = []
        angular_slope = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)
            precompound_factor.extend(distribution.precompound_fraction_values)
            angular_slope.extend(distribution.angular_distribution_slope_values)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)
        precompound_factor = np.array(precompound_factor)
        angular_slope = np.array(angular_slope)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("energy_out", data=energy_out)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=pdf)
        h5_group.create_dataset("precompound_factor", data=precompound_factor)
        h5_group.create_dataset("angular_slope", data=angular_slope)

    elif isinstance(data, ACEtk.continuous.EnergyAngleDistributionData):
        h5_group.attrs["type"] = "energy-angle-tabulated"

        # TODO: Preserve interpolation metadata and ACE CDFs for both the
        # outgoing-energy tables and their conditional angular distributions.
        if not data.interpolation_data.is_linear_linear:
            raise DataLibraryGenerationError(
                "Non-linearly-interpolated correlated-energy-angle is not supported"
            )

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies)
        dataset = h5_group.create_dataset("energy", data=energy)
        dataset.attrs["unit"] = "MeV"

        # Tabulated distributions
        offset = np.zeros(NE, dtype=int)
        energy_out = []
        pdf = []
        cosine_offset = []
        cosine = []
        cosine_pdf = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)

            for inner_distribution in distribution.distributions:
                cosine_offset.append(len(cosine_pdf))
                cosine.extend(inner_distribution.cosines)
                cosine_pdf.extend(inner_distribution.pdf)

        energy_out = np.array(energy_out)
        pdf = np.array(pdf)
        cosine_offset = np.array(cosine_offset)
        cosine = np.array(cosine)
        cosine_pdf = np.array(cosine_pdf)

        h5_group.create_dataset("offset", data=offset)
        dataset = h5_group.create_dataset("energy_out", data=energy_out)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=pdf)
        h5_group.create_dataset("cosine_offset", data=cosine_offset)
        h5_group.create_dataset("cosine", data=cosine)
        h5_group.create_dataset("cosine_pdf", data=cosine_pdf)

    elif isinstance(data, ACEtk.continuous.NBodyPhaseSpaceDistribution):
        h5_group.attrs["type"] = "N-body"

        if data.interpolation != 2:
            raise DataLibraryGenerationError(
                "Non-linearly-interpolable N-body energy distribution"
            )

        # FIXME: ACE LAW 66 values are normalized phase-space coordinates, not
        # energies in MeV.  Preserve number_particles and total_mass_ratio, then
        # apply the incident-energy-dependent phase-space kinematics at runtime.
        # The current reduced representation cannot reproduce LAW 66.
        dataset = h5_group.create_dataset("value", data=data.values)
        dataset.attrs["unit"] = "MeV"
        h5_group.create_dataset("pdf", data=data.pdf)

    else:
        raise DataLibraryGenerationError(f"Unsupported energy distribution: {data}")
