import importlib.util
import inspect
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture
def neutron_generator_util(monkeypatch):
    """Load the generator helpers without requiring the optional ACEtk package."""

    acetk = types.ModuleType("ACEtk")
    acetk.ContinuousEnergyTable = object
    monkeypatch.setitem(sys.modules, "ACEtk", acetk)
    path = (
        Path(__file__).parents[2]
        / "tools"
        / "data_library_generator"
        / "neutron"
        / "util.py"
    )
    monkeypatch.syspath_prepend(str(path.parent.parent))
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("neutron_generator_util", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_unsupported_interpolation_raises_generation_error(neutron_generator_util):
    interpolation_data = types.SimpleNamespace(interpolants=[999], boundaries=[])

    with pytest.raises(
        neutron_generator_util.DataLibraryGenerationError,
        match="Unsupported interpolation type in test data",
    ):
        neutron_generator_util.extract_interpolation_data(
            interpolation_data, "test data"
        )


def test_unsupported_multiplicity_raises_generation_error(neutron_generator_util):
    data = types.SimpleNamespace(type=999)

    with pytest.raises(
        neutron_generator_util.DataLibraryGenerationError,
        match="Unsupported multiplicity type: 999",
    ):
        neutron_generator_util.load_fission_multiplicity(data, object())


def test_decode_name_uses_header_temperature_and_retains_suffix(
    neutron_generator_util,
):
    header = types.SimpleNamespace(zaid="14028.99c", temperature=2.53e-8)
    ace_table = types.SimpleNamespace(
        header=header,
        atom_number=14,
        mass_number=28,
        isomeric_state=0,
    )

    file_name, nuclide_name, Z, A, S, temperature, suffix = (
        neutron_generator_util.decode_name(ace_table)
    )

    assert file_name == "Si28-293.59K.h5"
    assert nuclide_name == "Si28"
    assert (Z, A, S) == (14, 28, 0)
    assert temperature == pytest.approx(293.5943085)
    assert suffix == "99c"


def test_decode_name_uses_authoritative_metastable_identity(neutron_generator_util):
    header = types.SimpleNamespace(zaid="95642.99c", temperature=2.53e-8)
    ace_table = types.SimpleNamespace(
        header=header,
        atom_number=95,
        mass_number=242,
        isomeric_state=1,
    )

    file_name, nuclide_name, Z, A, S, _, suffix = neutron_generator_util.decode_name(
        ace_table
    )

    assert file_name == "Am242m1-293.59K.h5"
    assert nuclide_name == "Am242m1"
    assert (Z, A, S) == (95, 242, 1)
    assert suffix == "99c"


def test_get_ace_name_uses_explicit_suffix(neutron_generator_util):
    assert neutron_generator_util.get_ace_name(14, 28, "99c") == "14028.99c"


def test_shared_element_symbol_mappings(neutron_generator_util):
    assert neutron_generator_util.Z_FROM_SYMBOL["Si"] == 14
    assert neutron_generator_util.SYMBOL_FROM_Z[14] == "Si"
    assert neutron_generator_util.get_zaid("Si28") == (14, 28)


def test_public_helpers_have_complete_type_annotations(neutron_generator_util):
    helper_names = (
        "decode_name",
        "get_zaid",
        "get_ace_name",
        "extract_interpolation_data",
        "load_fission_multiplicity",
        "load_cosine_distribution",
        "load_energy_distribution",
    )

    for name in helper_names:
        signature = inspect.signature(getattr(neutron_generator_util, name))
        assert all(
            parameter.annotation is not inspect.Parameter.empty
            for parameter in signature.parameters.values()
        )
        assert signature.return_annotation is not inspect.Signature.empty
