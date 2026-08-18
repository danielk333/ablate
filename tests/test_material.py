import pytest

from metablate import material
from metablate.types import Material


def test_material_presets_are_available_and_physical():
    for name in material.MATERIALS:
        mat = material.get(name, as_dict=False)
        assert isinstance(mat, Material)


def test_material_get_accepts_case_and_rejects_unknown():
    assert material.get("COMETARY", as_dict=False).bulk_density == material.cometary.bulk_density

    with pytest.raises(ValueError):
        material.get("not-a-material")
