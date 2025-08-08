import importlib.metadata

import synthetic_bathymetry_inversion


def test_version():
    assert (
        importlib.metadata.version("synthetic_bathymetry_inversion")
        == synthetic_bathymetry_inversion.__version__
    )
