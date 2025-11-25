# Synthetic bathymetry inversion

This website documents the code and Jupyter notebooks included as part of the manuscript: ["Gravity Inversion for Sub-Ice Shelf Bathymetry: Strengths, Limitations, and Insights from Synthetic Modeling"](https://doi.org/10.5194/egusphere-2025-2380). The site is build from the following GitHub repository: [https://github.com/mdtanker/synthetic_bathymetry_inversion](https://github.com/mdtanker/synthetic_bathymetry_inversion)

While most of the code is from external packages, such as [Invert4Geom](https://github.com/mdtanker/invert4geom) for the inversion, there is some code supplied in `src/`, which can be viewed [here](api/synthetic_bathymetry_inversion).

All of the notebooks can be viewed at the left sidebar.

All the generated results are stored in the [Zenodo archive](https://doi.org/10.5281/zenodo.15614239).

<br/><br/>

# Instructions to re-run the notebooks

```{include} ../README.md
:start-after: SPHINX-START-badges -->
:end-before: <!-- SPHINX-END-badges
```

```{toctree}
:hidden:
index_inversions.md
```

```{toctree}
:hidden:
index_density_estimation_ensembles.md
```

```{toctree}
:hidden:
index_antarctica.md
```

```{toctree}
:hidden:
index_true_density_ensembles.md
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: 📖 API
api/synthetic_bathymetry_inversion
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: ℹ️ Other resources
changelog.md
Source code on GitHub <https://github.com/mdtanker/synthetic_bathymetry_inversion>
```
