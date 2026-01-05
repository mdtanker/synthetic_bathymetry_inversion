# Antarctic ice shelf analysis

This notebook iterates through all 164 Antarctic ice shelf and performs the following steps:
## 1) Extract all point measurements of bathymetry/bed topography
- within 20 km of the ice shelf boundary
- point measurements from IBCSO v2
- point measurements for Bedmachine source data from simglebeam, multibeam, over ice seismic, over ice and airborne radar, and REMA elevations for exposed rock
- IBCSO v2 swath bathymetry polygons are converted to point measurements with 100 m spacing
- shown as black dots in subplot **a** below
## 2) Compute grid of distances to the nearest point measurement
- use the above compiled point measurements
- calculate the median of the grid value within the outline of the ice shelf
- subplot **a** below
## 3) Calculate topography-corrected gravity disturbances
- extract gridded gravity disturbances from AntGG-2021 gravity compilation (subplot **c**)
- correct it for gravity effects of ice, water, and topography to get the topography-corrected gravity disturbances (subplot **e**)
    - use Bedmap2 grids of ice surface, ice base, and bed topography for calculation
- calculate the standard-deviation of the topography-corrected gravity disturbance with the outline of the ice shelf

An example of the analysis applied to the Fimbul Ice shelf:
```{image} Fimbul_info.png
:alt: fimbul
:class: bg-primary
:width: 1000px
:align: center
```

The results for each ice shelf, and plots of the results (as above), are all provided in the [Zenodo archive](https://doi.org/10.5281/zenodo.15614239):
  - constraint points: `results/ice_shelves/<ice-shelf-name>_constraints.csv.gz`
  - gravity anomalies: `results/ice_shelves/<ice-shelf-name>_grav_anomalies.csv.gz`
  - grids of distance to nearest bed measurements: `results/ice_shelves/<ice-shelf-name>_min_dist.nc`
  - plots of constraint points and distance to nearest bed measurements: `results/ice_shelves/<ice-shelf-name>_constraints.png`
  - Subplots of distance to nearest bed measurements, gravity error, gravity disturbance, starting gravity, gravity misfit, regional and residual misfit: `results/ice_shelves/<ice-shelf-name>info.png`

A table of the summarized results for all ice shelves are provided in `results/ice_shelves/ice_shelf_gravity_stats.csv`.

## Notebooks

```{nbgallery}
---
---
antarctic_ice_shelf_calculations
```

This notebook takes the results of the above notebook, and create the figures for the manuscript as well as some additional analysis.

```{nbgallery}
---
---
antarctic_ice_shelf_analysis
```
