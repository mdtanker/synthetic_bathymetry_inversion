# Inversion ensembles (1-4)
To investigate the performance of gravity inversions for bathymetry under a range of scenarios, we perform a series of inversions ensembles, covering a 2D parameter space, to see how the range of parameters affect the inversion results, and how sets of parameters influence each other.

## **Ensemble 1: Constraint spacings vs. Regional field strength**
This ensemble tests all combinations of 10 values of constraint (bathymetry point measurement) spacing, and 10 values of the strength (standard deviation) of the regional component of the gravity field. For each of these 100 inversions, we estimate the optimal density contrast value for the seafloor with a cross-validation of the constraint points, but use a set damping parameter value to limit the computationally expense damping cross-validation routine.

```{nbgallery}
---
---
ensemble_01_constraint_spacing_vs_regional_strength_density_estimation
```

## **Ensembles 2-4: Gravity flight line spacing vs. gravity noise**
These ensembles test all combinations of 10 values of gravity flight line spacing, and 10 values of the gravity data noise. For each inversion, we estimate the optimal density contrast value for the seafloor with a cross-validation of the constraint points. Ensemble 2 uses no regional gravity field, Ensemble 3 uses a medium-strength regional gravity field, and Ensemble 4 uses a strong regional gravity field.

```{nbgallery}
---
---
ensemble_02_grav_spacing_vs_noise_no_regional_density_estimation
ensemble_03_grav_spacing_vs_noise_medium_regional_density_estimation
ensemble_04_grav_spacing_vs_noise_strong_regional_density_estimation

```

## **Analysis and manuscript figures**
This notebooks loads the results of the Ensembles and creates the figures used in the manuscript.

```{nbgallery}
---
---
ensemble_figures
```
