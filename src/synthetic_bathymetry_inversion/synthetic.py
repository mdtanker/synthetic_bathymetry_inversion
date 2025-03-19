from __future__ import annotations

import math
import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from invert4geom import synthetic as inv_synthetic
from invert4geom import uncertainty
from invert4geom import utils as inv_utils
from polartoolkit import fetch, maps, profiles, utils
from shapely.geometry import LineString
import shapely
import scipy

from synthetic_bathymetry_inversion import logger

def load_synthetic_model(
    spacing: float = 1e3,
    inversion_region: tuple[float, float, float, float] = (
        -40e3,
        260e3,
        -1800e3,
        -1400e3,
    ),
    buffer: float = 0,
    zref: float = 0,
    bathymetry_density_contrast: float = 1476,
    basement_density_contrast: float = 100,
    basement: bool = False,
    gravity_noise: float | None = None,
    gravity_noise_wavelength: float = 50e3,
    plot_topography: bool = True,
    plot_gravity: bool = True,
    just_topography: bool = False,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    Function to perform all necessary steps to create a synthetic model for the examples
    in the documentation.

    Parameters
    ----------
    spacing : float, optional
        spacing of the grid and gravity, by default 1e3
    buffer : float, optional
        buffer to add around the region, by default 0. Buffer region used for creating
        topography and prisms, while inner region used for extent of gravity and
        constraints.
    zref : float , optional
        reference level to use, by default 0
    bathymetry_density_contrast : float, optional
        density contrast between bathymetry and water, by default 1476, (2500 - 1024)
    basement_density_contrast : float, optional
        density contrast between basement and water, by default 100
    basement : bool, optional
        set to True to include a basement model for the regional gravity field, by
        default False
    gravity_noise : float | None, optional
        decimal percentage noise level to add to gravity data, by default None
    gravity_noise_wavelength : float, optional
        wavelength of noise in km to add to gravity data, by default 50e3
    plot_topography : bool, optional
        plot the topography, by default False
    plot_gravity : bool, optional
        plot the gravity data, by default True
    just_topography : bool, optional
        return only the topography, by default False

    Returns
    -------
    true_topography : xarray.DataArray
        the true topography
    grav_df : pandas.DataFrame
        the gravity data
    """
    # inversion_region = (-40e3, 260e3, -1800e3, -1400e3)
    registration = "g"

    buffer_region = (
        vd.pad_region(inversion_region, buffer) if buffer != 0 else inversion_region
    )

    # get Ross Sea bathymetry and basement data
    bathymetry_grid = fetch.ibcso(
        layer="bed",
        reference="ellipsoid",
        region=buffer_region,
        spacing=spacing,
        registration=registration,
    ).rename({"x": "easting", "y": "northing"})

    if just_topography is True:
        return bathymetry_grid, None, None

    if basement is True:
        sediment_thickness = fetch.sediment_thickness(
            version="lindeque-2016",
            region=buffer_region,
            spacing=spacing,
            registration=registration,
        ).rename({"x": "easting", "y": "northing"})
        basement_grid = bathymetry_grid - sediment_thickness
        basement_grid = xr.where(
            basement_grid > bathymetry_grid, bathymetry_grid, basement_grid
        )
    else:
        basement_grid = None

    if plot_topography is True:
        fig = maps.plot_grd(
            bathymetry_grid,
            show_region=inversion_region,
            fig_height=10,
            title="Bathymetry",
            hist=True,
            cbar_yoffset=1,
            cmap="rain",
            reverse_cpt=True,
            cbar_label="elevation (m)",
            robust=True,
        )

        if basement is True:
            fig = maps.plot_grd(
                basement_grid,
                show_region=inversion_region,
                fig_height=10,
                title="Basement",
                hist=True,
                cbar_yoffset=1,
                cmap="rain",
                reverse_cpt=True,
                cbar_label="elevation (m)",
                fig=fig,
                origin_shift="xshift",
                robust=True,
                scalebar=True,
                scale_position="n-.05/-.03",
            )

        fig.show()

    # calculate forward gravity effects

    # bathymetry
    density_grid = xr.where(
        bathymetry_grid >= zref,
        bathymetry_density_contrast,
        -bathymetry_density_contrast,
    )
    bathymetry_prisms = inv_utils.grids_to_prisms(
        bathymetry_grid,
        zref,
        density=density_grid,
    )

    # basement
    if basement is True:
        density_grid = xr.where(
            basement_grid >= zref,
            basement_density_contrast,
            -basement_density_contrast,
        )
        basement_prisms = inv_utils.grids_to_prisms(
            basement_grid,
            zref,
            density=density_grid,
        )

    # make pandas dataframe of locations to calculate gravity
    # this represents the station locations of a gravity survey
    # create lists of coordinates
    coords = vd.grid_coordinates(
        region=inversion_region,
        spacing=spacing,
        pixel_register=False,
        extra_coords=1000,  # survey elevation
    )

    # grid the coordinates
    observations = vd.make_xarray_grid(
        (coords[0], coords[1]),
        data=coords[2],
        data_names="upward",
        dims=("northing", "easting"),
    ).upward

    grav_df = vd.grid_to_table(observations)

    grav_df["bathymetry_grav"] = bathymetry_prisms.prism_layer.gravity(
        coordinates=(
            grav_df.easting,
            grav_df.northing,
            grav_df.upward,
        ),
        field="g_z",
        progressbar=True,
    )

    if basement is True:
        grav_df["basement_grav"] = basement_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=True,
        )
        grav_df["basement_grav"] = grav_df.basement_grav - grav_df.basement_grav.mean()
    else:
        grav_df["basement_grav"] = 0

    # add forward gravity fields together to get the observed gravity
    grav_df["disturbance"] = grav_df.bathymetry_grav + grav_df.basement_grav

    # contaminate gravity with user-defined wavelength random noise
    if gravity_noise is not None:
        # long-wavelength noise
        grav_df["noise_free_disturbance"] = grav_df.disturbance
        cont = inv_synthetic.contaminate_with_long_wavelength_noise(
            grav_df.set_index(["northing", "easting"]).to_xarray().disturbance,
            coarsen_factor=None,
            spacing=gravity_noise_wavelength,
            noise_as_percent=False,
            noise=gravity_noise,
        )
        df = vd.grid_to_table(cont.rename("disturbance")).reset_index(drop=True)
        grav_df = pd.merge(  # noqa: PD015
            grav_df.drop(columns=["disturbance"], errors="ignore"),
            df,
            on=["easting", "northing"],
        )

        # short-wavelength noise
        # cont = inv_synthetic.contaminate_with_long_wavelength_noise(
        #     grav_df.set_index(["northing", "easting"]).to_xarray().disturbance,
        #     coarsen_factor=None,
        #     spacing=spacing, # * 2,
        #     noise_as_percent=False,
        #     noise=gravity_noise,
        # )
        # df = vd.grid_to_table(cont.rename("disturbance")).reset_index(drop=True)
        # grav_df = pd.merge(  # noqa: PD015
        #     grav_df.drop(columns=["disturbance"], errors="ignore"),
        #     df,
        #     on=["easting", "northing"],
        # )

        grav_df["uncert"] = gravity_noise

    grav_df["gravity_anomaly"] = grav_df.disturbance

    if plot_gravity is True:
        grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()

        fig = maps.plot_grd(
            grav_grid.bathymetry_grav,
            region=inversion_region,
            fig_height=10,
            title="Bathymetry gravity",
            hist=True,
            cbar_yoffset=1,
            cbar_label="mGal",
            robust=True,
        )

        if basement is True:
            fig = maps.plot_grd(
                grav_grid.basement_grav,
                region=inversion_region,
                fig_height=10,
                title="Basement gravity",
                hist=True,
                cbar_yoffset=1,
                cbar_label="mGal",
                fig=fig,
                origin_shift="xshift",
                robust=True,
            )

        fig = maps.plot_grd(
            grav_grid.disturbance,
            region=inversion_region,
            fig_height=10,
            title="Combined gravity",
            hist=True,
            cbar_yoffset=1,
            cbar_label="mGal",
            fig=fig,
            origin_shift="xshift",
            robust=True,
        )

        fig.show()

        if gravity_noise is not None:
            _ = utils.grd_compare(
                grav_grid.noise_free_disturbance,
                grav_grid.disturbance,
                fig_height=10,
                plot=True,
                grid1_name="Gravity",
                grid2_name=f"with {gravity_noise} mGal noise",
                title="Difference",
                title_font="18p,Helvetica-Bold,black",
                cbar_unit="mGal",
                cbar_label="gravity",
                # RMSE_decimals=0,
                region=inversion_region,
                inset=False,
                hist=True,
                cbar_yoffset=1,
                label_font="16p,Helvetica,black",
            )
    return bathymetry_grid, basement_grid, grav_df


def constraint_layout_number(
    num_constraints=None,
    latin_hypercube=False,
    shape=None,
    spacing=None,
    shift_stdev=0,
    region=None,
    shapefile=None,
    padding=None,
    add_outside_points=False,
    grid_spacing=None,
    plot=False,
    seed=0,
):
    full_region = region

    if shapefile is not None:
        bounds = gpd.read_file(shapefile).bounds
        region = [bounds.minx, bounds.maxx, bounds.miny, bounds.maxy]
        region = [x.values[0] for x in region]

    x = region[1] - region[0]
    y = region[3] - region[2]

    if (shape is None) and (num_constraints is None) and (spacing is None):
        msg = "must provide either shape, num_constraints, or spacing"
        raise ValueError(msg)

    if padding is not None:
        region = vd.pad_region(region, padding)

    width = region[1] - region[0]
    height = region[3] - region[2]

    if num_constraints == 0:
        constraints = pd.DataFrame(columns=["easting", "northing", "upward", "inside"])
    elif latin_hypercube:
        if num_constraints is None:
            msg = "need to set number of constraints if using latin hypercube"
            raise ValueError(msg)
        coord_dict = {
            "easting": {
                "distribution": "uniform",
                "loc": region[0],  # lower bound
                "scale": width,  # range
            },
            "northing": {
                "distribution": "uniform",
                "loc": region[2],  # lower bound
                "scale": height,  # range
            },
        }
        sampled_coord_dict = uncertainty.create_lhc(
            n_samples=num_constraints,
            parameter_dict=coord_dict,
            criterion="maximin",
        )
        constraints = pd.DataFrame(
            {
                "easting": sampled_coord_dict["easting"]["sampled_values"],
                "northing": sampled_coord_dict["northing"]["sampled_values"],
                "upward": np.ones_like(sampled_coord_dict["northing"]["sampled_values"])
                * 1e3,
            }
        )

    else:
        fudge_factor = 0
        while True:
            if num_constraints is not None:
                num_y = int(np.ceil((num_constraints / (x / y)) ** 0.5))
                num_x = int(np.ceil(num_constraints / num_y)) + fudge_factor
            elif shape is not None:
                num_x = shape[0]
                num_y = shape[1]
            else:
                num_x = None
                num_y = None
            if spacing is not None:
                pad = (0, 0)
            else:
                # if (num_x % 2 == 0) and (num_y % 2 == 0):
                pad = (-height / (num_y) / 2, -width / (num_x) / 2)
                # elif num_x % 2 == 0:
                #     pad = (0, -width/(num_x)/2)
                # elif num_y % 2 == 0:
                #     pad = (-height/(num_y)/2, 0)
                # else:
                #     pad = (0, 0)

            reg = vd.pad_region(region, pad)

            if spacing is not None:
                x = np.arange(reg[0], reg[1], spacing[0])
                y = np.arange(reg[2], reg[3], spacing[1])

                # center of region
                x_reg_mid = (reg[1] - reg[0]) / 2
                y_reg_mid = (reg[3] - reg[2]) / 2

                # center of arrays
                x_mid = (x[-1] - x[0]) / 2
                y_mid = (y[-1] - y[0]) / 2

                # shift to be centered
                xshift = x_reg_mid - x_mid
                yshift = y_reg_mid - y_mid
                x += xshift
                y += yshift
            else:
                if num_x == 1:
                    x = [(reg[1] + reg[0]) / 2]
                else:
                    x = np.linspace(reg[0], reg[1], num_x)
                if num_y == 1:
                    y = [(reg[3] + reg[2]) / 2]
                else:
                    y = np.linspace(reg[2], reg[3], num_y)

            # if len(x) == 1:
            coords = np.meshgrid(x, y)

            # turn coordinates into dataarray
            da = vd.make_xarray_grid(
                coords,
                data=np.ones_like(coords[0]) * 1e3,
                data_names="upward",
                dims=("northing", "easting"),
            )
            # turn dataarray into dataframe
            df = vd.grid_to_table(da)

            # add randomness to the points
            rand = np.random.default_rng(seed=seed)
            constraints = df.copy()
            constraints["northing"] = rand.normal(df.northing, shift_stdev)
            constraints["easting"] = rand.normal(df.easting, shift_stdev)

            # keep only set number of constraints
            if shape is not None or spacing is not None:
                break
            try:
                constraints = constraints.sample(n=num_constraints, random_state=seed)
            except ValueError:
                fudge_factor += 0.1
            else:
                break

    # check whether points are inside or outside of shp
    if shapefile is not None:
        gdf = gpd.GeoDataFrame(
            constraints,
            geometry=gpd.points_from_xy(x=constraints.easting, y=constraints.northing),
            crs="EPSG:3031",
        )
        constraints["inside"] = gdf.within(gpd.read_file(shapefile).geometry[0])
        try:  # noqa: SIM105
            constraints = constraints.drop(columns="geometry")
        except KeyError:
            pass
        # drop outside constraints
        constraints = constraints[constraints.inside]

    # ensure all points are within region
    constraints = utils.points_inside_region(
        constraints, region, names=("easting", "northing")
    )

    constraints = constraints.drop(columns="upward")

    if add_outside_points:
        constraints["inside"] = True

        # make empty grid
        coords = vd.grid_coordinates(
            region=full_region,
            spacing=grid_spacing,
            pixel_register=False,
        )
        grd = vd.make_xarray_grid(
            coords,
            data=np.ones_like(coords[0]) * 1e3,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward
        # mask to shapefile
        masked = utils.mask_from_shp(
            shapefile=shapefile,
            xr_grid=grd,
            masked=True,
        ).rename("upward")
        outside_constraints = vd.grid_to_table(masked).dropna()
        outside_constraints = outside_constraints.drop(columns="upward")
        outside_constraints["inside"] = False

        constraints = pd.concat([outside_constraints, constraints], ignore_index=True)

    if plot:
        fig = maps.basemap(
            fig_height=8,
            region=full_region,
            frame=True,
        )

        fig.plot(
            x=constraints.easting,
            y=constraints.northing,
            style="c2p",
            fill="black",
        )

        if shapefile is not None:
            fig.plot(
                shapefile,
                pen="0.2p,black",
            )

        fig.show()

    return constraints


def constraint_layout_spacing(
    spacing,
    shift_stdev=0,
    region=None,
    shapefile=None,
    padding=None,
    plot=False,
):
    if shapefile is not None:
        bounds = gpd.read_file(shapefile).bounds
        region = [bounds.minx, bounds.maxx, bounds.miny, bounds.maxy]
        region = [x.values[0] for x in region]

    # create regular grid, with set number of constraint points
    reg = vd.pad_region(region, padding) if padding is not None else region

    # start grid from edges
    # x = np.arange(reg[0], reg[1], spacing)
    # y = np.arange(reg[2], reg[3], spacing)

    # start grid from center
    x_mid = reg[0] + (reg[1] - reg[0]) / 2
    y_mid = reg[2] + (reg[3] - reg[2]) / 2
    x1 = np.arange(x_mid, reg[0], -spacing)
    x2 = np.arange(x_mid + spacing, reg[1], spacing)
    y1 = np.arange(y_mid, reg[2], -spacing)
    y2 = np.arange(y_mid + spacing, reg[3], spacing)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    coords = np.meshgrid(x, y)

    # turn coordinates into dataarray
    da = vd.make_xarray_grid(
        coords,
        data=np.ones_like(coords[0]) * 1e3,
        data_names="upward",
        dims=("northing", "easting"),
    )
    # turn dataarray into dataframe
    df = vd.grid_to_table(da)

    # add randomness to the points
    rand = np.random.default_rng(seed=0)
    constraints = df.copy()
    constraints["northing"] = rand.normal(df.northing, shift_stdev)
    constraints["easting"] = rand.normal(df.easting, shift_stdev)

    # check whether points are inside or outside of shp
    if shapefile is not None:
        gdf = gpd.GeoDataFrame(
            constraints,
            geometry=gpd.points_from_xy(x=constraints.easting, y=constraints.northing),
            crs="EPSG:3031",
        )
        constraints["inside"] = gdf.within(gpd.read_file(shapefile).geometry[0])
        constraints = constraints.drop(columns="geometry")
    else:
        constraints["inside"] = True

    # drop outside constraints
    constraints = constraints[constraints.inside]

    # ensure all points are within region
    constraints = utils.points_inside_region(
        constraints, region, names=("easting", "northing")
    )

    if plot:
        fig = maps.basemap(
            fig_height=8,
            region=region,
            frame=True,
        )

        fig.plot(
            x=constraints.easting,
            y=constraints.northing,
            style="c.1c",
            fill="black",
        )

        if shapefile is not None:
            fig.plot(
                shapefile,
                pen="0.2p,black",
            )

        fig.show()

    return constraints


def airborne_survey(
    along_line_spacing: float,
    grav_observation_height: float,
    region: tuple[float, float, float, float],
    ns_line_spacing: float | None = None,
    ew_line_spacing: float | None = None,
    ns_line_number: float | None = None,
    ew_line_number: float | None = None,
    padding: float | None = None,
    ns_lines_to_remove: list[int] | None = None,
    ew_lines_to_remove: list[int] | None = None,
    grav_grid: xr.DataArray | None = None,
    plot: bool = False,
):
    if padding is not None:
        region = vd.pad_region(region, padding)

    width = region[1] - region[0]
    height = region[3] - region[2]

    # center of region
    x_reg_mid = (region[1] - region[0]) / 2
    y_reg_mid = (region[3] - region[2]) / 2

    # simulate N-S tie lines
    if ns_line_spacing is not None:
        x = np.arange(region[0], region[1], ns_line_spacing)
        # center of arrays
        x_mid = (x[-1] - x[0]) / 2
        # shift to be centered
        xshift = x_reg_mid - x_mid
        x += xshift
    elif ns_line_number is not None:
        if ns_line_number == 1:
            x = [(region[1] + region[0]) / 2]
        else:
            pad = (0, -width / (ns_line_number) / 2)
            reg = vd.pad_region(region, pad)
            x = np.linspace(reg[0], reg[1], ns_line_number)

    y = np.arange(region[2], region[3], along_line_spacing)

    # remove select N-S lines,starting from left
    if ns_lines_to_remove is not None:
        x = np.delete(x, ns_lines_to_remove)

    # calculate median spacing
    # NS_points = [[0, i] for i in x]
    # NS_median_spacing = (
    #     np.median(
    #         vd.median_distance(
    #             NS_points,
    #             k_nearest=1,
    #         )
    #     )
    #     / 1e3
    # )
    coords = np.meshgrid(x, y)

    # turn coordinates into dataarray
    ties = vd.make_xarray_grid(
        coords,
        data=np.ones_like(coords[0]) * grav_observation_height,
        data_names="upward",
        dims=("northing", "easting"),
    )
    # turn dataarray into dataframe
    df_ties = vd.grid_to_table(ties)

    # give each tie line a number starting at 1000 in increments of 10
    df_ties["line"] = np.nan
    for i, j in enumerate(df_ties.easting.unique()):
        df_ties["line"] = np.where(df_ties.easting == j, 1000 + i * 10, df_ties.line)

    # simulate E-W flight line
    if ew_line_spacing is not None:
        y = np.arange(region[2], region[3], ew_line_spacing)
        # center of arrays
        y_mid = (y[-1] - y[0]) / 2
        # shift to be centered
        yshift = y_reg_mid - y_mid
        y += yshift
    elif ew_line_number is not None:
        if ew_line_number == 1:
            y = [(region[2] + region[3]) / 2]
        else:
            pad = (-height / (ew_line_number) / 2, 0)
            reg = vd.pad_region(region, pad)
            y = np.linspace(reg[2], reg[3], ew_line_number)

    x = np.arange(region[0], region[1], along_line_spacing)

    # remove select E-W lines, starting from bottom
    if ew_lines_to_remove is not None:
        y = np.delete(y, ew_lines_to_remove)

    coords = np.meshgrid(x, y)

    # turn coordinates into dataarray
    lines = vd.make_xarray_grid(
        coords,
        data=np.ones_like(coords[0]) * grav_observation_height,
        data_names="upward",
        dims=("northing", "easting"),
    )
    # turn dataarray into dataframe
    df_lines = vd.grid_to_table(lines)

    # give each line a number starting at 0 in increments of 10
    df_lines["line"] = np.nan
    for i, j in enumerate(df_lines.northing.unique()):
        df_lines["line"] = np.where(df_lines.northing == j, i + 1 * 10, df_lines.line)

    # merge dataframes
    df = pd.concat([df_ties, df_lines])

    # add a time column
    df["time"] = np.nan
    for i in df.line.unique():
        if i >= 1000:
            time = df[df.line == i].sort_values("northing").reset_index().index.values
        else:
            time = df[df.line == i].sort_values("easting").reset_index().index.values
        df.loc[df.line == i, "time"] = time

    # convert to geopandas
    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x=df.easting, y=df.northing),
        crs="EPSG:3031",
    )

    # calculate distance along each line
    df["dist_along_line"] = distance_along_line(
        df,
        line_col_name="line",
        time_col_name="time",
    )

    # calculate median spacing
    # np.median(vd.median_distance(
    #     (constraint_points.easting, constraint_points.northing),
    #     k_nearest=1,
    # ))/1e3

    # df["median_spacing"] = median_spacing

    # sample gravity at points and regrid
    if grav_grid is not None:
        df = profiles.sample_grids(
            df,
            grav_grid,
            "gravity_anomaly",
            coord_names=("easting", "northing"),
        )

    if plot:
        fig = maps.basemap(
            region=vd.pad_region(region, 0.1 * (region[1] - region[0])),
            frame=True,
        )
        fig.plot(
            x=[region[0], region[0], region[1], region[1], region[0]],
            y=[region[2], region[3], region[3], region[2], region[2]],
            pen=".5p,black",
            label="inversion region",
        )

        if grav_grid is not None:
            pygmt.makecpt(
                cmap="viridis",
                series=[df.gravity_anomaly.min(), df.gravity_anomaly.max()],
                background=True,
            )
            fig.plot(
                x=df.easting,
                y=df.northing,
                style="c0.1c",
                fill=df.gravity_anomaly,
                cmap=True,
            )
            maps.add_colorbar(fig, cbar_label="mGal", cbar_yoffset=1)
        fig.plot(
            df[["easting", "northing"]],
            style="p",
            fill="red",
            label="observation points",
        )

        fig.legend()
        fig.show()

    return df


def min_distance_linspace(
    start: float,
    stop: float,
    num: float,
):
    """
    Return evenly spaced values which result in the the minimal max distance between any
    of the values and anypoint between start and stop.
    """
    if num == 0:
        return []
    half_range = (stop-start)/2
    first_n_odd_numbers = [2*i + 1 for i in range(num)]
    return [start+(half_range/num) * x for x in first_n_odd_numbers]


def average_consecutive_difference(numbers):
    """
    Calculates the average difference between consecutive numbers in a list.
    """
    if len(numbers) < 2:
        return np.nan
    differences = [numbers[i+1] - numbers[i] for i in range(len(numbers) - 1)]
    return sum(differences) / len(differences)


def rotated_airborne_survey(
    along_line_spacing: float,
    grav_observation_height: float,
    survey_polygon: shapely.geometry.polygon.Polygon,
    line_spacing: float | None = None,
    tie_spacing: float | None = None,
    line_numbers: float | None = None,
    tie_numbers: float | None = None,
    padding: float | None = None,
    mask: shapely.geometry.polygon.Polygon | None = None,
    proximity_mask: shapely.geometry.polygon.Polygon | None = None,
    survey_spacing_mask: shapely.geometry.polygon.Polygon | None = None,
    plot: bool = False,
):
    """
    Create a synthetic airborne gravity survey within a survey polygon. Specify either
    number of lines or line spacing, and either number of ties or tie spacing. Line are
    automatically oriented along the longest edge of the polygon, and ties are oriented
    perpendicular to the lines. If number of lines or ties is supplied, will make them
    evenly spaced to minimize the maximum distance between any point and any line. Will
    return a dataframe with the survey points and survey metadata such as line spacing
    and gravity point proximity. If a 'mask' is supplied, only points within the mask
    will be retained. If a 'proximity_mask' is supplied, the proximity will only be
    calculated with the masked. If a 'survey_spacing_mask' is supplied, the regions used
    to calculate the spacing of lines and ties will be determined from the mask's
    region.
    """
    if mask is not None:
        assert isinstance(mask, shapely.geometry.multipolygon.MultiPolygon | shapely.geometry.polygon.Polygon), "mask must be a shapely polygon"
    if proximity_mask is not None:
        assert isinstance(proximity_mask, shapely.geometry.multipolygon.MultiPolygon | shapely.geometry.polygon.Polygon), "proximity_mask must be a shapely polygon"
    if survey_spacing_mask is not None:
        assert isinstance(survey_spacing_mask, shapely.geometry.multipolygon.MultiPolygon | shapely.geometry.polygon.Polygon), "survey_spacing_mask must be a shapely polygon"

    assert isinstance(survey_polygon, shapely.geometry.multipolygon.MultiPolygon | shapely.geometry.polygon.Polygon), "polygon must be a shapely polygon"

    if line_spacing is not None and line_numbers is not None:
        raise ValueError("line_spacing and line_numbers cannot both be set")
    if line_spacing is None and line_numbers is None:
        raise ValueError("line_spacing or line_numbers must be set")
    if tie_spacing is not None and tie_numbers is not None:
        raise ValueError("tie_spacing and tie_numbers cannot both be set")
    if tie_spacing is None and tie_numbers is None:
        raise ValueError("tie_spacing or tie_numbers must be set")

    # center of region
    center = (survey_polygon.centroid.x, survey_polygon.centroid.y)

    # extract rotation of polygon
    # angle between longest edge and x-axis
    def _azimuth(point1, point2):
        """azimuth between 2 points (interval 0 - 180)"""

        angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
        return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

    def _dist(a, b):
        """distance between points"""
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def azimuth(mrr):
        """azimuth of minimum_rotated_rectangle"""
        bbox = list(mrr.exterior.coords)
        axis1 = _dist(bbox[0], bbox[3])
        axis2 = _dist(bbox[0], bbox[1])

        if axis1 <= axis2:
            az = _azimuth(bbox[0], bbox[1])
        else:
            az = _azimuth(bbox[0], bbox[3])

        return az
    angle = azimuth(survey_polygon)
    logger.info("polygon rotated by %s degrees", angle)

    # rotate polygon to align with x/y axes
    gdf = gpd.GeoDataFrame(geometry=[survey_polygon])
    polygon_unrotated = gdf.rotate(-angle, origin=center).iloc[0]
    survey_region = utils.region_to_bounding_box(polygon_unrotated.bounds)

    if survey_spacing_mask is not None:
        # rotate mask to align with x/y axes
        gdf = gpd.GeoDataFrame(geometry=[survey_spacing_mask])
        mask_unrotated = gdf.rotate(-angle, origin=center).iloc[0]
        region_for_survey_spacing = utils.region_to_bounding_box(mask_unrotated.bounds)
    else:
        region_for_survey_spacing = survey_region

    if padding is not None:
        region_for_survey_spacing = vd.pad_region(region_for_survey_spacing, -padding)

    # determine if x or y is the longer edge
    if survey_region[1] - survey_region[0] > survey_region[3] - survey_region[2]:
        # x is the longer edge
        new_survey_region = survey_region
    else:
        # y is the longer edge
        new_survey_region = (
            survey_region[2],
            survey_region[3],
            survey_region[0],
            survey_region[1],
        )
        region_for_survey_spacing = (
            region_for_survey_spacing[2],
            region_for_survey_spacing[3],
            region_for_survey_spacing[0],
            region_for_survey_spacing[1],
        )

    ###
    ###
    # Make tie lines
    ###
    ###
    if tie_spacing is not None:
        # simulate tie lines spaced equally along long edge
        ties_across_coords = np.arange(*region_for_survey_spacing[:2], tie_spacing)
        # shift to be centered
        ties_mid = np.mean(ties_across_coords)
        ties_reg_mid = np.mean(region_for_survey_spacing[:2])
        ties_across_coords += ties_reg_mid - ties_mid
    elif tie_numbers is not None:
        ties_across_coords = min_distance_linspace(*region_for_survey_spacing[:2], tie_numbers)
    if len(ties_across_coords) == 0:
        df_ties = pd.DataFrame(columns=["easting", "northing", "line"])
    else:
        # simulate points along tie lines (inline with short edge)
        ties_along_coords = np.arange(*new_survey_region[2:], along_line_spacing)
        # combine along and across lines
        coords = np.meshgrid(ties_across_coords, ties_along_coords)

        # turn coordinates into dataarray
        ties = vd.make_xarray_grid(
            coords,
            data=np.ones_like(coords[0]) * grav_observation_height,
            data_names="upward",
            dims=("northing", "easting"),
        )
        # turn dataarray into dataframe
        df_ties = vd.grid_to_table(ties)

        # give each tie line a number starting at 1000 in increments of 10
        df_ties["line"] = np.nan
    for i, j in enumerate(df_ties.easting.unique()):
        df_ties["line"] = np.where(df_ties.easting == j, 1000 + i * 10, df_ties.line)

    ###
    ###
    # Make flight lines
    ###
    ###
    if line_spacing is not None:
        # simulate flight lines spaced evenly along the shorter edge
        lines_across_coords = np.arange(*region_for_survey_spacing[2:], line_spacing)
        # shift to be centered
        lines_mid = np.mean(lines_across_coords)
        lines_reg_mid = np.mean(region_for_survey_spacing[2:])
        lines_across_coords += lines_reg_mid - lines_mid
    elif line_numbers is not None:
        lines_across_coords = min_distance_linspace(*region_for_survey_spacing[2:], line_numbers)

    if len(lines_across_coords) == 0:
        df_lines = pd.DataFrame(columns=["easting", "northing", "line"])
    else:
        # simulate points along lines (inline with long edge)
        lines_along_coords = np.arange(*new_survey_region[:2], along_line_spacing)
        # combine along and across lines
        coords = np.meshgrid(lines_along_coords, lines_across_coords)

        # turn coordinates into dataarray
        lines = vd.make_xarray_grid(
            coords,
            data=np.ones_like(coords[0]) * grav_observation_height,
            data_names="upward",
            dims=("northing", "easting"),
        )
        # turn dataarray into dataframe
        df_lines = vd.grid_to_table(lines)

        # give each line a number starting at 0 in increments of 10
        df_lines["line"] = np.nan
        for i, j in enumerate(df_lines.northing.unique()):
            df_lines["line"] = np.where(df_lines.northing == j, i + 1 * 10, df_lines.line)

    # merge dataframes
    df = pd.concat([df_ties, df_lines])

    ###
    ###
    # rotate back to original orientation
    ###
    ###
    # convert to geopandas
    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x=df.easting, y=df.northing),
        crs="EPSG:3031",
    )

    # convert points into lines
    lines = df.groupby(['line'])['geometry'].apply(lambda x: LineString(x.tolist()))
    lines = gpd.GeoDataFrame(lines, geometry='geometry')
    lines["line"] = lines.index

    # rotate lines about the center of the survey region
    lines_rotated = lines.rotate(
        angle,
        origin=(np.mean(survey_region[0:2]), np.mean(survey_region[2:])),
    )

    # convert back to points
    rotated_df = lines_rotated.get_coordinates(index_parts=True)

    # add back line number
    rotated_df = rotated_df.reset_index()
    rotated_df = rotated_df.drop(columns="level_1")

    # rename cols
    rotated_df = rotated_df.rename(columns={"x": "easting", "y": "northing"})
    assert len(rotated_df) == len(df)

    df = rotated_df

    # convert to geopandas
    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x=df.easting, y=df.northing),
        crs="EPSG:3031",
    )

    # calculate flight distance using along line spacing
    flight_kms = (len(df)*along_line_spacing)/1e3

    if mask is not None:
        # mask to shape file
        df["inside"] = df.within(mask)

        # recalculate flight distance
        distance_outside_mask = (len(df[~df.inside])*along_line_spacing)/1e3
        flight_kms -= distance_outside_mask

        df = df[df.inside]
        df = df.drop(columns="inside")

    # add a time column
    df["time"] = np.nan
    for i in df.line.unique():
        if i >= 1000:
            time = df[df.line == i].sort_values("northing").reset_index().index.values
        else:
            time = df[df.line == i].sort_values("easting").reset_index().index.values
        df.loc[df.line == i, "time"] = time

    # calculate distance along each line
    df["dist_along_line"] = distance_along_line(
        df,
        line_col_name="line",
        time_col_name="time",
    )

    ###
    ###
    # Calculate some stats of the survey
    ###
    ###
    tie_spacing = average_consecutive_difference(ties_across_coords)/1e3
    line_spacing = average_consecutive_difference(lines_across_coords)/1e3

    if len(lines_across_coords) > 1 and len(ties_across_coords) > 1:
        survey_spacing = np.mean([line_spacing, tie_spacing])
    elif len(lines_across_coords) > 1:
        survey_spacing = line_spacing
    elif len(ties_across_coords) > 1:
        survey_spacing = tie_spacing
    else:
        survey_spacing = np.nan

    # calculate gravity data proximity
    region = utils.region_to_bounding_box(survey_polygon.bounds)
    coords = vd.grid_coordinates(
        region=region,
        spacing=100,
    )
    grid = vd.make_xarray_grid(coords, np.ones_like(coords[0]), data_names="z").z
    min_dist = inv_utils.dist_nearest_points(
        df,
        grid,
    ).min_dist/1e3

    # clip to survey polygon
    min_dist = utils.mask_from_shp(
        shapefile=gpd.GeoDataFrame(geometry=[survey_polygon]),
        xr_grid=min_dist,
        invert=False,
        masked=True,
    )

    # clip to mask
    if mask is not None:
        min_dist = utils.mask_from_shp(
            shapefile=gpd.GeoDataFrame(geometry=[mask]),
            xr_grid=min_dist,
            invert=False,
            masked=True,
        )

    if proximity_mask is not None:
        min_dist = utils.mask_from_shp(
            shapefile=gpd.GeoDataFrame(geometry=[proximity_mask]),
            xr_grid=min_dist,
            invert=False,
            masked=True,
        )
    median_proximity = np.nanmedian(min_dist)
    max_proximity = np.nanmax(min_dist)

    if plot:
        if line_numbers is not None:
            title = f"{line_numbers} lines, {tie_numbers} ties"
        elif line_spacing is not None:
            title = f"{round(line_spacing, 2)} km line spacing, {round(tie_spacing, 2)} km tie spacing"
        else: title = None
        # fig = maps.basemap(
        #     simple_basemap=True,
        #     simple_basemap_version="measures-v2",
        #     region=regions.combine_regions(
        #         utils.region_to_bounding_box(survey_polygon.bounds),
        #         utils.region_to_bounding_box(polygon_unrotated.bounds),
        #     ),
        #     scalebar=True,
        #     title=title,
        # )
        fig = maps.plot_grd(
            min_dist,
            cmap="dense",
            hist=True,
            cpt_lims=(0, utils.get_min_max(min_dist, robust=True,)[1]),
            cbar_label=f"distance to nearest datapoint, median {round(median_proximity,2)} (km)",
            simple_basemap=True,
            simple_basemap_version="measures-v2",
            region=region,
            scalebar=True,
            title=title,
        )
        fig.plot(
            data=gpd.GeoDataFrame(geometry=[survey_polygon]),
            pen="1p,black",
            label="supplied survey polygon",
        )
        # fig.plot(
        #     data=gpd.GeoDataFrame(geometry=[polygon_unrotated]),
        #     pen="1p,blue",
        #     label="unrotated survey polygon",
        # )
        if len(ties_across_coords) > 0:
            fig.plot(
                df[df.line>=1000][["easting", "northing"]],
                style="p.5p",
                fill="blue",
                label="tie lines",
            )
        if len(lines_across_coords) > 0:
            fig.plot(
                df[df.line<1000][["easting", "northing"]],
                style="p.5p",
                fill="red",
                label="flight lines",
            )
        if mask is not None:
            fig.plot(
                data=gpd.GeoDataFrame(geometry=[mask]),
                pen="1p,magenta",
                label="mask",
            )
        fig.legend()
        fig.show()

    # add survey info to attrs
    df.attrs = {
        "number_ties": len(df[df.line>=1000].line.unique()),
        "number_lines": len(df[df.line<1000].line.unique()),
        "tie_spacing": tie_spacing,
        "line_spacing": line_spacing,
        "survey_spacing": survey_spacing,
        "flight_kms": flight_kms,
        "median_proximity": median_proximity,
        "max_proximity": max_proximity,
        "proximity_skew": scipy.stats.skew(min_dist.values.ravel(), nan_policy="omit"),
    }
    return df


def distance_along_line(
    data: gpd.GeoDataFrame | pd.DataFrame,
    line_col_name: str = "line",
    time_col_name: str = "unixtime",
) -> pd.Series:
    """
    Calculate the distances along each flight line in meters, assuming the lowest time
    value is the start of each lines. If you don't have time information, you can pass
    the index of the dataframe as the time column.

    Parameters
    ----------
    data : gpd.GeoDataFrame | pd.DataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have a set geometry column.
    line_col_name : str, optional
        Column name specifying the line number, by default "line"
    time_col_name : str, optional
        Column name containing time in seconds for each datapoint, by default "unixtime"

    Returns
    -------
    pd.Series
        The distance along each line in meters
    """

    gdf = data.copy()

    gdf["dist_along_line"] = np.nan
    for i in gdf[line_col_name].unique():
        line = gdf[gdf[line_col_name] == i]
        dist = line.distance(line.sort_values(by=time_col_name).geometry.iloc[0]).values
        gdf.loc[gdf[line_col_name] == i, "dist_along_line"] = dist

    return gdf.dist_along_line


def filter_flight_lines(
    df: gpd.GeoDataFrame | pd.DataFrame,
    filt_type: str,
    data_column: str,
    distance_column: str = "dist_along_line",
    line_column: str = "line",
    pad_width_percentage: float = 10,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    _summary_

    Parameters
    ----------
    df : gpd.GeoDataFrame | pd.DataFrame
        _description_
    filt_type : str
        a string with format "<type><width>+h" where type is GMT filter type, width is
        the filter width in same units as distance column, and optional +h switches from
        low-pass to high-pass filter; e.g. "g10+h" is a 10m high-pass Gaussian filter.
    data_column : str
        _description_
    distance_column : str, optional
        _description_, by default "dist_along_line"
    line_column : str, optional
        _description_, by default "line"
    pad_width_percentage : float, optional
        _description_, by default 10

    Returns
    -------
    gpd.GeoDataFrame | pd.DataFrame
        _description_
    """

    df = df.copy()

    for i in df[line_column].unique():
        # subset data from 1 line
        line = df[df[line_column] == i]
        line = line[[distance_column, data_column]]

        # get data spacing
        distance = line[distance_column].values
        data_spacing = np.median(np.diff(distance))

        # pad distance of 10% of line distance
        pad_dist = (distance.max() - distance.min()) * (pad_width_percentage / 100)
        pad_dist = round(pad_dist / data_spacing) * data_spacing

        # get the number of points to pad
        # n_pad = int(pad_dist / data_spacing)

        # add pad points to distance values
        lower_pad = np.arange(distance.min() - pad_dist, distance.min(), data_spacing,)
        upper_pad = np.arange(distance.max(), distance.max() + pad_dist, data_spacing,)
        vals = np.concatenate((lower_pad, upper_pad))
        new_dist = pd.DataFrame({distance_column: vals})

        # pad the line, fill padded values in data with nearest value
        padded = pd.concat([line.reset_index(),new_dist],).sort_values(by=distance_column).set_index("index")
        padded = padded.fillna(method='ffill').fillna(method='bfill').reset_index()
        padded = padded.rename(columns={"index":"true_index"})

        # filter the padded data
        filtered = pygmt.filter1d(
            padded[[distance_column, data_column]],
            end=True,
            time_col=0,
            filter_type=filt_type,
        ).rename(columns={0:distance_column, 1:data_column})

        # un-pad the data
        filtered["index"] = padded.true_index
        filtered = filtered.set_index("index")
        filtered = filtered[filtered.index.isin(line.index)]

        # replace original data with filtered data
        df.loc[df[line_column] == i, data_column] = filtered[data_column]

    return df[data_column]

def scipy_interp1d(
    df,
    to_interp=None,
    interp_on=None,
    method=None,
):
    """
    interpolate NaN's in "to_interp" column, based on values from "interp_on" column
    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next'
    use kwargs to pass other arguments to scipy.interpolate.interp1d()
    """
    df1 = df.copy()

    # drop NaN's
    df_no_nans = df1.dropna(subset=[to_interp, interp_on], how="any")

    # define interpolation function
    f = scipy.interpolate.interp1d(
        df_no_nans[interp_on],
        df_no_nans[to_interp],
        kind=method,
    )

    # get interpolated values at points with NaN's
    values = f(df1[df1[to_interp].isnull()][interp_on])

    # fill NaN's  with values
    df1.loc[df1[to_interp].isnull(), to_interp] = values

    return df1