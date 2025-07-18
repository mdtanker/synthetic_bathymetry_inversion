from __future__ import annotations  # pylint: disable=too-many-lines

import typing

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import scipy as sp
import shapely
import verde as vd
import xarray as xr

try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None
from invert4geom import regional
from invert4geom import utils as invert4geom_utils
from matplotlib import patheffects
from polartoolkit import fetch, maps, profiles, utils
from tqdm.autonotebook import tqdm

from synthetic_bathymetry_inversion import logger

inverted_shelves = [
    "Ross",  # tinto 2019
    "Amery",  # yang 2021
    "Borchgrevink",  # eisermann 2021
    "Baudouin",  # eisermann 2021
    "George_VI",  # constantino 2020
    "Thwaites",  # tinto and bell 2011, jordan 2020, millan 2017
    "Cook",  # constantino and tinto 2023
    "Ninnis",  # constantino and tinto 2023
    "Ekstrom",  # eisermann 2020
    "Atka",  # eisermann 2020
    "Jelbart",  # eisermann 2020
    "Fimbul",  # eisermann 2020
    "Vigrid",  # eisermann 2020
    "Crosson",  # jordan 2020, millan 2017
    "Dotson",  # jordan 2020, millan 2017
    "Getz",  # wei 2020, millan 2020, cochran 2020
    "Totten",  # greenbaum 2015, vankova 2023
    "Brunt_Stancomb",  # hodgson 2019
    "Pine_Island",  # millan 2017, muto 2013, muto 2016
    "Abbot",  # cochran 2014
    "LarsenC",  # cochran and bell 2012
    "Nivl",  # eisermann 2024
    "Lazarev",  # eisermann 2024
    "Venable",  # Locke et al. 2025
]


def combine_offshore_onshore_points(
    region: tuple[float, float, float, float],
    version: str = "bedmachine",
    bedmap_version: str = "all",
    grounded_as_points: bool = False,
    spacing: float = 1e3,
) -> pd.DataFrame:
    """
    Create a dataframe of constraint points for a region. The points are a combination
    of IBCSO v2 points offshore and your choice of bedmap or bedmachine
    points. The bedmap points can be either from bedmap versions 1,2,3 or all combined,
    or you can treat each grid cell (1km) of the Bedmap2 grid as a
    constraint point. The bedmachine points are extracted from the grid of data id
    values.

    Parameters
    ----------
    region : tuple[float, float, float, float]
        region to extract points for
    version : str, optional
        choose between bedmachine or bedmap points, by default "bedmachine"
    bedmap_version : str, optional
        choose which version of bedmap data to use, 'bedmap1', 'bedmap2', 'bedmap3', or
        'all' combined, by default "all"
    grounded_as_points : bool, optional
        treat grid cells of grounded ice as points rather than using bedmap points, by
        default False
    spacing : float, optional
        sampling spacing to use for converted IBCSO multibeam polygons to points, by
        default 1e3

    Returns
    -------
    pd.DataFrame
        a dataframe of constraint points
    """

    # fetch points for region
    ibcso_points_gdf, ibcso_polygons_gdf = fetch.ibcso_coverage(region=region)

    # drop points with TID of 45 meaning it's from gravity inversion
    ibcso_points_gdf = ibcso_points_gdf[ibcso_points_gdf.dataset_tid != 45]

    # convert polygons (mostly swath multibeam) to grid of points which are within the
    # polygon with a grid spacing specified
    polygon_points = None
    if len(ibcso_polygons_gdf) > 0:
        try:
            polygon_points = polygons_to_points(
                ibcso_polygons_gdf,
                spacing=spacing,
            )
        except ValueError as e:
            logger.error(e)
            logger.error("Failed to convert polygons to points")

    # combine ibcso points and polygon points
    total_ibcso_points = pd.concat([ibcso_points_gdf, polygon_points])

    # drop geometry column
    total_ibcso_points = total_ibcso_points.drop(columns=["geometry"])

    if version == "bedmap":
        # used bedmap points just for grounded region
        if grounded_as_points is False:
            bedmap_points_gdf = fetch.bedmap_points(
                region=region, version=bedmap_version
            )

            # drop points without bed elevations
            bedmap_points_gdf = bedmap_points_gdf.dropna(
                subset=["bedrock_altitude (m)"]
            )

            # get grid of rock outcrops
            rock_mask = fetch.bedmap2(
                layer="rockmask",
                region=region,
                spacing=1e3,
                verbose="q",
            ).rename("bedrock_altitude (m)")

            # convert to dataframe
            rock_df = vd.grid_to_table(rock_mask).dropna()

            # merge with bedmap points
            bedmap_points_gdf = pd.concat([rock_df, bedmap_points_gdf])

            # convert to a geodataframe
            geometry = [
                shapely.geometry.Point(xy)
                for xy in zip(bedmap_points_gdf.x, bedmap_points_gdf.y)
            ]
            bedmap_points_gdf = gpd.GeoDataFrame(
                bedmap_points_gdf, crs="EPSG:3031", geometry=geometry
            )

            # get grounding line shapefile
            grounding_line = gpd.read_file(
                fetch.groundingline(version="measures-v2"), engine="pyogrio"
            )

            # add column indicating if points are grounded or not
            bedmap_points_gdf = gpd.tools.sjoin(
                bedmap_points_gdf,
                grounding_line,
                predicate="within",
                how="left",
            )

            # discard points offshore
            bedmap_points_grounded = bedmap_points_gdf[
                bedmap_points_gdf.NAME == "Grounded"
            ]

            # rename bedmap columns to match ibcso and drop unused columns
            bedmap_points_grounded = bedmap_points_grounded[
                ["x", "y", "project"]
            ].copy()
            bedmap_points_grounded = bedmap_points_grounded.rename(
                columns={
                    "x": "easting",
                    "y": "northing",
                    "project": "dataset_name",
                }
            )

        # treat every gridcell of grounded ice as a constraint point
        elif grounded_as_points is True:
            # get grid with 0 for grounded and 1 for floating ice
            grounded_da = fetch.bedmap2(
                layer="icemask_grounded_and_shelves",
                region=region,
                spacing=1e3,
                verbose="q",
            )
            # turn into dataframe
            bedmap_points_grounded = vd.grid_to_table(grounded_da).dropna()

            # keep only grounded points
            bedmap_points_grounded = bedmap_points_grounded[
                bedmap_points_grounded.z == 0
            ]

            # add column for project
            bedmap_points_grounded["dataset_name"] = "bedmap2_grid_points"

            # rename columns
            bedmap_points_grounded = bedmap_points_grounded.rename(
                columns={
                    "x": "easting",
                    "y": "northing",
                }
            )

            # drop some columns
            bedmap_points_grounded = bedmap_points_grounded.drop(columns=["z"])

        bedmap_points_grounded["onshore"] = True
        points = bedmap_points_grounded

    elif version == "bedmachine":
        number_to_source = {
            0: "no_data",
            1: "rema",
            2: "radar",
            7: "seismic",
            10: "multibeam",
        }

        dataid = fetch.bedmachine(layer="dataid")
        # drop no data points
        dataid_ds = dataid.where(dataid != 0, drop=True)

        dataid_df = (
            vd.grid_to_table(dataid_ds)
            .dropna()
            .rename(columns={"x": "easting", "y": "northing"})
        )

        dataid_df["source"] = dataid_df.dataid.map(number_to_source)

        # drop seismic points (only for Amery and Ronne-Filchner) and seem to be erroneous # noqa: E501
        dataid_df = dataid_df[dataid_df.source != "seismic"]

        dataid_df = dataid_df[["easting", "northing", "dataid", "source"]]

        dataid_df = utils.points_inside_region(dataid_df, region)
        dataid_df["onshore"] = np.where(dataid_df.dataid == 10, False, True)
        points = dataid_df

    else:
        msg = "version must be either 'bedmap' or 'bedmachine'"
        raise ValueError(msg)

    # add column indicating if points are offshore
    total_ibcso_points["onshore"] = False

    return pd.concat(
        [
            total_ibcso_points.drop(columns=["weight", "dataset_tid"]),
            points,
        ],
        ignore_index=True,
        sort=False,
    )


def get_ice_shelves():
    # read into a geodataframe
    ice_shelves = gpd.read_file(fetch.antarctic_boundaries(version="IceShelf"))

    # get list of unique ice shelf names
    names = ice_shelves.NAME.unique()

    logger.info("Number of unmerged ice shelves: %s", len(names))

    # logger.info(names)

    # get Series of first words in ice shelf names and number of shelves with that name
    first_words = pd.Series(n.split("_")[0] for n in names).value_counts()

    # get list of lists of ice shelves which should be combined
    shelves_to_combine = {
        "Ronne_Filchner": ["Ronne", "Filchner"],
    }
    dont_combine = ["Hamilton"]
    for i, j in first_words.items():
        if (j > 1) & (i not in dont_combine):
            shelves_to_combine[i] = [n for n in names if n.startswith(i)]

    shelves_to_merge = []
    num_sub_shelves = 0
    for k, v in shelves_to_combine.items():
        logger.info("Combining %s %s sub-shelves", len(v), k)

        num_sub_shelves += len(v)
        # merge into new shelf
        shelf = ice_shelves[ice_shelves.NAME.isin(v)].dissolve()
        shelf.NAME = k

        # remove old shelves
        ice_shelves = ice_shelves[~ice_shelves.NAME.isin(v)]

        shelves_to_merge.append(shelf)

    logger.info("Number of removed sub-shelves: %s", num_sub_shelves)

    # add to dataframe
    ice_shelves = pd.concat([ice_shelves] + shelves_to_merge)  # noqa: RUF005

    # there are 2 Fox ice shelves, one in East Antarctica and one in West Antarctica
    # append the region to the name to differentiate
    ice_shelves.loc[
        (ice_shelves.NAME == "Fox") & (ice_shelves.Regions == "West"), "NAME"
    ] = "Fox_West"
    ice_shelves.loc[
        (ice_shelves.NAME == "Fox") & (ice_shelves.Regions == "East"), "NAME"
    ] = "Fox_East"

    # calculate area in sq km of each ice shelf
    ice_shelves["area_km"] = round(ice_shelves.area / (1000**2), 2)

    # sort by area
    ice_shelves = ice_shelves.sort_values(by="area_km", ascending=False).reset_index(
        drop=True
    )

    logger.info("Number of merged ice shelves: %s", len(ice_shelves))
    # get coordinate columns
    # ice_shelves["easting"] = ice_shelves.get_coordinates().x
    # ice_shelves["northing"] = ice_shelves.get_coordinates().y

    return ice_shelves


def plot_ice_shelf_names(
    fig: pygmt.Figure,
    ice_shelves: gpd.GeoDataFrame,
    font="8p,Helvetica",
    offset=".4c/.4c+v.6p",
    justify="BL",
    shadow=True,
    shadow_font="2p,white",
    names_as_numbers=False,
):
    if names_as_numbers:
        text = [str(i) for i in range(1, len(ice_shelves) + 1)]
    else:
        text = [x.replace("_", " ") for x in ice_shelves.NAME]

    # plot white shadow behind names
    if shadow:
        fig.text(
            x=ice_shelves.geometry.centroid.x,
            y=ice_shelves.geometry.centroid.y,
            text=text,
            font=f"{font},black,-={shadow_font}",
            justify=justify,
            offset=offset,
            no_clip=True,
        )
    # plot names
    fig.text(
        x=ice_shelves.geometry.centroid.x,
        y=ice_shelves.geometry.centroid.y,
        text=text,
        font=f"{font},black",
        justify=justify,
        offset=offset,
        no_clip=True,
    )


def constraints_and_min_distances_single(
    ice_shelf: gpd.GeoSeries,
    buffer: float = 20e3,
    spacing: float = 100,
    version: str = "bedmachine",
    bedmap_version: str = "all",
    grounded_as_points: bool = False,
    fname: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    # define region around each ice shelf with 10km buffer
    reg = utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
    reg = vd.pad_region(reg, buffer)

    # get ibcso points offshore and bedmap points onshore for the region
    bed_points = combine_offshore_onshore_points(
        bedmap_version=bedmap_version,
        region=reg,
        version=version,
        grounded_as_points=grounded_as_points,
        spacing=spacing,
    )

    if fname is not None:
        bed_points.to_csv(
            f"{fname}_constraints.csv.gz",
            sep=",",
            na_rep="",
            header=True,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )
    # add column indicating which ice shelf each point is within
    # bed_points = gpd.tools.sjoin(
    #     bed_points,
    #     gdf,
    #     predicate="within",
    #     how="left",
    # )
    # use only points within specific ice shelf (ungrounded)
    # bed_points = bed_points[bed_points.NAME==gdf.iloc[0].NAME]

    coords = vd.grid_coordinates(
        region=reg,
        spacing=spacing,
    )
    grid = vd.make_xarray_grid(coords, np.ones_like(coords[0]), data_names="z").z

    # calculate minimum distance between each grid cell and the nearest point
    try:
        min_dist = (
            invert4geom_utils.normalized_mindist(
                bed_points,
                grid=grid,
            )
            / 1e3
        )
    except ValueError as e:
        logger.error(e)
        logger.error("issue with calculating minimum distance for %s", gdf.NAME)
        return
    # mask to the ice shelf outline
    min_dist = utils.mask_from_shp(
        shapefile=gdf,
        grid=min_dist,
        invert=False,
        masked=True,
    )

    if fname is not None:
        min_dist.to_netcdf(f"{fname}_min_dist.nc")

    if plot:
        fname = fname if save_plot else None
        plot_constraints_and_min_distances(
            gdf.iloc[0],
            min_dist,
            bed_points,
            fname,
        )


def constraints_and_min_distances(
    ice_shelves: gpd.GeoDataFrame,
    buffer: float = 20e3,
    version: str = "bedmachine",
    spacing: float = 100,
    bedmap_version: str = "all",
    grounded_as_points: bool = False,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    """
    Compute the median of the minimum distances (in km) between each grid cell and the
    nearest constraint point (bedmap onshore, ibcso offshore) for each supplied ice
    shelf shapefile. Save the constraints to a csv and the minimum distance grid to a
    netcdf. Optionally, plot the results.

    Parameters
    ----------
    ice_shelves : gpd.GeoDataFrame
        ice shelf shapefiles, output from `get_ice_shelves`
    buffer : float, optional
        buffer zone in meters around ice shelf bounding box to include points in, by
        default 20e3
    spacing : float, optional
        grid spacing in meters to use for computing minimum distances and converted
        polygons to points, by default 100
    bedmap_version : str, optional
        which bedmap points to include, either "bedmap1", "bedmap2", "bedmap3", or
        "all", by default "all"
    grounded_as_points : bool, optional
        use bedmap grid cells as points rather than actual points, by default False
    file_path : str | None, optional
        path to save the constraints and minimum distance files, by default None
    plot : bool, optional
        show a plot for each ice shelf, by default False

    Returns
    -------
    gpd.GeoDataFrame
        Ice shelf geodataframe with additional column "median_min_dist" containing the
        median minimum distance to the nearest point
    """
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for _i, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")
        constraints_and_min_distances_single(
            row,
            buffer=buffer,
            version=version,
            spacing=spacing,
            bedmap_version=bedmap_version,
            grounded_as_points=grounded_as_points,
            fname=file_path + row.NAME,
            plot=plot,
            save_plot=save_plot,
        )


def load_constraints_and_min_distances(
    ice_shelves: gpd.GeoDataFrame,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        try:
            constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load constraints for %s", row.NAME)
            continue
        try:
            min_dist = xr.open_dataarray(f"{file_path}{row.NAME}_min_dist.nc")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load minimum distances for %s", row.NAME)
            continue

        gdf.loc[i, "median_constraint_distance"] = min_dist.median().values
        gdf.loc[i, "mean_constraint_distance"] = min_dist.mean().values
        gdf.loc[i, "max_constraint_distance"] = min_dist.max().values
        gdf.loc[i, "constraint_proximity_skewness"] = sp.stats.skew(
            min_dist.values.ravel(), nan_policy="omit"
        )

        if plot:
            fname = file_path + row.NAME if save_plot else None

            plot_constraints_and_min_distances(
                row,
                min_dist=min_dist,
                constraints_df=constraints_df,
                fname=fname,
            )

    return gdf


def add_single_constraint(
    ice_shelves: gpd.GeoDataFrame,
    file_path: str | None = None,
    spacing=100,
    buffer=20e3,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        try:
            constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load constraints for %s", row.NAME)
            continue
        logger.debug("Loaded %s constraints for %s", len(constraints_df), row.NAME)

        # define region around each ice shelf with 10km buffer
        reg = utils.region_to_bounding_box(row.geometry.bounds)
        reg = vd.pad_region(reg, buffer)

        coords = vd.grid_coordinates(
            region=reg,
            spacing=spacing,
        )
        grid = vd.make_xarray_grid(coords, np.ones_like(coords[0]), data_names="z").z

        # calculate minimum distance between each grid cell and the nearest point
        min_dist = (
            invert4geom_utils.normalized_mindist(
                constraints_df,
                grid=grid,
            )
            / 1e3
        )

        # mask to the ice shelf outline
        min_dist = utils.mask_from_shp(
            shapefile=row,
            grid=min_dist,
            invert=False,
            masked=True,
        )
        logger.debug("Calculated minimum distances for %s", row.NAME)

        df = vd.grid_to_table(min_dist)
        min_dist_coords = df.iloc[df.min_dist.argmax()]

        min_dist_coords = pd.DataFrame(
            [{"easting": min_dist_coords.easting, "northing": min_dist_coords.northing}]
        )

        # add 1 constraint at point of max proximity
        constraints_df_update = pd.concat(
            [constraints_df, min_dist_coords],
            ignore_index=True,
        )

        # calculate minimum distance between each grid cell and the nearest point
        min_dist_update = (
            invert4geom_utils.normalized_mindist(
                constraints_df_update,
                grid=grid,
            )
            / 1e3
        )

        # mask to the ice shelf outline
        min_dist_update = utils.mask_from_shp(
            shapefile=row,
            grid=min_dist_update,
            invert=False,
            masked=True,
        )
        median_constraint_distance = min_dist.median().values
        updated_median_constraint_distance = min_dist_update.median().values
        median_proximity_change = (
            median_constraint_distance - updated_median_constraint_distance
        )
        gdf.loc[i, "median_constraint_distance"] = median_constraint_distance
        gdf.loc[i, "updated_median_constraint_distance"] = (
            updated_median_constraint_distance
        )
        gdf.loc[i, "median_proximity_change"] = median_proximity_change
        gdf.loc[i, "percent_median_proximity_change"] = (
            median_proximity_change / median_constraint_distance
        ) * 100

    return gdf


def plot_constraints_and_min_distances(
    ice_shelf: gpd.GeoSeries,
    min_dist: xr.DataArray,
    constraints_df: pd.DataFrame,
    fname: str | None = None,
):
    name = ice_shelf.NAME
    region = vd.get_region(
        (
            min_dist.easting.values,
            min_dist.northing.values,
        ),
    )

    fig = maps.plot_grd(
        min_dist,
        region=region,
        cmap="dense",
        hist=True,
        cpt_lims=(0, utils.get_min_max(min_dist, robust=False)[1]),
        cbar_label="Distance to nearest constraint (km)",
        cbar_font="18p,Helvetica",
        inset=True,
        inset_width=0.2,
        # inset_position=f"jTL+-{12*.2}c/0c",
        inset_position="jTR+jTL",
        scalebar=True,
        scalebar_box="+gwhite@30+p0.5p,gray30,solid+r3p",
        scalebar_position="jBL+o0.6c/0.6c",
        points=constraints_df,
        points_style="p1.5p",
        points_fill="black",
        # simple_basemap=True,
        # simple_basemap_version="measures-v2",
        modis_basemap=True,
        modis_version="125m",
        modis_transparency=60,
    )
    fig.plot(
        constraints_df.iloc[0], style="p2p", fill="black", label="existing constraints"
    )
    # plot outline of ice shelf
    fig.plot(
        gpd.GeoDataFrame(geometry=[ice_shelf.geometry]),
        pen="1.2p,salmon",
        label="MEaSUREs ice shelf boundary",
    )
    # add ice shelf name as title
    fig.text(
        position="TC",
        justify="BC",
        text=f"{name.replace("_", " ")} Ice Shelf",
        offset="0c/1c",
        font="20p,Helvetica",
        no_clip=True,
    )
    # add text stating median value
    fig.text(
        position="TC",
        justify="BC",
        text=f"Median; {round(np.nanmedian(min_dist),2)} km",
        offset="0c/.3c",
        font="16p,Helvetica",
        no_clip=True,
    )
    fig.legend(position="jTR+jTR", box="+gwhite@30+p1p,gray30,solid")
    fig.show()

    if fname is not None:
        fig.savefig(f"{fname}_constraints.png")


def gravity_anomalies_single(
    ice_shelf: gpd.GeoSeries,
    regional_grav_kwargs: dict,
    constraints_df: pd.DataFrame,
    buffer: float = 10e3,
    spacing: float = 10e3,
    progressbar: bool = False,
    fname: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    # define region around each ice shelf with 10km buffer
    reg = utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
    reg = vd.pad_region(reg, buffer)

    logger.debug("calculated region %s", reg)
    logger.debug("fetching, subsetting, and resampling datasets")

    # download gravity data
    eigen_gravity = fetch.gravity(
        version="eigen",
        spacing=5e3,
        region=reg,
        registration="g",
        verbose="q",
    )
    antgg_disturbance = fetch.gravity(
        version="antgg-2021",
        anomaly_type="DG",
        spacing=5e3,
        region=reg,
        registration="g",
        verbose="q",
    )
    antgg_uncertainty = fetch.gravity(
        version="antgg-2021",
        anomaly_type="Err",
        spacing=5e3,
        region=reg,
        registration="g",
        verbose="q",
    )

    # download topography
    surface = fetch.bedmap2(
        layer="surface",
        spacing=spacing,
        region=reg,
        fill_nans=True,
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x": "easting", "y": "northing"})
    icebase = fetch.bedmap2(
        layer="icebase",
        spacing=spacing,
        region=reg,
        fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid) # noqa: E501
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x": "easting", "y": "northing"})
    bed = fetch.bedmap2(
        layer="bed",
        spacing=spacing,
        region=reg,
        fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid) # noqa: E501
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x": "easting", "y": "northing"})

    # make gravity dataframe
    grav_df = (
        xr.merge(
            [
                eigen_gravity.gravity,
                antgg_disturbance.gravity_disturbance,
                antgg_disturbance.ellipsoidal_height,
                antgg_uncertainty.error,
            ]
        )
        .to_dataframe()
        .reset_index()
        .rename(columns={"x": "easting", "y": "northing"})
    )

    # check layers to cross
    # anywhere bed is above icebase, set equal to icebase
    bed = xr.where(bed > icebase, icebase, bed)

    # anywhere bed is above surface, set equal to surface
    bed = xr.where(bed > surface, surface, bed)

    # anywhere icebase is above surface, set equal to surface
    icebase = xr.where(icebase > surface, surface, icebase)

    # check it worked
    assert np.all(surface - icebase) >= 0
    assert np.all(surface - bed) >= 0
    assert np.all(icebase - bed) >= 0

    # set densities
    air_density = 1
    ice_density = 917
    water_density = 1025
    rock_density = 2670

    logger.debug("calculating terrain mass effect components")
    # make prism layers for terrain mass effect components
    ice_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=surface,
        reference=0,
        density=xr.where(
            surface >= 0, ice_density - air_density, air_density - ice_density
        ),
    )

    water_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=icebase,
        reference=0,
        density=xr.where(
            icebase >= 0, water_density - ice_density, ice_density - water_density
        ),
    )

    rock_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=bed,
        reference=0,
        density=xr.where(
            bed >= 0,
            rock_density - water_density,
            water_density - rock_density,
        ),
    )

    # calculate terrain mass effect components
    coords = (grav_df.easting, grav_df.northing, grav_df.ellipsoidal_height)
    grav_df["ice_surface_grav"] = ice_surface_prisms.prism_layer.gravity(
        coordinates=coords,
        field="g_z",
        progressbar=progressbar,
    )
    grav_df["water_surface_grav"] = water_surface_prisms.prism_layer.gravity(
        coordinates=coords,
        field="g_z",
        progressbar=progressbar,
    )
    grav_df["rock_surface_grav"] = rock_surface_prisms.prism_layer.gravity(
        coordinates=coords,
        field="g_z",
        progressbar=progressbar,
    )

    # calculate anomalies
    grav_df["terrain_mass_effect"] = (
        grav_df.ice_surface_grav
        + grav_df.water_surface_grav
        + grav_df.rock_surface_grav
    )
    grav_df["partial_topo_free_disturbance"] = (
        grav_df.gravity_disturbance
        - grav_df.ice_surface_grav
        - grav_df.water_surface_grav
    )
    grav_df["topo_free_disturbance"] = (
        grav_df.partial_topo_free_disturbance - grav_df.rock_surface_grav
    )

    # calculate gravity misfit, regional and residual
    # requires columns "starting_gravity" and "gravity_anomaly"
    grav_df["starting_gravity"] = grav_df.rock_surface_grav
    grav_df["gravity_anomaly"] = grav_df.partial_topo_free_disturbance

    logger.debug("calculating gravity misfit and regional/residual components")

    try:
        grav_df = regional.regional_separation(
            grav_df=grav_df,
            constraints_df=constraints_df,
            **regional_grav_kwargs,
        )
    except Exception as e:
        logger.error(e)
        logger.error("Error in regional separation")

        grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity
        grav_df["reg"] = np.nan
        grav_df["res"] = np.nan

    logger.debug("calculating stats for each anomaly type")
    # create mask for ice shelf
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()
    grav_grid["ice_shelf_mask"] = utils.mask_from_shp(
        shapefile=gdf,
        grid=grav_grid.partial_topo_free_disturbance,
        invert=False,
    ).rename("ice_shelf_mask")

    # subset data to ice shelf
    grav_df = grav_grid.to_dataframe().reset_index()

    if fname is not None:
        grav_df.to_csv(
            f"{fname}_grav_anomalies.csv.gz",
            sep=",",
            na_rep="",
            header=True,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )
    if plot:
        fname = fname if save_plot else None
        plot_grav_anomalies(
            gdf,
            grav_df,
            constraints_df,
            fname,
        )


def gravity_anomalies(
    ice_shelves: gpd.GeoDataFrame,
    regional_grav_kwargs: dict,
    buffer: float = 10e3,
    spacing: float = 10e3,
    progressbar: bool = False,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for _i, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")

        try:
            constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load constraints for %s", row.NAME)
            continue

        gravity_anomalies_single(
            row,
            regional_grav_kwargs=regional_grav_kwargs,
            constraints_df=constraints_df,
            buffer=buffer,
            spacing=spacing,
            progressbar=progressbar,
            fname=file_path + row.NAME,
            plot=plot,
            save_plot=save_plot,
        )


def load_grav_anomalies(
    ice_shelves: gpd.GeoDataFrame,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )

    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        try:
            constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load constraints for %s", row.NAME)
            continue

        try:
            grav_df = pd.read_csv(f"{file_path}{row.NAME}_grav_anomalies.csv.gz")
        except FileNotFoundError as e:
            logger.error(e)
            logger.error("Failed to load gravity anomalies for %s", row.NAME)
            continue

        # subset data to ice shelf
        grav_df_subset = grav_df[grav_df.ice_shelf_mask == True]  # noqa: E712 # pylint: disable=singleton-comparison

        # calculate stats on following columns
        anoms = [
            "gravity_disturbance",
            "partial_topo_free_disturbance",
            "topo_free_disturbance",
            "starting_gravity",
            # "gravity_anomaly",
            # "misfit",
            "reg",
            "res",
            "error",
        ]
        stats_df = pd.DataFrame(
            {
                "root_mean_square": [utils.rmse(grav_df_subset[a]) for a in anoms],
                "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
                "stdev": [grav_df_subset[a].std() for a in anoms],
            },
            index=anoms,
        )

        for row_name, _ in stats_df.iterrows():
            for col_name in stats_df.columns:
                gdf.loc[i, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]

        if plot:
            fname = file_path + row.NAME if save_plot else None
            plot_grav_anomalies(
                gdf,
                grav_df=grav_df,
                constraints_df=constraints_df,
                fname=fname,
            )

    return gdf


def plot_grav_anomalies(
    ice_shelf: gpd.GeoSeries,
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    fname: str | None = None,
):
    name = ice_shelf.iloc[0].NAME

    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()
    grav_grid = grav_grid.where(grav_grid.ice_shelf_mask == True, np.nan)  # noqa: E712 # pylint: disable=singleton-comparison

    anoms = [
        "gravity_disturbance",
        "topo_free_disturbance",
        "starting_gravity",
        "reg",
        "res",
    ]
    anom_titles = [
        "Gravity disturbance",
        "Topo-free disturbance",
        "Starting gravity",
        "Regional misfit",
        "Residual misfit",
    ]
    grids = [grav_grid[a] for a in anoms]
    cmaps = ["viridis"] * (len(anoms) - 1) + ["balance+h0"]
    reverse_cpts = [False] * (len(grids) - 1) + [True]
    insets = [False] * (len(grids))
    cbar_labels = [
        f"stdev: {round(grav_df[grav_df.ice_shelf_mask==True][a].std(),0)} mGal"  # noqa: E712 # pylint: disable=singleton-comparison
        for a in anoms
    ]
    offshore_points = constraints_df[constraints_df.onshore == False]  # noqa: E712 # pylint: disable=singleton-comparison
    if offshore_points.empty:
        offshore_points = None
    point_sets = [offshore_points] * (len(grids))
    scalebars = [True] + [False] * (len(grids) - 1)
    titles = anom_titles

    if np.isnan(grav_grid.error.values).all():
        logger.error("No uncertainty data for %s", name)
    else:
        cmaps.insert(0, "thermal")
        cbar_labels.insert(
            0, f"uncertainty; RMSE: {round(utils.rmse(grav_grid.error))} mGal"
        )
        reverse_cpts.insert(0, False)
        insets.insert(0, True)
        scalebars.insert(0, False)
        point_sets.insert(0, constraints_df)
        titles.insert(0, "AntGG gravity uncertainty")
        grids.insert(0, grav_grid.error)

    try:
        fig = maps.subplots(
            grids,
            fig_title=f"{name.replace("_", " ")} Ice Shelf",
            cmaps=cmaps,
            reverse_cpts=reverse_cpts,
            insets=insets,
            robust=True,
            coast=True,
            coast_version="measures-v2",
            hist=True,
            cbar_labels=cbar_labels,
            titles=titles,
            point_sets=point_sets,
            cbar_font="15p,Helvetica,black",
            points_style="c1p",
            scalebars=scalebars,
            simple_basemap=True,
            simple_basemap_version="measures-v2",
        )
        fig.show()
        if fname is not None:
            fig.savefig(f"{fname}_grav_anomalies.png")

    except Exception as e:
        logger.error(e)
        logger.error("Failed to plot %s", name)


def polygons_to_points(
    polygons: gpd.GeoDataFrame,
    spacing: float = 100,
) -> pd.DataFrame:
    """
    Convert a geodataframe of polygons to a grid of points with a specified spacing.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        Geodataframe of polygons to convert to points
    spacing : float, optional
        Spacing between points, by default 100

    Returns
    -------
    pd.DataFrame
        Dataframe of points with columns "easting", "northing", and "geometry"
    """
    points_list = []
    for _, row in polygons.iterrows():
        points_list.append(polygon_to_points(row, spacing=spacing))

    return pd.concat(points_list)


def polygon_to_points(
    polygon: gpd.GeoSeries,
    spacing: float = 100,
) -> pd.DataFrame:
    """
    Convert a polygon shapefile to a grid of points with a specified spacing. Also
    include the vertices of the polygon.

    Parameters
    ----------
    polygon : gpd.GeoSeries
        Polygon shapefile to convert to points
    spacing : float, optional
        Spacing between points, by default 100

    Returns
    -------
    pd.DataFrame
        Dataframe of points with columns "easting", "northing", and "geometry"
    """

    # get bounds of polygon
    bounds = polygon.geometry.bounds

    # create grid of points
    x_coords = np.arange(bounds[0], bounds[2], spacing)
    y_coords = np.arange(bounds[1], bounds[3], spacing)
    points = [shapely.geometry.Point(x, y) for x in x_coords for y in y_coords]

    # Filter points within the polygon
    points_in_polygon = [point for point in points if polygon.geometry.contains(point)]

    # add vertices of polygon
    points_in_polygon.extend(
        [shapely.geometry.Point(x, y) for (x, y) in polygon.geometry.exterior.coords]
    )

    # add points along edges
    points_in_polygon.extend(
        [
            shapely.geometry.Point(x, y)
            for (x, y) in polygon.geometry.segmentize(spacing).exterior.coords
        ]
    )

    # convert to geodataframe
    points_gdf = gpd.GeoDataFrame(geometry=points_in_polygon)

    # add easting and northing columns
    points_gdf["easting"] = [p.x for p in points_gdf.geometry]
    points_gdf["northing"] = [p.y for p in points_gdf.geometry]

    return points_gdf


def plot_ice_shelf_info(
    ice_shelf: gpd.GeoSeries,
    grav_df: pd.DataFrame | None = None,
    constraints_df: pd.DataFrame | None = None,
    min_dist: xr.DataArray | None = None,
    fig_path: str | None = None,
):
    name = ice_shelf.NAME.iloc[0]

    region = vd.get_region(
        (
            min_dist.easting.values,
            min_dist.northing.values,
        )
    )
    grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()
    grav_grid = grav_grid.where(grav_grid.ice_shelf_mask == True, np.nan)  # noqa: E712 # pylint: disable=singleton-comparison

    anoms = [
        "gravity_disturbance",
        "topo_free_disturbance",
        "starting_gravity",
        "reg",
        "res",
    ]
    anom_titles = [
        "Gravity disturbance",
        "Topo-free disturbance (misfit)",
        "Starting gravity",
        "Regional misfit",
        "Residual misfit",
    ]
    grids = [grav_grid[a] for a in anoms]

    cmaps = ["viridis"] * (len(anoms) - 1) + ["balance+h0"]
    insets = [False] * (len(grids) - 1) + [True]
    cbar_labels = [
        f"stdev: {round(grav_df[grav_df.ice_shelf_mask==True][a].std(),0)} mGal"  # pylint: disable=singleton-comparison # noqa: E712
        for a in anoms
    ]
    point_sets = [None, None, None, None, constraints_df]
    scalebars = [False] * (len(grids) - 1) + [True]
    titles = anom_titles

    # add plotting elements for mindist
    cmaps.insert(0, "dense")
    cbar_labels.insert(0, f"median; {round(np.nanmedian(min_dist),2)} km")
    insets.insert(0, False)
    scalebars.insert(0, False)
    point_sets.insert(0, constraints_df)
    titles.insert(0, "Constraint proximity")
    grids.insert(0, min_dist)

    # add plotting elements for uncertainty if available
    if np.isnan(grav_grid.error.values).all():
        logger.error("No uncertainty data for %s", name)
    else:
        cmaps.insert(1, "thermal")
        cbar_labels.insert(
            1, f"uncertainty; RMSE: {round(utils.rmse(grav_grid.error))} mGal"
        )
        insets.insert(1, False)
        scalebars.insert(1, False)
        point_sets.insert(1, None)
        titles.insert(1, "AntGG gravity uncertainty")
        grids.insert(1, grav_grid.error)

    try:
        fig = maps.subplots(
            grids,
            fig_height=15,
            region=region,
            fig_title=f"{name.replace("_", " ")} Ice Shelf",
            cmaps=cmaps,
            insets=insets,
            inset_width=0.6,
            inset_position="jTR+jTL+o-0/1.5",  # {15*.3}c/1.5c",
            robust=True,
            coast=True,
            coast_pen="1.2p,salmon",
            coast_version="measures-v2",  # default is depoorter-2013
            hist=True,
            cbar_labels=cbar_labels,
            titles=titles,
            point_sets=point_sets,
            cbar_font="18p,Helvetica,black",
            points_style="p1p",
            scalebars=scalebars,
            # scalebar_box="+gwhite",#@30+p0.5p,gray30,solid+r3p",
            scalebar_position="jTR+jTL+o0.5c/0.5c",
            # simple_basemap=True,
            # simple_basemap_version="measures-v2",
            modis_basemap=True,
            modis_version="125m",
            modis_transparency=60,
            yshift_extra=1,
        )
        fig.show()
        if fig_path is not None:
            fig.savefig(f"{fig_path}{name}_info.png")

    except Exception as e:
        logger.error(e)
        logger.error("Failed to plot %s", name)


def load_ice_shelf_info_single(
    ice_shelf: gpd.GeoSeries,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    name = gdf.NAME.iloc[0]

    try:
        min_dist = xr.open_dataarray(f"{file_path}{name}_min_dist.nc")
        gdf.loc[gdf.index, "median_constraint_distance"] = min_dist.median().values
        gdf.loc[gdf.index, "mean_constraint_distance"] = min_dist.mean().values
        gdf.loc[gdf.index, "max_constraint_distance"] = min_dist.max().values
        gdf.loc[gdf.index, "constraint_proximity_skewness"] = sp.stats.skew(
            min_dist.values.ravel(), nan_policy="omit"
        )

    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Failed to load minimum distances for %s", name)
        return gdf
    try:
        constraints_df = pd.read_csv(f"{file_path}{name}_constraints.csv.gz")
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Failed to load constraints for %s", name)
        constraints_df = None
        return gdf
    try:
        grav_df = pd.read_csv(f"{file_path}{name}_grav_anomalies.csv.gz")
    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Failed to load gravity data for %s", name)
        return gdf

    # sample the constraint distance grid into the gravity dataframe
    grav_df = profiles.sample_grids(
        grav_df,
        min_dist,
        sampled_name="constraint_distance",
        verbose="q",
    )

    grav_df["residual_constraint_proximity_ratio"] = grav_df.res / (
        1 / grav_df.constraint_distance
    )
    grav_df["regional_constraint_proximity_ratio"] = grav_df.reg / (
        1 / grav_df.constraint_distance
    )

    # subset data to ice shelf
    grav_df_subset = grav_df[grav_df.ice_shelf_mask == True]  # noqa: E712 # pylint: disable=singleton-comparison

    # calculate stats on following columns
    anoms = [
        "gravity_disturbance",
        "partial_topo_free_disturbance",
        "topo_free_disturbance",
        # "dist_topo_free_dist_ratio",
        "starting_gravity",
        # "gravity_anomaly",
        # "misfit",
        "reg",
        "res",
        "error",
        "residual_constraint_proximity_ratio",
        "regional_constraint_proximity_ratio",
    ]
    stats_df = pd.DataFrame(
        {
            "rms": [utils.rmse(grav_df_subset[a]) for a in anoms],
            # "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
            "stdev": [grav_df_subset[a].std() for a in anoms],
        },
        index=anoms,
    )

    for row_name, _ in stats_df.iterrows():
        for col_name in stats_df.columns:
            gdf.loc[gdf.index, f"{row_name}_{col_name}"] = stats_df.loc[row_name][
                col_name
            ]

    if plot:
        fig_path = file_path if save_plot else None

        plot_ice_shelf_info(
            gdf,
            grav_df=grav_df,
            constraints_df=constraints_df,
            min_dist=min_dist,
            fig_path=fig_path,
        )

    return gdf


def load_ice_shelf_info(
    ice_shelves: gpd.GeoSeries,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )

    shelves = []
    for _i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        shelves.append(
            load_ice_shelf_info_single(
                row,
                file_path=file_path,
                plot=plot,
                save_plot=save_plot,
            )
        )

    return pd.concat(shelves)


def add_shelves_to_ensembles(
    x: str,
    y: str,
    ice_shelves: gpd.GeoDataFrame,
    shelves_to_label: list[str] | None = None,
    ax: typing.Any | None = None,
    col_to_add_to_label: str | None = None,
    legend: bool = True,
    legend_cols: int = 2,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.3, 0.5),
    seperate_inverted_shelves: bool = True,
    fontsize: float = 8,
):
    gdf = ice_shelves.copy()

    if shelves_to_label is None:
        shelves_to_label = gdf.NAME.unique().tolist()

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        xlims = None
        ylims = None
    else:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

    if xlims is not None and ylims is not None:
        gdf[x] = np.where(gdf[x] < min(xlims), min(xlims), gdf[x])
        gdf[x] = np.where(gdf[x] > max(xlims), max(xlims), gdf[x])
        gdf[y] = np.where(gdf[y] < min(ylims), min(ylims), gdf[y])
        gdf[y] = np.where(gdf[y] > max(ylims), max(ylims), gdf[y])

    texts = []

    for ind, row in gdf.iterrows():
        if col_to_add_to_label is not None:
            if isinstance(col_to_add_to_label, list | tuple):
                vals = [f"{round(row[x])}" for x in col_to_add_to_label]
                add_to_label = f": {"/".join(vals)} m"
            elif isinstance(col_to_add_to_label, str):
                add_to_label = f": {round(row[col_to_add_to_label])} m"
        else:
            add_to_label = ""
        if seperate_inverted_shelves:
            # plot inverted shelves as red stars and red labels
            if row.NAME in inverted_shelves:
                ax.scatter(
                    row[x],
                    row[y],
                    color="r",
                    marker="*",
                    s=60,
                    linewidths=0.8,
                    edgecolor="white",
                    label=f"{ind+1}) {row.NAME.replace('_', ' ')}{add_to_label}",
                    clip_on=False,
                    zorder=10,
                )
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        f"{ind+1}",
                        fontsize=fontsize + 2,
                        color="r",
                        fontweight="normal",
                        path_effects=[
                            patheffects.withStroke(linewidth=2, foreground="white")
                        ],
                    )
                )
            else:
                # plot other shelves as black dots and black labels
                if row.NAME in shelves_to_label:
                    ax.scatter(
                        row[x],
                        row[y],
                        color="black",
                        s=6,
                        label=f"{ind+1}) {row.NAME.replace('_', ' ')}{add_to_label}",
                        clip_on=False,
                        zorder=10,
                    )
                    texts.append(
                        ax.text(
                            row[x],
                            row[y],
                            f"{ind+1}",
                            fontsize=fontsize,
                            color="black",
                            fontweight="normal",
                            path_effects=[
                                patheffects.withStroke(
                                    linewidth=1.5, foreground="white"
                                )
                            ],
                        )
                    )
                else:
                    ax.scatter(
                        row[x],
                        row[y],
                        color="black",
                        s=6,
                        clip_on=False,
                        zorder=10,
                    )
        else:
            # plot other shelves as black dots and black labels
            if row.NAME in shelves_to_label:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=6,
                    label=f"{ind+1}) {row.NAME.replace('_', ' ')}{add_to_label}",
                    clip_on=False,
                    zorder=10,
                )
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        f"{ind+1}",
                        fontsize=fontsize,
                        color="black",
                        fontweight="normal",
                        path_effects=[
                            patheffects.withStroke(linewidth=1.5, foreground="white")
                        ],
                    )
                )
            else:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=6,
                    clip_on=False,
                    zorder=10,
                )
    if adjust_text is None:
        logger.error("adjust_text not found, please install adjustText")
        return
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "-", "color": "k", "lw": 0.8},
        ax=ax,
        expand=(1.2, 1.2),
    )

    if legend:
        leg = ax.legend(
            loc=legend_loc,
            title="Ice Shelves",
            title_fontproperties={"size": 14},
            bbox_to_anchor=legend_bbox_to_anchor,
            columnspacing=0,
            markerscale=0,
            fontsize=10,
            ncol=legend_cols,
            # labelspacing=0.4,
            handlelength=0,
            # handletextpad=0.2,
            borderpad=0.2,
            # borderaxespad=0.2,
        )
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            text.set_color(handle.get_facecolor()[0])


def ensemble_scatterplot(
    x: str,
    y: str,
    ice_shelves: gpd.GeoDataFrame,
    figsize=(5, 5),
    label_shelves: bool = True,
    shelves_to_label: list[str] | None = None,
    col_to_add_to_label: str | None = None,
    legend: bool = True,
    legend_cols: int = 2,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: tuple[float, float] = (1.3, 0.5),
    legend_borderaxespad: float = 0,
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    logx=False,
    logy=False,
):
    gdf = ice_shelves.copy()

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if shelves_to_label is None:
        shelves_to_label = gdf.NAME.unique().tolist()

    if xlims is not None:
        gdf = gdf[(gdf[x] >= xlims[0]) & (gdf[x] <= xlims[1])]
    if ylims is not None:
        gdf = gdf[(gdf[y] >= ylims[0]) & (gdf[y] <= ylims[1])]

    texts = []
    for ind, (_i, row) in enumerate(gdf.iterrows()):
        if col_to_add_to_label is not None:
            add_to_label = f": {round(row[col_to_add_to_label])} m"
        else:
            add_to_label = ""
        # plot inverted shelves as red stars and red labels
        if row.NAME in inverted_shelves:
            ax.scatter(
                row[x],
                row[y],
                color="red",
                marker="*",
                s=12,
                label=f"{ind+1} {row.NAME.replace('_', ' ')}{add_to_label}",
            )
            if label_shelves:
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        f"{ind+1}",
                        fontsize=8,
                        color="red",
                        fontweight="normal",
                        path_effects=[
                            patheffects.withStroke(linewidth=1.5, foreground="white")
                        ],
                    )
                )
        else:
            # plot other shelves as black dots and black labels
            if row.NAME in shelves_to_label:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                    label=f"{ind+1} {row.NAME.replace('_', ' ')}{add_to_label}",
                )
                if label_shelves:
                    texts.append(
                        ax.text(
                            row[x],
                            row[y],
                            f"{ind+1}",
                            fontsize=8,
                            color="black",
                            fontweight="normal",
                            path_effects=[
                                patheffects.withStroke(
                                    linewidth=1.5, foreground="white"
                                )
                            ],
                        )
                    )
            else:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                )

    # ax.set_xlim(xlims)
    # ax.set_ylim(ylims)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    if adjust_text is None:
        logger.error("adjust_text not found, please install adjustText")
        return None
    adjust_text(
        texts,
        arrowprops={"arrowstyle": "-", "color": "k", "lw": 0.8},
        ax=ax,
        expand=(1.2, 1.2),
    )

    if label_shelves and legend:
        leg = ax.legend(
            loc=legend_loc,
            title="Ice Shelves",
            bbox_to_anchor=legend_bbox_to_anchor,
            borderaxespad=legend_borderaxespad,
            markerscale=0,
            fontsize=8,
            ncol=legend_cols,
        )
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            text.set_color(handle.get_facecolor()[0])

    return fig
