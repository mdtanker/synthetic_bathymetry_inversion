from __future__ import annotations

import string

import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from invert4geom import uncertainty
from polartoolkit import utils
import plotly.graph_objects as go
from kneebow.rotor import Rotor

sns.set_theme()


def plot_2var_ensemble(
    df,
    x,
    y,
    figsize=(9, 6),
    x_title=None,
    y_title=None,
    background="score",
    background_title=None,
    background_cmap=cmo.matter, # pylint: disable=no-member
    background_lims=None,
    background_cpt_lims=None,
    points_color=None,
    points_share_cmap=False,
    points_size=None,
    points_scaling=1,
    points_label=None,
    points_title=None,
    points_color_log=False,
    points_cmap=cmo.gray_r, # pylint: disable=no-member
    points_lims=None,
    points_edgecolor="black",
    background_color_log=False,
    background_robust=False,
    points_robust=False,
    plot_contours=None,
    contour_color="black",
    plot_title=None,
    logx=False,
    logy=False,
    flipx=False,
    colorbar: bool = True,
    colorbar_axes: tuple = (.95, 0.1, 0.05, .8),
    constrained_layout: bool =True,
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=constrained_layout)
    df = df.copy()

    ax.grid(which="major", visible=False)

    norm = mpl.colors.LogNorm() if background_color_log is True else None

    cmap = plt.get_cmap(background_cmap)

    if background_lims is None:
        background_lims = utils.get_min_max(df[background], robust=background_robust)
    else:
        cmap.set_under("g")

    grd = df.set_index([y, x]).to_xarray()[background]

    if background_cpt_lims is not None:
        background_lims = background_cpt_lims

    plot_background = grd.plot(
        ax=ax,
        cmap=cmap,
        vmin=background_lims[0],
        vmax=background_lims[1],
        norm=norm,
        # norm=mpl.colors.BoundaryNorm(
        #   np.linspace(background_lims[0],
        #   background_lims[1],
        #   10), cmap.N),
        edgecolors="w",
        linewidth=0.5,
        add_colorbar=False,
        # xticks=df[x].unique().round(-2),
        # yticks=df[y].unique().astype(int),
    )
    # ax.set_xticks(ax.get_xticks()[1:-1])

    if colorbar:
        cax = fig.add_axes(colorbar_axes)
        cbar = plt.colorbar(plot_background, extend="both", cax=cax)
        cbar.set_label(background_title)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    # x = df[x].unique()
    # y = df[y].unique()
    # plt.xticks(x[:-1]+0.5)
    # plt.yticks(y[:-1]+0.5)
    y_ticks = list(df[y].unique()[::2])  # .append(df[y].unique()[-1])
    y_ticks.append(df[y].unique()[-1])
    ax.set_yticks(y_ticks)

    # x_ticks = list(df[x].unique()[::2])#.append(df[y].unique()[-1])
    # x_ticks.append(df[x].unique()[-1])
    # ax.set_xlim(df[x].min(), df[x].max())
    # ax.set_xticks(x_ticks)

    if plot_contours is not None:
        contour = grd.plot.contour(
            levels=plot_contours,
            colors=contour_color,
            ax=ax,
            linewidths=3,
        )
        if colorbar:
            cbar.add_lines(contour)

    if (points_color is not None) or (points_size is not None):
        if points_color is None:
            points_color = "b"
        if points_size is None:
            points_size = 50

        points_cmap = plt.get_cmap(points_cmap)

        if points_share_cmap is True:
            points_cmap = cmap
            vmin = background_lims[0]
            vmax = background_lims[1]
        else:
            if points_lims is None:
                points_lims = utils.get_min_max(df[points_color], robust=points_robust)
            else:
                points_cmap.set_under("g")

            if points_color_log is True:
                norm = mpl.colors.LogNorm(
                    vmin=points_lims[0],
                    vmax=points_lims[1],
                )
                vmin = None
                vmax = None
            else:
                norm = None
                vmin = points_lims[0]
                vmax = points_lims[1]

        points = ax.scatter(
            df[x],
            df[y],
            s=points_size * points_scaling,
            c=df[points_color],
            cmap=points_cmap,
            zorder=10,
            edgecolors=points_edgecolor,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            label=points_label,
        )
        if isinstance(points_size, pd.Series):
            kw = {"prop": 'sizes', "num": 3, "func": lambda s: s / points_scaling}
            ax.legend(
                *points.legend_elements(**kw),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                title=points_size.name,
            )
        if (points_share_cmap is False) and (colorbar):
            cbar2 = fig.colorbar(points, extend="both")
            try:  # noqa: SIM105
                cbar2.set_label(points_color.name)
            except AttributeError:
                pass
            if points_title is None:
                points_title = points_label
            cbar2.set_label(points_title)
        # else:
        #     cbar2 = cbar

        if points_label is not None:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(1, 1),
            )

    if flipx:
        ax.invert_xaxis()

    # ax.ticklabel_format(useOffset=False, style='plain')

    if x_title is None:
        x_title = x
    if y_title is None:
        y_title = y
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    ax.set_title(plot_title, fontsize=16)

    return fig


def plot_ensemble_as_lines(
    results,
    x,
    y,
    groupby_col,
    figsize=(5, 3.5),
    x_lims=None,
    y_lims=None,
    x_label=None,
    y_label="Bathymetry RMSE (m)",
    cbar_label=None,
    markersize=7,
    logy=False,
    logx=False,
    trend_line=False,
    horizontal_line=None,
    horizontal_line_label=None,
    horizontal_line_label_loc="best",
    horizontal_line_color="gray",
    plot_title=None,
    slope_min_max=False,
    slope_min=False,
    slope_max=False,
    slope_mean=False,
    slope_decimals=3,
    trend_line_text_loc=(0.05, 0.95),
    flipx=False,
    colorbar: bool = True,
    ax=None,
    plot_elbows=False,
    plot_maximums=False,
    plot_minimums=False,
):
    sns.set_theme()

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=False)
    else:
        fig, _ = plt.subplots(figsize=figsize, constrained_layout=False)
        ax1 = ax

    grouped = results.groupby(groupby_col)

    norm = plt.Normalize(
        vmin=results[groupby_col].values.min(), vmax=results[groupby_col].values.max()
    )
    slopes = []
    lines = []
    for _, (name, group) in enumerate(grouped):
        ax1.plot(
            group[x],
            group[y],
            ".-",
            markersize=markersize,
            color=plt.cm.viridis(norm(name)), # pylint: disable=no-member
        )
        if trend_line:
            z = np.polyfit(group[x], group[y], 1)
            slopes.append(z[0])
            lines.append(np.poly1d(z)(results[x]))
        if plot_elbows:


            rotor = Rotor()

            df = group[[x, y]].copy()
            # df = df[df[x]<5]

            # df = pd.concat([
            #     df,
            #     pd.DataFrame({x: np.arange(df[x].min(), df[x].max(), 1)})]
            # ).drop_duplicates(subset=x)
            # df = df.sort_values(x).reset_index(drop=True)

            # df = synthetic.scipy_interp1d(
            #     df,
            #     to_interp=y,
            #     interp_on=x,
            #     method="cubic",
            # )

            df = df.sort_values(x).reset_index(drop=True)

            rotor.fit_rotate(df)

            elbow_ind = rotor.get_elbow_index()
            ax1.scatter(
                x=df[x].iloc[elbow_ind],
                y=df[y].iloc[elbow_ind],
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                color=plt.cm.viridis(norm(name)), # pylint: disable=no-member
                s=60,
                zorder=20,
            )
            print(f"Elbow x value: {df[x].iloc[elbow_ind]}")

        if plot_maximums:
            df = group[[x, y]].copy()

            # df = pd.concat([
            #     df,
            #     pd.DataFrame({x: np.arange(df[x].min(), df[x].max(), 1)})]
            # ).drop_duplicates(subset=x)
            # df = df.sort_values(x).reset_index(drop=True)

            # df = synthetic.scipy_interp1d(
            #     df,
            #     to_interp=y,
            #     interp_on=x,
            #     method="cubic",
            # )

            df = df.sort_values(x).reset_index(drop=True)

            max_ind = df[y].idxmax()
            ax1.scatter(
                x=df[x].iloc[max_ind],
                y=df[y].iloc[max_ind],
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                color=plt.cm.viridis(norm(name)), # pylint: disable=no-member
                s=60,
                zorder=20,
            )
            print(f"Max y value at x value: {df[x].iloc[max_ind]}")

        if plot_minimums:
            df = group[[x, y]].copy()
            # df = df[df[x]<5]

            # df = pd.concat([
            #     df,
            #     pd.DataFrame({x: np.arange(df[x].min(), df[x].max(), 1)})]
            # ).drop_duplicates(subset=x)
            # df = df.sort_values(x).reset_index(drop=True)

            # df = synthetic.scipy_interp1d(
            #     df,
            #     to_interp=y,
            #     interp_on=x,
            #     method="cubic",
            # )

            df = df.sort_values(x).reset_index(drop=True)

            min_ind = df[y].idxmin()
            ax1.scatter(
                x=df[x].iloc[min_ind],
                y=df[y].iloc[min_ind],
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                color=plt.cm.viridis(norm(name)), # pylint: disable=no-member
                s=60,
                zorder=20,
            )
            print(f"Min y value at x value: {df[x].iloc[min_ind]}")

    if trend_line:
        if slope_min_max:
            text = f"$min\ slope={round(min(slopes),slope_decimals)}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmin(slopes)], "r", lw=1)

            text = f"$max\ slope={round(max(slopes),slope_decimals)}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1] - 0.1,
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmax(slopes)], "r", lw=1)
        elif slope_min:
            text = f"$min\ slope={round(min(slopes),slope_decimals)}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmin(slopes)], "r", lw=1)
        elif slope_max:
            text = f"$max\ slope={round(max(slopes),slope_decimals)}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmax(slopes)], "r", lw=1)

        if slope_mean:
            # text = rf"$mean\ slope={round(np.median(slopes),slope_decimals)}$"
            # plt.gca().text(
            #     trend_line_text_loc[0],
            #     trend_line_text_loc[1] - 0.1,
            #     text,
            #     transform=plt.gca().transAxes,
            #     fontsize=10,
            #     verticalalignment="top",
            # )
            # ax1.plot(results[x], lines[np.argsort(slopes)[len(slopes) // 2]], "r", lw=1)

        # else:
            z = np.polyfit(results[x], results[y], 1)
            y_hat = np.poly1d(z)(results[x])

            ax1.plot(results[x], y_hat, "r", lw=1)
            text = f"$slope={round(z[0], slope_decimals)}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            linewidth=2,
            color=horizontal_line_color,
            linestyle="dashed",
            label=horizontal_line_label,
        )

    ax1.set_xlabel(
        x_label,
    )
    # ax1.set_xticks(list(ax1.get_xticks()) + list(ax1.get_xlim()))

    if x_lims is not None:
        ax1.set_xlim(x_lims)
    if y_lims is not None:
        ax1.set_ylim(y_lims)
    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel(y_label)

    if (horizontal_line is not None) & (horizontal_line_label is not None):
        plt.legend(loc=horizontal_line_label_loc)

    if flipx:
        ax1.invert_xaxis()

    if colorbar:
        # pass
        cax = fig.add_axes([0.93, 0.1, 0.05, 0.8])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label)

    plt.title(plot_title)
    # plt.tight_layout()

    return fig


def plot_1var_ensemble(
    df,
    x,
    y,
    title,
    xlabel,
    ylabel,
    highlight_points=None,
    horizontal_line=None,
    horizontal_line_label=None,
    starting_error=None,
    logy=False,
    logx=False,
):
    sns.set_theme()

    df = df.copy()

    df = df.sort_values(x)

    _fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title(title)

    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            linewidth=2,
            color="gray",
            linestyle="dashed",
            label=horizontal_line_label,
        )

    ax1.plot(df[x], df[y], "bd-", markersize=7, label="inverted")
    ax1.set_xlabel(
        xlabel,
        # color="b",
    )

    if starting_error is not None:
        ax1.plot(
            df[x],
            df[starting_error],
            "g.-",
            markersize=10,
            label="starting",
            zorder=1,
        )

    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel(ylabel)
    # ax1.tick_params(axis="x", colors='b', which="both")
    ax1.set_zorder(2)

    if highlight_points is not None:
        for i, ind in enumerate(highlight_points):
            plt.plot(
                df[x].loc[ind],
                df[y].loc[ind],
                "s",
                markersize=12,
                color="b",
                zorder=3,
            )
            plt.annotate(
                string.ascii_lowercase[i + 1],
                (df[x].loc[ind], df[y].loc[ind]),
                fontsize=15,
                color="white",
                ha="center",
                va="center",
                zorder=4,
            )

    plt.legend(loc="best")
    plt.tight_layout()


def uncert_plots(
    results,
    inversion_region,
    bathymetry,
    deterministic_bathymetry=None,
    constraint_points=None,
    weight_by=None,
):
    if (weight_by == "constraints") & (constraint_points is None):
        msg = "must provide constraint_points if weighting by constraints"
        raise ValueError(msg)

    stats_ds = uncertainty.merged_stats(
        results=results,
        plot=True,
        constraints_df=constraint_points,
        weight_by=weight_by,
        region=inversion_region,
    )

    try:
        mean = stats_ds.weighted_mean
        stdev = stats_ds.weighted_stdev
    except AttributeError:
        mean = stats_ds.z_mean
        stdev = stats_ds.z_stdev

    _ = utils.grd_compare(
        bathymetry,
        mean,
        region=inversion_region,
        plot=True,
        grid1_name="True topography",
        grid2_name="Stochastic mean inverted topography",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        grounding_line=False,
        reverse_cpt=True,
        cmap="rain",
        points=constraint_points,
        points_style="x.3c",
    )
    _ = utils.grd_compare(
        np.abs(bathymetry - mean),
        stdev,
        region=inversion_region,
        plot=True,
        grid1_name="Stochastic error",
        grid2_name="Stochastic uncertainty",
        cmap="thermal",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        grounding_line=False,
        points=constraint_points,
        points_style="x.3c",
        points_fill="white",
    )

    if deterministic_bathymetry is not None:
        _ = utils.grd_compare(
            np.abs(bathymetry - deterministic_bathymetry),
            stdev,
            region=inversion_region,
            plot=True,
            grid1_name="Deterministic error",
            grid2_name="Stochastic uncertainty",
            cmap="thermal",
            robust=True,
            hist=True,
            inset=False,
            verbose="q",
            title="difference",
            grounding_line=False,
            points=constraint_points,
            points_style="x.3c",
            points_fill="white",
        )

    return stats_ds


def plotly_profiles(
    data,
    y: tuple[str],
    x="dist_along_line",
    y_axes=None,
    xlims=None,
    ylims=None,
    **kwargs,
):
    """
    plot data profiles with plotly
    currently only allows 3 separate y axes, set with "y_axes", starting with 1
    """
    df = data.copy()

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    y_axes = ["" for _ in y] if y_axes is None else [str(x) for x in y_axes]
    assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
    # convert y axes to plotly expected format: "y", "y2", "y3" ...
    y_axes = [s.replace("1", "") for s in y_axes]
    y_axes = [f"y{x}" for x in y_axes]

    # lim x and y ranges
    if xlims is not None:
        df = df[df[x].between(*xlims)]
    if ylims is not None:
        df = df[df[y].between(*ylims)]

    # set plotting mode
    modes = kwargs.get("modes")
    if modes is None:
        modes = ["markers" for _ in y]

    # set marker properties
    marker_sizes = kwargs.get("marker_sizes")
    marker_symbols = kwargs.get("marker_symbols")
    if marker_sizes is None:
        marker_sizes = [2 for _ in y]
    if marker_symbols is None:
        marker_symbols = ["circle" for _ in y]

    fig = go.Figure()

    # iterate through data columns
    for i, col in enumerate(y):
        fig.add_trace(
            go.Scatter(
                mode=modes[i],
                x=df[x],
                y=df[col],
                name=col,
                marker_size=marker_sizes[i],
                marker_symbol=marker_symbols[i],
                yaxis=y_axes[i],
            )
        )

    unique_axes = len(pd.Series(y_axes).unique())

    if unique_axes >= 1:
        y_axes_args = {"yaxis":{"title": y[y_axes.index('y')]}}
        x_domain = [0, 1]
    if unique_axes >= 2:
        y_axes_args["yaxis2"] = { # pylint: disable=possibly-used-before-assignment
            "title":y[y_axes.index("y2")], "overlaying":"y", "side":"right"
        }
        x_domain = [0, 1]
    if unique_axes >= 3:
        y_axes_args["yaxis3"] = { # pylint: disable=possibly-used-before-assignment
            "title":y[y_axes.index('y3')],
            "anchor":"free",
            "overlaying":"y",
        }
        x_domain = [0.15, 1]
    else:
        pass

    fig.update_layout(
        title_text=kwargs.get("title"),
        xaxis={"title": x, "domain": x_domain}, # pylint: disable=possibly-used-before-assignment
        **y_axes_args,
    )

    return fig