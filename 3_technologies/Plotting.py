import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.tri import Triangulation

from Helpers import POINTS, _normalize_bary, bary_to_xy

# Plotting ------------------------------------------------------------------

def plot_three_ternaries_95pairs(
    ll_grids,
    flavors,
    fe_arr, fmu_arr, ftau_arr,
    captions=("In-ice Radio", "Earth-Skimming", "All Technologies Combined"),
    savepath="MC_outputs/triangles_3panel.png",
    interval=0.1,
    textbox_text=None,
):

    plt.rcParams.update({
        "font.size": 20,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
    })

    _MARKERS = ["o", "s", "X", "D", "^", "v", "<", ">", "P", "h"]
    _marker_state = {"map": {}, "idx": 0}

    def _marker_for(name: str) -> str:
        if name not in _marker_state["map"]:
            if _marker_state["idx"] >= len(_MARKERS):
                raise ValueError(
                    f"Ran out of unique markers (only {len(_MARKERS)} available)"
                )
            _marker_state["map"][name] = _MARKERS[_marker_state["idx"]]
            _marker_state["idx"] += 1
        return _marker_state["map"][name]

    def _fmt_label(name: str, fe: float, fmu: float, ftau: float) -> str:
        return rf"{name}"

    def _prep_data(ll_grid):
        best_idx = int(np.nanargmax(ll_grid))
        best_fe, best_fmu, best_ftau = flavors[best_idx]
        x, y = bary_to_xy(fe_arr, fmu_arr, ftau_arr)
        m = np.isfinite(ll_grid) & np.isfinite(x) & np.isfinite(y)
        x_plot, y_plot = x[m], y[m]
        z_plot = ll_grid[m] - np.nanmax(ll_grid)
        return (best_fe, best_fmu, best_ftau), x_plot, y_plot, z_plot

    def _triangle_frame(ax):
        h = np.sqrt(3)/2.0
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
        ax.plot(verts[:, 0], verts[:, 1], lw=1.8, color="black", zorder=5)

        ax.text(0.05, -0.015, r"$f_\tau=1$", ha="right", va="top", fontsize=20, clip_on=False)
        ax.text(0.95, -0.020, r"$f_e=1$",   ha="left",  va="top", fontsize=20, clip_on=False)
        ax.text(0.50,  h+0.025, r"$f_\mu=1$", ha="center", va="bottom", fontsize=20, clip_on=False)

        edge_bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95)
        ax.text(0.10, 0.50, r"$f_\tau$", rotation=60,  ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)
        ax.text(0.90, 0.50, r"$f_\mu$", rotation=-60, ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)
        ax.text(0.50, -0.15, r"$f_e$",                 ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.12, h + 0.06)
        ax.axis("off")

    def _edge_ticks(ax, step=0.2, ticklen=0.035, lab=True, tick_labelsize=16):
        h = np.sqrt(3)/2.0
        v_LHS = np.array([0.5, h])
        v_RHS = np.array([-0.5, h])

        def seg_from_dir(dir_vec, length):
            d = dir_vec / np.linalg.norm(dir_vec) * length
            return d[0], d[1]

        dx_b, dy_b = seg_from_dir(v_LHS, ticklen)
        for t in np.arange(step, 1.0, step):
            x0, y0 = t, 0.0
            ax.plot([x0, x0 + dx_b], [y0, y0 + dy_b], lw=1.3, color="black")
            if lab:
                ax.text(x0, -0.070, f"{t:.1f}", ha="center", va="top", fontsize=tick_labelsize, clip_on=False)

        for s in np.arange(step, 1.0, step):
            x0 = 1.0 - 0.5 * s
            y0 = h * s
            ax.plot([x0, x0 + ticklen], [y0, y0], lw=1.3, color="black")
            if lab:
                ax.text(x0 + 0.045, y0, f"{s:.1f}", ha="left", va="center", fontsize=tick_labelsize, clip_on=False)

        dx_l, dy_l = seg_from_dir(v_RHS, ticklen)
        for u in np.arange(step, 1.0, step):
            x0 = 0.5 * u
            y0 = h * u
            ax.plot([x0, x0 + dx_l], [y0, y0 + dy_l], lw=1.3, color="black")
            if lab:
                tau_tick = 1.0 - u
                ax.text(x0 - 0.045, y0, f"{tau_tick:.1f}", ha="right", va="center", fontsize=tick_labelsize, clip_on=False)

    def _draw_const_lines(ax, step=0.1, lw=1.1, alpha=0.2):
        h = np.sqrt(3)/2.0
        for fe in np.arange(step, 1.0, step):
            x0, y0 = fe, 0.0
            x1, y1 = fe + 0.5*(1 - fe), h*(1 - fe)
            ax.plot([x0, x1], [y0, y1], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)
        for fmu in np.arange(step, 1.0, step):
            x0, y0 = 0.5*fmu, h*fmu
            xb, yb = 1 - (1 - fmu), 0.0
            ax.plot([x0, xb], [y0, yb], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)
        for ft in np.arange(step, 1.0, step):
            x0, y0 = 1 - 0.5*ft, h*ft
            x1, y1 = 0.5*ft, h*ft
            ax.plot([x0, x1], [y0, y1], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)

    n_panels = 3
    if len(captions) != n_panels:
        raise ValueError("captions must contain exactly 3 panel labels.")
    fig_w = 6 * n_panels
    fig_h = 7.5
    fig, axs = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), constrained_layout=False)

    level95 = np.log(1.0 - 0.68)

    color_schemes = {
        "blue":  {"c68": "#1e90ff", "a68": 0.25, "c95": "#add8e6", "a95": 0.35},
        "pink":  {"c68": "#ff69b4", "a68": 0.25, "c95": "#ffb6c1", "a95": 0.35},
        "green": {"c68": "#228b22", "a68": 0.25, "c95": "#90ee90", "a95": 0.35},
        "red":   {"c68": "#c62828", "a68": 0.25, "c95": "#ef9a9a", "a95": 0.35},
    }

    handles_top = []
    handles_points = []

    if len(ll_grids) == 6:
        handles_top = [
            Patch(facecolor="gray", edgecolor="black", linestyle="--", linewidth=1.5,
                  alpha=0.20, label="Set A (95% CL)"),
            Patch(facecolor="gray", edgecolor="black", linestyle="-", linewidth=1.5,
                  alpha=0.35, label="Set B (95% CL)"),
            Patch(facecolor="white", edgecolor="none", alpha=0.0, label=""),
        ]
        grid_groups = [(0, 1), (2, 3), (4, 5)]
        panel_hues = ["green", "blue", "pink"]

        for ax, (i, j), scheme_key in zip(axs, grid_groups, panel_hues):
            scheme = color_schemes[scheme_key]
            _triangle_frame(ax)
            _draw_const_lines(ax, interval, lw=1.1, alpha=0.2)
            _edge_ticks(ax, step=0.2, ticklen=0.036, lab=True, tick_labelsize=16)

            alpha_pair = (scheme["a95"]*2.8, max(0.15, scheme["a95"] * 0.6))
            for k, ls, fill_alpha in zip((i, j), ("-", "--"), alpha_pair):
                (_, _, _), x_plot, y_plot, z_plot = _prep_data(ll_grids[k])
                tri = Triangulation(x_plot, y_plot)

                ax.tricontourf(
                    tri, z_plot,
                    levels=[level95, np.max(z_plot)],
                    colors=[scheme["c95"]],
                    alpha=fill_alpha,
                    zorder=0,
                )
                ax.tricontour(
                    tri, z_plot,
                    levels=[level95],
                    colors=[scheme["c68"]],
                    linewidths=2.0,
                    linestyles=[ls],
                    zorder=2
                )

    else:
        grid_groups = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
        per_panel_colors = ["red", "green", "blue"]
        per_panel_linestyles = ["-", "--", ":"]
        handles_top = [
            Patch(facecolor=color_schemes["red"]["c95"],  edgecolor=color_schemes["red"]["c68"],
                  linewidth=1.8, label="combined (68% CL)", alpha=color_schemes["red"]["a95"]),
            Patch(facecolor=color_schemes["green"]["c95"], edgecolor=color_schemes["green"]["c68"],
                  linewidth=1.8, label = r"$\nu_e$ CC only (68% CL)", alpha=color_schemes["green"]["a95"]),
            Patch(facecolor=color_schemes["blue"]["c95"], edgecolor=color_schemes["blue"]["c68"],
                  linewidth=1.8, label="multi-shower only (68% CL)", alpha=color_schemes["blue"]["a95"]),
        ]

        for ax, (i, j, k) in zip(axs, grid_groups):
            _triangle_frame(ax)
            _draw_const_lines(ax, interval, lw=1.1, alpha=0.2)
            _edge_ticks(ax, step=0.2, ticklen=0.036, lab=True, tick_labelsize=16)

            for idx, g in enumerate((i, j, k)):
                color_key = per_panel_colors[idx]
                scheme = color_schemes[color_key]
                ls = per_panel_linestyles[idx]

                (_, _, _), x_plot, y_plot, z_plot = _prep_data(ll_grids[g])
                tri = Triangulation(x_plot, y_plot)

                ax.tricontourf(
                    tri, z_plot,
                    levels=[level95, np.max(z_plot)],
                    colors=[scheme["c95"]],
                    alpha=scheme["a95"],
                    zorder=0,
                )
                ax.tricontour(
                    tri, z_plot,
                    levels=[level95],
                    colors=[scheme["c68"]],
                    linewidths=2.0,
                    linestyles=[ls],
                    zorder=2,
                )

    for name, (pfe, pfmu, pftau) in POINTS.items():
        pfe, pfmu, pftau = _normalize_bary(pfe, pfmu, pftau)
        px, py = bary_to_xy(np.array([pfe]), np.array([pfmu]), np.array([pftau]))
        mkr = _marker_for(name)
        for ax in axs:
            ax.scatter(px, py, s=90, facecolors="white", edgecolors="black",
                       marker=mkr, linewidths=1.1, zorder=6)
        handles_points.append(
            Line2D([0], [0], marker=mkr, markersize=10,
                   markerfacecolor='white', markeredgecolor='black',
                   linestyle='None', label=_fmt_label(name, pfe, pfmu, pftau))
        )

    handles_all = handles_top + handles_points

    fig.legend(
        handles=handles_all,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=True, fancybox=True, framealpha=0.98,
        edgecolor="black", facecolor="white",
    )

    for ax, caption in zip(axs, captions):
        ax.text(
            0.5, -0.12, caption,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=22,
            clip_on=False,
        )

    if textbox_text is not None:
        fig.text(
            0.975, 0.87, str(textbox_text),
            ha="right", va="bottom",
            fontsize=18,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.95),
        )

    wspace = 0.16
    top = 0.86
    bottom = 0.18
    fig.subplots_adjust(top=top, bottom=bottom, left=0.04, right=0.98, wspace=wspace)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_three_ternaries(
    ll_grids,
    flavors,
    fe_arr, fmu_arr, ftau_arr,
    common_caption=None,
    savepath="MC_outputs/triangles_3panel.png",
    interval=0.1,
    captions=None,
    panel_captions=None,
    textbox_text=None,
):
    import os, numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.tri import Triangulation
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 20,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
    })

    _MARKERS = ["o", "s", "X", "D", "^", "v", "<", ">", "P", "h"]
    _marker_state = {"map": {}, "idx": 0}

    def _marker_for(name: str) -> str:
        if name not in _marker_state["map"]:
            if _marker_state["idx"] >= len(_MARKERS):
                raise ValueError(
                    f"Ran out of unique markers (only {len(_MARKERS)} available)"
                )
            _marker_state["map"][name] = _MARKERS[_marker_state["idx"]]
            _marker_state["idx"] += 1
        return _marker_state["map"][name]

    def _fmt_label(name: str, fe: float, fmu: float, ftau: float) -> str:
        return rf"{name}"

    def _prep_data(ll_grid):
        best_idx = int(np.nanargmax(ll_grid))
        best_fe, best_fmu, best_ftau = flavors[best_idx]
        x, y = bary_to_xy(fe_arr, fmu_arr, ftau_arr)
        m = np.isfinite(ll_grid) & np.isfinite(x) & np.isfinite(y)
        x_plot, y_plot = x[m], y[m]
        z_plot = ll_grid[m] - np.nanmax(ll_grid)
        return (best_fe, best_fmu, best_ftau), x_plot, y_plot, z_plot

    def _triangle_frame(ax):
        h = np.sqrt(3)/2.0
        verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h], [0.0, 0.0]])
        ax.plot(verts[:, 0], verts[:, 1], lw=1.8, color="black", zorder=5)

        ax.text(0.05, -0.015, r"$f_\tau=1$", ha="right", va="top", fontsize=20, clip_on=False)
        ax.text(0.95, -0.020, r"$f_e=1$",   ha="left",  va="top", fontsize=20, clip_on=False)
        ax.text(0.50,  h+0.025, r"$f_\mu=1$", ha="center", va="bottom", fontsize=20, clip_on=False)

        edge_bbox = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95)
        ax.text(0.10, 0.50, r"$f_\tau$", rotation=60,  ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)
        ax.text(0.90, 0.50, r"$f_\mu$", rotation=-60, ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)
        ax.text(0.50, -0.15, r"$f_e$",                 ha="center", va="center", fontsize=30, bbox=edge_bbox, clip_on=False)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.12, h + 0.06)
        ax.axis("off")

    def _edge_ticks(ax, step=0.2, ticklen=0.035, lab=True, tick_labelsize=16):
        h = np.sqrt(3)/2.0
        v_LHS = np.array([0.5, h])
        v_RHS = np.array([-0.5, h])

        def seg_from_dir(dir_vec, length):
            d = dir_vec / np.linalg.norm(dir_vec) * length
            return d[0], d[1]

        dx_b, dy_b = seg_from_dir(v_LHS, ticklen)
        for t in np.arange(step, 1.0, step):
            x0, y0 = t, 0.0
            ax.plot([x0, x0 + dx_b], [y0, y0 + dy_b], lw=1.3, color="black")
            if lab:
                ax.text(x0, -0.070, f"{t:.1f}", ha="center", va="top", fontsize=tick_labelsize, clip_on=False)

        for s in np.arange(step, 1.0, step):
            x0 = 1.0 - 0.5 * s
            y0 = h * s
            ax.plot([x0, x0 + ticklen], [y0, y0], lw=1.3, color="black")
            if lab:
                ax.text(x0 + 0.045, y0, f"{s:.1f}", ha="left", va="center", fontsize=tick_labelsize, clip_on=False)

        dx_l, dy_l = seg_from_dir(v_RHS, ticklen)
        # Keep tau-edge ticks fixed at 0.2, 0.4, 0.6, 0.8.
        for u in (0.2, 0.4, 0.6, 0.8):
            x0 = 0.5 * u
            y0 = h * u
            ax.plot([x0, x0 + dx_l], [y0, y0 + dy_l], lw=1.3, color="black")
            if lab:
                tau_tick = 1.0 - u
                ax.text(x0 - 0.045, y0, f"{tau_tick:.1f}", ha="right", va="center", fontsize=tick_labelsize, clip_on=False)

    def _draw_const_lines(ax, step=0.1, lw=1.1, alpha=0.2):
        h = np.sqrt(3)/2.0
        for fe in np.arange(step, 1.0, step):
            x0, y0 = fe, 0.0
            x1, y1 = fe + 0.5*(1 - fe), h*(1 - fe)
            ax.plot([x0, x1], [y0, y1], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)
        for fmu in np.arange(step, 1.0, step):
            x0, y0 = 0.5*fmu, h*fmu
            xb, yb = 1 - (1 - fmu), 0.0
            ax.plot([x0, xb], [y0, yb], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)
        for ft in np.arange(step, 1.0, step):
            x0, y0 = 1 - 0.5*ft, h*ft
            x1, y1 = 0.5*ft, h*ft
            ax.plot([x0, x1], [y0, y1], lw=lw, ls="-", color="black", alpha=alpha, zorder=1)

    n_panels = len(ll_grids)
    if n_panels < 1:
        raise ValueError("ll_grids must contain at least one grid")
    if captions is not None and panel_captions is not None:
        raise ValueError("Use either captions or panel_captions, not both.")
    if captions is None:
        captions = panel_captions
    if captions is None:
        captions = [f"({i + 1})" for i in range(n_panels)]
    if len(captions) != n_panels:
        raise ValueError("captions length must match number of ll_grids")

    fig_w = 6 * n_panels 
    fig_h = 7.5
    fig, axs = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), constrained_layout=False)
    if n_panels == 1:
        axs = [axs] 

    levels_inc = np.sort([np.log(1.0 - 0.95), np.log(1.0 - 0.68)]) 
    color_scheme = {"c68": "#1e90ff", "a68": 0.25, "c95": "#add8e6", "a95": 0.35}
    #color_scheme= {"c68": "#ff69b4", "a68": 0.25, "c95": "#ffb6c1", "a95": 0.35} #pink
    # color_scheme= {"c68": "#228b22", "a68": 0.25, "c95": "#90ee90", "a95": 0.35} #green

    handles_top = [
        Patch(facecolor=color_scheme["c68"], edgecolor='none', alpha=color_scheme["a68"], label='68% CL'),
        Patch(facecolor=color_scheme["c95"], edgecolor='none', alpha=color_scheme["a95"], label='95% CL'),
        Line2D([0], [0], marker='*', markersize=16, markerfacecolor='C0', markeredgecolor='black',
               linestyle='None', label='Best fit'),
    ]
    handles_points = []

    for panel_idx, (ax, ll_grid) in enumerate(zip(axs, ll_grids)):
        (best_fe, best_fmu, best_ftau), x_plot, y_plot, z_plot = _prep_data(ll_grid)
        tri = Triangulation(x_plot, y_plot)

        _triangle_frame(ax)
        _draw_const_lines(ax, interval, lw=1.1, alpha=0.2)
        _edge_ticks(ax, step=0.2, ticklen=0.036, lab=True, tick_labelsize=16)

        ax.tricontourf(
            tri, z_plot,
            levels=[levels_inc[0], levels_inc[1]],
            colors=[color_scheme["c95"]], alpha=color_scheme["a95"], zorder=0
        )
        ax.tricontourf(
            tri, z_plot,
            levels=[levels_inc[1], np.max(z_plot)],
            colors=[color_scheme["c68"]], alpha=color_scheme["a68"], zorder=0
        )

        ax.tricontour(tri, z_plot, levels=levels_inc, colors='k', linewidths=1.9,
                      linestyles=['--', '-'])

        ax.text(
            0.5, -0.12, captions[panel_idx],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=24, 
            clip_on=False,
        )

    for name, (pfe, pfmu, pftau) in POINTS.items():
        pfe, pfmu, pftau = _normalize_bary(pfe, pfmu, pftau)
        px, py = bary_to_xy(np.array([pfe]), np.array([pfmu]), np.array([pftau]))
        mkr = _marker_for(name)
        for ax in axs:
            ax.scatter(px, py, s=90, facecolors="white", edgecolors="black",
                       marker=mkr, linewidths=1.1, zorder=6)
        handles_points.append(
            Line2D([0], [0], marker=mkr, markersize=10,
                   markerfacecolor='white', markeredgecolor='black',
                   linestyle='None', label=_fmt_label(name, pfe, pfmu, pftau))
        )

    handles_all = handles_top + handles_points

    fig.legend(
        handles=handles_all,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=True, fancybox=True, framealpha=0.98,
        edgecolor="black", facecolor="white",
    )

    if textbox_text is not None:
        fig.text(
            0.975, 0.87, str(textbox_text),
            ha="right", va="bottom",
            fontsize=18,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.95),
        )

    wspace = 0.16 if n_panels > 1 else 0.02
    top = 0.84
    bottom = 0.19 if common_caption else 0.14
    fig.subplots_adjust(top=top, bottom=bottom, left=0.04, right=0.98, wspace=wspace)

    if common_caption:
        fig.text(0.5, 0.05, common_caption, ha="center", va="center", fontsize=25)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
