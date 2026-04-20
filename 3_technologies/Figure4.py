
## Figure 4: Flux Model 

import os
import matplotlib.pyplot as plt
import numpy as np

from Flux import build_flux_hypothesis

GEV_PER_PEV = 1e6

plt.rcParams.update(
    {
        "font.size": 20,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
    }
)

COLOR_SCHEMES = {
    "mid": {"c68": "#1e90ff", "a68": 0.25, "c95": "#add8e6", "a95": 0.35},
    "low": {"c68": "#ff69b4", "a68": 0.25, "c95": "#ffb6c1", "a95": 0.35},
    "high": {"c68": "#228b22", "a68": 0.25, "c95": "#90ee90", "a95": 0.35},
}


def _e2phi(energy_gev: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.asarray(energy_gev, dtype=float) ** 2 * np.asarray(phi, dtype=float)


def _smooth_loglog_curve(x: np.ndarray, y: np.ndarray, n_pts: int = 40) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_smooth = np.logspace(np.log10(x[0]), np.log10(x[-1]), n_pts)
    y_smooth = np.exp(np.interp(np.log(x_smooth), np.log(x), np.log(y)))
    return x_smooth, y_smooth


def _plot_horizontal_range(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min,
    y_max,
    color: str,
    label: str,
    x_center=None,
) -> tuple[float, float]:
    x_mid = np.sqrt(x_min * x_max) if x_center is None else float(x_center)
    y_mid = np.sqrt(y_min * y_max)
    ax.hlines(
        y=y_mid,
        xmin=x_min,
        xmax=x_max,
        colors=color,
        linewidth=4.2,
        label=label,
        zorder=6,
    )
    ax.vlines(
        x=x_mid,
        ymin=y_min,
        ymax=y_max,
        colors=color,
        linewidth=4.2,
        zorder=6,
    )
    return x_mid, y_mid

 

def main() -> None:
    e_low, phi_low = build_flux_hypothesis("low")
    e_mid, phi_mid = build_flux_hypothesis("mid")
    e_high, phi_high = build_flux_hypothesis("high")

    e2_low = _e2phi(e_low, phi_low)
    e2_mid = _e2phi(e_mid, phi_mid)
    e2_high = _e2phi(e_high, phi_high)
    e_low_s, e2_low_s = _smooth_loglog_curve(e_low, e2_low)
    e_mid_s, e2_mid_s = _smooth_loglog_curve(e_mid, e2_mid)
    e_high_s, e2_high_s = _smooth_loglog_curve(e_high, e2_high)
    e_low_pev = e_low_s / GEV_PER_PEV
    e_mid_pev = e_mid_s / GEV_PER_PEV
    e_high_pev = e_high_s / GEV_PER_PEV

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)

   
    ax.loglog(
        e_low_pev,
        e2_low_s,
        color=COLOR_SCHEMES["low"]["c95"],
        linewidth=3.0,
        linestyle="--",
        label="Low flux",
    )

    ax.loglog(
        e_mid_pev,
        e2_mid_s,
        color=COLOR_SCHEMES["mid"]["c95"],
        linewidth=3.0,
        label="Mid flux",
    )

    ax.loglog(
        e_high_pev,
        e2_high_s,
        color=COLOR_SCHEMES["high"]["c95"],
        linewidth=3.0,
        linestyle="-.",
        label="High flux",
    )

    km3_y_min = (2e-8) / 3.0
    km3_y_max = (2e-7) / 3.0
    km3_x_mid, km3_y_mid = _plot_horizontal_range(
        ax,
        x_min=8e7 / GEV_PER_PEV,
        x_max=2e9 / GEV_PER_PEV,
        y_min=km3_y_min,
        y_max=km3_y_max,
        color="#008080",
        label="_nolegend_",
        x_center=2e8 / GEV_PER_PEV,
    )
    anita_y_min = (1.5e-6) / 3.0
    anita_y_max = (1.0e-5) / 3.0

    anita_x_mid, anita_y_mid = _plot_horizontal_range(
        ax,
        x_min=5e9 / GEV_PER_PEV,
        x_max=3e10 / GEV_PER_PEV,
        y_min=anita_y_min,
        y_max=anita_y_max,
        color="#ff8c00",
        label="_nolegend_",
        x_center=1e10 / GEV_PER_PEV,
    )
    ax.text(
        km3_x_mid * 1.08,
        km3_y_mid * 1.05,
        r"KM3Net $\nu_\mu$",
        color="#008080",
        ha="left",
        va="bottom",
        fontsize=18,
    )
    ax.text(
        anita_x_mid * 0.92,
        anita_y_mid * 0.88,
        r"ANITA-IV $\nu_\tau$",
        color="#ff8c00",
        ha="right",
        va="top",
        fontsize=18,
    )

    ax.set_xlabel("Energy (PeV)")
    ax.set_xlim(1e4 / GEV_PER_PEV, 8e10 / GEV_PER_PEV)
    ax.set_ylim(1e-10, 5e-6)
    ax.set_ylabel(r"$E^2 \Phi$ (GeV cm$^{-2}$ s$^{-1}$ sr$^{-1}$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower left", frameon=True)

    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "MC_outputs",
        "figure10.png",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
