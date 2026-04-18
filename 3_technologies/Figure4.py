import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import LinearTriInterpolator, Triangulation
from scipy.stats import chi2

import EventGeneration as eventgen
import Figure3 as fig3
import Helpers as helpers
from Flux import build_flux_hypothesis
from Helpers import bary_to_xy, generate_flavor_grid

T_EXPOSURE = 10 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi


def _apply_plotting_text_style():
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


def _compute_ll_panels_from_figure3(base_dir):
    """Recompute the three Figure3 likelihood panels using the same settings."""
    energies_mid, flux_mid = build_flux_hypothesis("mid")
    helpers.energies_mid = energies_mid
    helpers.fluxMid = flux_mid
    helpers.OMEGA = OMEGA
    helpers.T_EXPOSURE = T_EXPOSURE
    eventgen.energies_mid = energies_mid
    eventgen.fluxMid = flux_mid

    fe0, fmu0, ftau0 = 0.30, 0.36, 0.34
    fe0_p2, fmu0_p2, ftau0_p2 = 0.17, 0.45, 0.37

    flavors = generate_flavor_grid(step=fig3.GRID_STEP)

    decade_13_15 = np.logspace(4, 6, int((6 - 4) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_15_17 = np.logspace(6, 8, int((8 - 6) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_17_19 = np.logspace(8, 10, int((10 - 8) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)

    ll_panel1 = fig3.compute_mese21_ll_decade(
        flavors, decade_13_15, fe0, fmu0, ftau0, base_dir
    )

    ll_mese21_15_17 = fig3.compute_mese21_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, base_dir
    )
    ll_icgen2_15_17 = fig3.compute_icgen2_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, base_dir
    )
    ll_earth_15_17 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_radio_15_17 = fig3.compute_radio_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_panel2 = ll_mese21_15_17 + ll_icgen2_15_17 + ll_earth_15_17 + ll_radio_15_17

    ll_earth_17_19 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_radio_17_19 = fig3.compute_radio_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_panel3 = ll_earth_17_19 + ll_radio_17_19

    return flavors, [ll_panel1, ll_panel2, ll_panel3]


def _profile_ll_vs_fmu(flavors, ll_grid):
    """Profile likelihood in muon fraction f_mu by maximizing over f_e and f_tau."""
    fmu = np.array([f[1] for f in flavors], dtype=float)
    ll = np.asarray(ll_grid, dtype=float)

    valid = np.isfinite(fmu) & np.isfinite(ll)
    fmu = fmu[valid]
    ll = ll[valid]
    if fmu.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    fmu_uniq = np.unique(np.round(fmu, 10))
    ll_prof = np.full(fmu_uniq.shape, -np.inf, dtype=float)

    for i, val in enumerate(fmu_uniq):
        m = np.isclose(fmu, val, atol=1e-10)
        if np.any(m):
            ll_prof[i] = np.nanmax(ll[m])

    finite = np.isfinite(ll_prof)
    return fmu_uniq[finite], ll_prof[finite]


def _pdf_from_ll(x, ll):
    z = np.asarray(ll, dtype=float)
    z = z - np.nanmax(z)
    w = np.exp(z)
    area = np.trapezoid(w, np.asarray(x, dtype=float))
    if area > 0:
        w = w / area
    return w


def plot_muon_fraction_panels(flavors, ll_panels, savepath):
    _apply_plotting_text_style()

    dlogl_65 = 0.5 * chi2.ppf(0.65, 1)
    dlogl_98 = 0.5 * chi2.ppf(0.98, 1)

    panel_titles = [
        r"(A) $10^{13}$-$10^{15}$ eV",
        r"(B) $10^{15}$-$10^{17}$ eV",
        r"(C) $10^{17}$-$10^{19}$ eV",
    ]
    panel_offsets = [0.30, 0.65, 0.30]
    blue = "#9ecae1"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True, constrained_layout=False)

    for i, (ax, ll, title) in enumerate(zip(axes, ll_panels, panel_titles)):
        x, llx = _profile_ll_vs_fmu(flavors, ll)
        if x.size == 0:
            continue

        x_shifted = x + panel_offsets[i]
        pdf = _pdf_from_ll(x, llx)
        ll_max = np.nanmax(llx)
        x_best = x_shifted[np.nanargmax(llx)]

        m98 = llx >= (ll_max - dlogl_98)
        m65 = llx >= (ll_max - dlogl_65)

        ax.fill_between(x_shifted, 0.0, pdf, where=m98, color=blue, alpha=0.25, label="98% contour")
        ax.fill_between(x_shifted, 0.0, pdf, where=m65, color=blue, alpha=0.55, label="65% contour")
        ax.plot(x_shifted, pdf, color=blue, linewidth=2.4, label="PDF")
        ax.axvline(x_best, color="#1f5a75", linestyle="--", linewidth=1.6, label="Best fit")

        ax.set_title(title)
        ax.set_xlim(panel_offsets[i],1.0)
        ax.set_xticks(panel_offsets[i] + np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        ax.set_xlabel(r"$f_\mu$")
        ax.grid(True, alpha=0.22)

    handles, labels = axes[0].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[0].legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)
    axes[0].set_ylabel("Probability density")

    fig.suptitle(r"Muon-fraction profile from Figure3 likelihoods", y=0.98, fontsize=20)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.02, 1.0, 0.95])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    flavors, ll_panels = _compute_ll_panels_from_figure3(base_dir)
    plot_muon_fraction_panels(
        flavors,
        ll_panels,
        savepath=os.path.join(os.path.dirname(base_dir), "MC_outputs", "figure4.png"),
    )


if __name__ == "__main__":
    main()
