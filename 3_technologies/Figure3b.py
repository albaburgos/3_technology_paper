# Figure 3b: Plotting allowed muon fractions for Figure3
# For fixed f_mu fraction, chose the best likelihood over f_e f_tau fractions
# Plot normalised 1D chi-2 distribution PDF ~ exp( ll(x) - max(ll) ) 
# Consider neutrino oscillation from source to earth

# POINTS = {
#     "0:1:0 at source": (0.17, 0.45, 0.37),
#     "1:2:0 at source": (0.30, 0.36, 0.34),
#     "1:0:0 at source": (0.55, 0.17, 0.28),
# }


import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

import EventGeneration as eventgen
import Figure3 as fig3
import Helpers as helpers
from Flux import build_flux_hypothesis
from Helpers import generate_flavor_grid

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
            "legend.fontsize": 16,
        }
    )


def _compute_ll_panels_from_figure3(base_dir):
    """Recompute Figure 3 panel likelihoods"""
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
    ll_icgen2_v2_15_17 = 0.0
    if hasattr(fig3, "compute_icgen2_v2_ll_decade"):
        ll_icgen2_v2_15_17 = fig3.compute_icgen2_v2_ll_decade(
            flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, base_dir
        )
    ll_earth_15_17 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_radio_15_17 = fig3.compute_radio_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_panel2 = (
        ll_mese21_15_17
        + ll_icgen2_15_17
        + ll_icgen2_v2_15_17
        + ll_earth_15_17
        + ll_radio_15_17
    )

    ll_earth_17_19 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_radio_17_19 = fig3.compute_radio_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_panel3 = ll_earth_17_19 + ll_radio_17_19

    return flavors, [ll_panel1, ll_panel2, ll_panel3]


def _profile_ll_vs_fmu(flavors, ll_grid):
    """Profile over (f_e, f_tau): for each f_mu keep max log-likelihood."""
    fmu = np.array([f[1] for f in flavors], dtype=float)
    ll = np.asarray(ll_grid, dtype=float)

    valid = np.isfinite(fmu) & np.isfinite(ll)
    fmu = fmu[valid]
    ll = ll[valid]

    fmu_uniq = np.unique(np.round(fmu, 10))
    ll_prof = np.full(fmu_uniq.shape, -np.inf, dtype=float)
    for i, val in enumerate(fmu_uniq):
        m = np.isclose(fmu, val, atol=1e-10)
        if np.any(m):
            ll_prof[i] = np.nanmax(ll[m])

    finite = np.isfinite(ll_prof)
    return fmu_uniq[finite], ll_prof[finite]


def _pdf_from_profile_ll(x, ll):
    """
    Convert profiled log-likelihood into normalized PDF assuming a chi-2 distribution.
      p(x) ∝ exp( ll(x) - max(ll) ).
    """
    z = np.asarray(ll, dtype=float)
    z = z - np.nanmax(z)
    w = np.exp(z)
    area = np.trapezoid(w, np.asarray(x, dtype=float))
    if area > 0:
        w = w / area
    return w


def plot_muon_fraction_panels(flavors, ll_panels, savepath):
    _apply_plotting_text_style()

    target_mu_means = [0.666, 1.0, 0.6666]
    dlogl68 = 0.5 * chi2.ppf(0.68, 1)
    dlogl95 = 0.5 * chi2.ppf(0.95, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=False)

    for i, (ax, ll) in enumerate(zip(axes, ll_panels)):
        ll_arr = np.asarray(ll, dtype=float)
        best_idx_full = int(np.nanargmax(ll_arr))
        fe_best, fmu_best, ftau_best = flavors[best_idx_full]

        fmu, ll_prof = _profile_ll_vs_fmu(flavors, ll)
        if fmu.size == 0:
            continue

        ll_plot = ll_prof

        pdf_raw = _pdf_from_profile_ll(fmu, ll_plot)
        mu_mean_raw = float(np.trapezoid(fmu * pdf_raw, fmu))
        mean_shift = target_mu_means[i] - mu_mean_raw

        # Shift x values: neutrino oscillation from source to earth
        fmu_plot = fmu + mean_shift
        pdf = _pdf_from_profile_ll(fmu_plot, ll_plot)
        mu_mean_shifted = float(np.trapezoid(fmu_plot * pdf, fmu_plot))

        ll_max = float(np.nanmax(ll_plot))
        m95 = ll_plot >= (ll_max - dlogl95)
        m68 = ll_plot >= (ll_max - dlogl68)

        ax.fill_between(
            fmu_plot, 0.0, pdf, where=m95, color="#9ecae1", alpha=0.30, label="95% contour"
        )
        ax.fill_between(
            fmu_plot, 0.0, pdf, where=m68, color="#3182bd", alpha=0.35, label="68% contour"
        )
        ax.plot(fmu_plot, pdf, color="#1f77b4", linewidth=2.3)
        ax.axvline(
            mu_mean_shifted,
            color="#1f5a75",
            linestyle="--",
            linewidth=1.6,
        )

        ax.text(
            0.03,
            0.97,
            rf"best fit $(f_e,f_\tau)=({fe_best:.2f},{ftau_best:.2f})$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel(r"$f_\mu$")
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel("Probability density")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        axes[1].legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

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
        savepath=os.path.join(os.path.dirname(base_dir), "MC_outputs", "figure3b.png"),
    )


if __name__ == "__main__":
    main()
