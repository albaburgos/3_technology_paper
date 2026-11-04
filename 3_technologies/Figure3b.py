# Figure 3b: Confidence levels on source muon fraction via Wilks' theorem.
# Maps source fmu → earth fractions via PMNS averaged oscillation and
# evaluates the likelihood directly at those earth fractions, so the profile
# along the oscillation curve needs no interpolation of the ternary grid.

import os

import matplotlib.pyplot as plt
import numpy as np

import EventGeneration as eventgen
import Figure3 as fig3
import Helpers as helpers
import PMNS_matrix as P
from Flux import build_flux_hypothesis

T_EXPOSURE = 15 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi
N_CURVE = 500  # number of source fmu values along the oscillation curve

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

def _compute_ll_panels_from_figure3(base_dir, fs):
    """Figure 3 panel likelihoods evaluated at the earth flavor fractions that the
    source fractions `fs` (ftau_source = 0) oscillate to."""
    energies_mid, flux_mid = build_flux_hypothesis("high")
    helpers.energies_mid = energies_mid
    helpers.fluxMid = flux_mid
    helpers.OMEGA = OMEGA
    helpers.T_EXPOSURE = T_EXPOSURE
    eventgen.energies_mid = energies_mid
    eventgen.fluxMid = flux_mid

    fe0,    fmu0    = oscillate_to_earth(2.0 / 3.0)  
    ftau0   = 1.0 - fe0 - fmu0
    fe0_p2, fmu0_p2 = oscillate_to_earth(1.0)    
    ftau0_p2 = 1.0 - fe0_p2 - fmu0_p2

    fe_c, fmu_c = oscillate_to_earth(fs)
    ftau_c = np.clip(1.0 - fe_c - fmu_c, 0.0, None)
    flavors = list(zip(fe_c.tolist(), fmu_c.tolist(), ftau_c.tolist()))

    decade_13_15 = np.logspace(4, 6, int((6 - 4) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_15_17 = np.logspace(6, 8, int((8 - 6) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_17_19 = np.logspace(8, 10, int((10 - 8) / fig3.BIN_WIDTH_LOG10) + 1, base=10.0)

    ll_panel1 = fig3.compute_icgen2_toise_ll_decade_perfect(
        flavors, decade_13_15, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )

    ll_earth_15_17 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_radio_15_17 = fig3.compute_radio_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )

    ll_gen2_toise_15_17 = fig3.compute_icgen2_toise_ll_decade_perfect(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )

    ll_panel2 = (
        ll_gen2_toise_15_17
        + ll_earth_15_17
        + ll_radio_15_17
    )

    

    ll_earth_17_19 = fig3.compute_earthskimming_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_radio_17_19 = fig3.compute_radio_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )

    ll_gen2_toise_17_19 = fig3.compute_icgen2_toise_ll_decade_perfect(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )

    ll_panel3 = (
        ll_earth_17_19
        + ll_radio_17_19
        + ll_gen2_toise_17_19
    )

    return [ll_panel1, ll_panel2, ll_panel3]


def oscillate_to_earth(fmu_source):
    """Map source fmu fraction to earth (fe, fmu) via averaged PMNS oscillation.
    Assume perfect knowledge of neutrino mixing parameters.

    Assumes ftau_source = 0, fe_source = 1 - fmu_source
    P[α,β] = Σ_i |U_{αi}|²|U_{βi}|² 
    f_earth[β] = Σ_α f_source[α] * P[α, β]
    """
    fmu_source = np.asarray(fmu_source, dtype=float)
    fe_source = 1.0 - fmu_source  # ftau_source = 0 prior
    fe_earth  = P.P[0, 0] * fe_source + P.P[1, 0] * fmu_source
    fmu_earth = P.P[0, 1] * fe_source + P.P[1, 1] * fmu_source
    return fe_earth, fmu_earth



def _profile_on_curve(ll_curve):
    """Profile likelihood ratio along the oscillation curve.

    Returns z = ell - ell_max (≤ 0) and profile = exp(z)
    """
    ll = np.asarray(ll_curve, dtype=float)
    z = ll - np.nanmax(ll)
    return z, np.exp(z)


def plot_muon_fraction_panels(fs, ll_panels, savepath):
    _apply_plotting_text_style()

    # Wilks thresholds for 1 DOF
    Z_68 = -0.5 * 1.000  # 68% CL
    Z_95 = -0.5 * 3.84  # 95% CL

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=False)
    fs = np.asarray(fs, dtype=float)

    for i, (ax, ll) in enumerate(zip(axes, ll_panels)):
        z, profile = _profile_on_curve(ll)

        ax.fill_between(
            fs, 0.0, profile, where=(z >= Z_95), color="#9ecae1", alpha=0.30, label="95% CL"
        )
        ax.fill_between(
            fs, 0.0, profile, where=(z >= Z_68), color="#3182bd", alpha=0.35, label="68% CL"
        )
        ax.plot(fs, profile, color="#1f77b4", linewidth=2.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel(r"$f_{\mu,\rm s}$")
        ax.grid(True, alpha=0.22)

    axes[0].set_ylabel(r"$\lambda(f_{\mu,\mathrm{s}})$")
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        axes[1].legend(dedup.values(), dedup.keys(), loc="upper right", frameon=True)

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    fig.tight_layout(rect=[0.01, 0.02, 1.0, 0.95])
    fig.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fs = np.linspace(0.0, 1.0, N_CURVE)
    ll_panels = _compute_ll_panels_from_figure3(base_dir, fs)

    plot_muon_fraction_panels(
        fs,
        ll_panels,
        savepath=os.path.join(os.path.dirname(base_dir), "MC_outputs", "figure3b.png"),
    )
