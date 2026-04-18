## 3-Technology Paper, Figure 1
# (1) In-Ice Radio & Radio (2) Earth-Skimming (3) All Technologies Combined

import os
import numpy as np

import EventGeneration as eventgen
import Helpers as helpers
from Flux import build_flux_hypothesis
from EventGeneration import (
    loglike_combined,
    loglike_combined_tau,
    obs_events_radio,
    obs_events_tau,
)
from Helpers import (
    generate_flavor_grid,
    geometric_edges_from_centers,
    log_interp,
    logx_interp,
    read_effarea_csv,
    read_effarea_csv_radio,
)
from Plotting import plot_three_ternaries

BIN_WIDTH_LOG10 = 0.4
T_EXPOSURE = 10 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi
GRID_STEP = 1 / 40

E0 = 1e5

# Radio classifier performance curves (energies are in GeV after conversion).
EMT_GEV = np.array([1e17, 2e17, 3e17, 4e17, 5e17, 6e17, 9e17, 2e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
MU_EFF = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.28, 0.37, 0.42, 0.46], dtype=float)
TAU_EFF = np.array([0.035, 0.03, 0.03, 0.035, 0.04, 0.04, 0.05, 0.09, 0.14, 0.18, 0.23], dtype=float)
ETRUTH_GEV = np.array([1e17, 2e17, 4e17, 1e18, 2e18, 3e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
TRUTH_EFF = np.array([0.1, 0.22, 0.34, 0.5, 0.6, 0.63, 0.65, 0.67, 0.67], dtype=float)


def compute_radio_ll(flavors, coarse_edges, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir):
    fe_arr = np.array([f[0] for f in flavors], dtype=float)
    fmu_arr = np.array([f[1] for f in flavors], dtype=float)
    ftau_arr = np.array([f[2] for f in flavors], dtype=float)

    eff8_path = os.path.join(base_dir, "effareas8.csv")
    E, A_NC, A_mu, A_tau, A_e = read_effarea_csv_radio(eff8_path)
    edges = geometric_edges_from_centers(E)

    e_mid = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    mux = logx_interp(e_mid, EMT_GEV, MU_EFF)
    taux = logx_interp(e_mid, EMT_GEV, TAU_EFF)
    truthx = logx_interp(e_mid, ETRUTH_GEV, TRUTH_EFF)

    nobs_total, nobs_ml, nobs_mult, _ = obs_events_radio(
        E, edges, A_NC, A_e, A_mu, A_tau,
        coarse_edges, fe0, fmu0, ftau0, fe0, fmu0, ftau0,
        truthx, mux, taux, energies_mid, flux_mid,
    )

    ll_radio = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        ll_radio[j] = loglike_combined(
            E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges,
            fe, fmu, ftau, fe0, fmu0, ftau0,
            Nobs_total=nobs_total.astype(int),
            Nobs_ML=np.rint(nobs_ml).astype(int),
            Nobs_mult=np.rint(nobs_mult).astype(int),
            T_vec=truthx,
            F_vec=0.02 * np.ones_like(truthx),
            rmu_vec=mux,
            rtau_vec=taux,
            energies=energies_mid,
            flux=flux_mid,
        )
    return ll_radio, fe_arr, fmu_arr, ftau_arr


def compute_tau_ll(flavors, coarse_edges, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir):
    # TAMBO-based ebar proxy used for Glashow probability per energy.
    tambo_angle = 4 * np.pi / 0.1
    tambo_years = 1.0
    tambo_e_base = np.array([3e5, 4e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
    tambo_all_ap = np.array([1, 9, 50, 150, 500, 1000, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
    tambo_tau_ap = np.array([1, 9, 50, 150, 500, 600, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

    e_tambo = np.logspace(np.log10(tambo_e_base.min()), np.log10(tambo_e_base.max()), 200)
    aom_tau = log_interp(e_tambo, tambo_e_base, tambo_tau_ap)
    aom_all = log_interp(e_tambo, tambo_e_base, tambo_all_ap)
    aom_e = np.abs(aom_all - aom_tau)
    a_tambo_e = aom_e * 10e4 / (tambo_angle * tambo_years)
    a_tambo_tau = aom_tau * 10e4 / (tambo_angle * tambo_years)
    p_vebar = a_tambo_e / np.clip(a_tambo_e + a_tambo_tau, 1e-12, None)

    mask = (e_tambo >= coarse_edges[0]) & (e_tambo < coarse_edges[-1])
    e_sel = e_tambo[mask]
    p_sel = p_vebar[mask]
    valid = np.isfinite(e_sel) & np.isfinite(p_sel)
    e_sel = e_sel[valid]
    p_sel = p_sel[valid]
    bin_indices = np.digitize(e_sel, coarse_edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(coarse_edges) - 2)
    num_bins = len(coarse_edges) - 1
    sum_per_bin = np.zeros(num_bins, dtype=float)
    cnt_per_bin = np.zeros(num_bins, dtype=int)
    np.add.at(sum_per_bin, bin_indices, p_sel)
    np.add.at(cnt_per_bin, bin_indices, 1)
    p_vebar_binned = np.zeros(num_bins, dtype=float)
    nz = cnt_per_bin > 0
    p_vebar_binned[nz] = sum_per_bin[nz] / cnt_per_bin[nz]

    eff9_path = os.path.join(base_dir, "effareas9.csv")
    Et, A_mut, A_taut, A_et = read_effarea_csv(eff9_path)
    edgest = geometric_edges_from_centers(Et)

    base_tau, glashow = obs_events_tau(
        Et, edgest, A_et, A_mut, A_taut, coarse_edges,
        fe0, fmu0, ftau0, fe0, fmu0, ftau0,
        p_vebar_binned, energies_mid, flux_mid,
    )

    ll_tau = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        ll_tau[j] = loglike_combined_tau(
            Et, edgest, A_et, A_mut, A_taut, coarse_edges,
            fe, fmu, ftau, fe0, fmu0, ftau0,
            base_tau.astype(int),
            np.rint(glashow).astype(int),
            p_vebar_binned,
            energies_mid,
            flux_mid,
        )
    return ll_tau


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    coarse_edges = np.logspace(8, 10, int((10 - 8) / BIN_WIDTH_LOG10) + 1, base=10)
    fe0, fmu0, ftau0 = 0.30, 0.36, 0.34

    energies_mid, flux_mid = build_flux_hypothesis("high")

    helpers.energies_mid = energies_mid
    helpers.fluxMid = flux_mid
    helpers.OMEGA = OMEGA
    helpers.T_EXPOSURE = T_EXPOSURE
    eventgen.energies_mid = energies_mid
    eventgen.fluxMid = flux_mid

    flavors = generate_flavor_grid(step=GRID_STEP)
    ll_radio, fe_arr, fmu_arr, ftau_arr = compute_radio_ll(
        flavors, coarse_edges, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_tau = compute_tau_ll(
        flavors, coarse_edges, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_combined = ll_radio + ll_tau

    plot_three_ternaries(
        [ll_radio, ll_tau, ll_combined],
        flavors=flavors,
        fe_arr=fe_arr,
        fmu_arr=fmu_arr,
        ftau_arr=ftau_arr,
        captions=("(1) In-Ice Radio & Radio $10^{2}$–$10^{4}$ PeV ", "(2) Earth-Skimming $10^{2}$–$10^{4}$ PeV ", "(3) All Technologies $10^{2}$–$10^{4}$ PeV"
        ),
        savepath="MC_outputs/figure8.png",
    )


if __name__ == "__main__":
    main()
