"""Figure 3: flavor sensitivity by energy decade.

Panels:
1) 10^13-10^15 eV: MESE21
2) 10^15-10^17 eV: MESE21 + ic-gen2  + Earth-skimming + Radio
3) 10^17-10^19 eV: Earth-skimming + Radio
"""

import os

import numpy as np
import pandas as pd

import EventGeneration as eventgen
import Helpers as helpers
from EventGeneration import (
    events_per_coarse_bin_single,
    loglike_combined,
    loglike_combined_tau,
    loglike_totals_poisson,
    obs_events_radio,
    obs_events_tau,
)
from Flux import build_flux_hypothesis
from Helpers import (
    generate_flavor_grid,
    geometric_edges_from_centers,
    log_interp,
    logx_interp,
    read_eff_area_csv_gen,
    read_effarea_csv,
    read_effarea_csv_radio,
    read_effarea_csv_stacked,
)
from Plotting import plot_three_ternaries

BIN_WIDTH_LOG10 = 0.4
T_EXPOSURE = 10 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi
GRID_STEP = 1 / 40

# Radio classifier curves (GeV)
EMT_GEV = np.array([1e17, 2e17, 3e17, 4e17, 5e17, 6e17, 9e17, 2e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
MU_EFF = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.28, 0.37, 0.42, 0.46], dtype=float)
TAU_EFF = np.array([0.035, 0.03, 0.03, 0.035, 0.04, 0.04, 0.05, 0.09, 0.14, 0.18, 0.23], dtype=float)
ETRUTH_GEV = np.array([1e17, 2e17, 4e17, 1e18, 2e18, 3e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
TRUTH_EFF = np.array([0.1, 0.22, 0.34, 0.5, 0.6, 0.63, 0.65, 0.67, 0.67], dtype=float)


def _compute_p_vebar_binned(coarse_edges: np.ndarray) -> np.ndarray:
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

    nbin = len(coarse_edges) - 1
    if nbin <= 0:
        return np.array([], dtype=float)
    if e_sel.size == 0:
        return np.zeros(nbin, dtype=float)

    idx = np.digitize(e_sel, coarse_edges, right=False) - 1
    idx = np.clip(idx, 0, nbin - 1)

    sum_per_bin = np.zeros(nbin, dtype=float)
    cnt_per_bin = np.zeros(nbin, dtype=int)
    np.add.at(sum_per_bin, idx, p_sel)
    np.add.at(cnt_per_bin, idx, 1)

    p_binned = np.zeros(nbin, dtype=float)
    nz = cnt_per_bin > 0
    p_binned[nz] = sum_per_bin[nz] / cnt_per_bin[nz]
    return p_binned


def _resolve_path(base_dir: str, filename: str) -> str:
    candidates = [
        os.path.join(base_dir, filename),
        os.path.join(os.path.dirname(base_dir), "final", filename),
        os.path.join(os.path.dirname(base_dir), "Final", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename} in expected locations.")


def compute_mese21_ll_decade(
    flavors,
    coarse_edges,
    fe0,
    fmu0,
    ftau0,
    base_dir,
):
    mese21_path = _resolve_path(base_dir, "effareasMESE21.csv")
    E_mese21, triplets_mese21 = read_effarea_csv_stacked(mese21_path)
    edges_mese21 = geometric_edges_from_centers(E_mese21)

    k_asimov = []
    for (a_mu_d, a_tau_d, a_e_d) in triplets_mese21:
        k_d = events_per_coarse_bin_single(
            E_mese21,
            edges_mese21,
            a_e_d,
            a_mu_d,
            a_tau_d,
            coarse_edges,
            fe0,
            fmu0,
            ftau0,
            norm=1.0,
        )
        k_asimov.append(np.asarray(k_d, dtype=float))

    ll_grid = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        ll = 0.0
        for (a_mu_d, a_tau_d, a_e_d), k_d in zip(triplets_mese21, k_asimov):
            lam_d = events_per_coarse_bin_single(
                E_mese21,
                edges_mese21,
                a_e_d,
                a_mu_d,
                a_tau_d,
                coarse_edges,
                fe,
                fmu,
                ftau,
                norm=1.0,
            )
            ll += loglike_totals_poisson(k_d, np.asarray(lam_d, dtype=float))
        ll_grid[j] = ll
    return ll_grid


def compute_icgen2_ll_decade(
    flavors,
    coarse_edges,
    fe0,
    fmu0,
    ftau0,
    base_dir,
):
    gen2_path = _resolve_path(base_dir, "effareasgen2.csv")
    E_gen2, track_triplet, casc_triplet = read_eff_area_csv_gen(gen2_path)
    edges_gen2 = geometric_edges_from_centers(E_gen2)

    a_mu_t, a_tau_t, a_e_t = track_triplet
    a_mu_c, a_tau_c, a_e_c = casc_triplet

    k_asimov_track = events_per_coarse_bin_single(
        E_gen2,
        edges_gen2,
        a_e_t,
        a_mu_t,
        a_tau_t,
        coarse_edges,
        fe0,
        fmu0,
        ftau0,
        norm=1.0,
    )
    k_asimov_casc = events_per_coarse_bin_single(
        E_gen2,
        edges_gen2,
        a_e_c,
        a_mu_c,
        a_tau_c,
        coarse_edges,
        fe0,
        fmu0,
        ftau0,
        norm=1.0,
    )

    ll_grid = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        lam_t = events_per_coarse_bin_single(
            E_gen2,
            edges_gen2,
            a_e_t,
            a_mu_t,
            a_tau_t,
            coarse_edges,
            fe,
            fmu,
            ftau,
            norm=1.0,
        )
        lam_c = events_per_coarse_bin_single(
            E_gen2,
            edges_gen2,
            a_e_c,
            a_mu_c,
            a_tau_c,
            coarse_edges,
            fe,
            fmu,
            ftau,
            norm=1.0,
        )
        ll_grid[j] = (
            loglike_totals_poisson(np.asarray(k_asimov_track, dtype=float), np.asarray(lam_t, dtype=float))
            + loglike_totals_poisson(np.asarray(k_asimov_casc, dtype=float), np.asarray(lam_c, dtype=float))
        )
    return ll_grid


def compute_earthskimming_ll_decade(
    flavors,
    coarse_edges,
    fe0,
    fmu0,
    ftau0,
    energies_mid,
    flux_mid,
    base_dir,
):
    eff9_path = _resolve_path(base_dir, "effareas9.csv")
    Et, A_mut, A_taut, A_et = read_effarea_csv(eff9_path)
    edgest = geometric_edges_from_centers(Et)

    p_vebar_binned = _compute_p_vebar_binned(coarse_edges)
    tau_obs, glash_obs = obs_events_tau(
        Et,
        edgest,
        A_et,
        A_mut,
        A_taut,
        coarse_edges,
        fe0,
        fmu0,
        ftau0,
        fe0,
        fmu0,
        ftau0,
        p_vebar_binned,
        energies_mid,
        flux_mid,
    )

    ll_grid = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        ll_grid[j] = loglike_combined_tau(
            Et,
            edgest,
            A_et,
            A_mut,
            A_taut,
            coarse_edges,
            fe,
            fmu,
            ftau,
            fe0,
            fmu0,
            ftau0,
            tau_obs.astype(int),
            np.rint(glash_obs).astype(int),
            p_vebar_binned,
            energies_mid,
            flux_mid,
        )
    return ll_grid


def compute_radio_ll_decade(
    flavors,
    coarse_edges,
    fe0,
    fmu0,
    ftau0,
    energies_mid,
    flux_mid,
    base_dir,
):
    eff8_path = _resolve_path(base_dir, "effareas8.csv")
    E, A_NC, A_mu, A_tau, A_e = read_effarea_csv_radio(eff8_path)
    edges = geometric_edges_from_centers(E)

    e_mid = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    mux = logx_interp(e_mid, EMT_GEV, MU_EFF)
    taux = logx_interp(e_mid, EMT_GEV, TAU_EFF)
    truthx = logx_interp(e_mid, ETRUTH_GEV, TRUTH_EFF)

    nobs_total, nobs_ml, nobs_mult, _ = obs_events_radio(
        E,
        edges,
        A_NC,
        A_e,
        A_mu,
        A_tau,
        coarse_edges,
        fe0,
        fmu0,
        ftau0,
        fe0,
        fmu0,
        ftau0,
        truthx,
        mux,
        taux,
        energies_mid,
        flux_mid,
    )

    ll_grid = np.full(len(flavors), -np.inf, dtype=float)
    for j, (fe, fmu, ftau) in enumerate(flavors):
        ll_grid[j] = loglike_combined(
            E,
            edges,
            A_NC,
            A_e,
            A_mu,
            A_tau,
            coarse_edges,
            fe,
            fmu,
            ftau,
            fe0,
            fmu0,
            ftau0,
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
    return ll_grid


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    energies_mid, flux_mid = build_flux_hypothesis("mid")
    helpers.energies_mid = energies_mid
    helpers.fluxMid = flux_mid
    helpers.OMEGA = OMEGA
    helpers.T_EXPOSURE = T_EXPOSURE
    eventgen.energies_mid = energies_mid
    eventgen.fluxMid = flux_mid

    fe0, fmu0, ftau0 = 0.30, 0.36, 0.34
    fe0_p2, fmu0_p2, ftau0_p2 = 0.17, 0.45, 0.37

    flavors = generate_flavor_grid(step=GRID_STEP)
    fe_arr = np.array([f[0] for f in flavors], dtype=float)
    fmu_arr = np.array([f[1] for f in flavors], dtype=float)
    ftau_arr = np.array([f[2] for f in flavors], dtype=float)

    decade_13_15 = np.logspace(4, 6, int((6 - 4) / BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_15_17 = np.logspace(6, 8, int((8 - 6) / BIN_WIDTH_LOG10) + 1, base=10.0)
    decade_17_19 = np.logspace(8, 10, int((10 - 8) / BIN_WIDTH_LOG10) + 1, base=10.0)

    ll_mese21_13_15 = compute_mese21_ll_decade(
        flavors, decade_13_15, fe0, fmu0, ftau0, base_dir
    )

    ll_mese21_15_17 = compute_mese21_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, base_dir
    )
    ll_icgen2_15_17 = compute_icgen2_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, base_dir
    )
  
    ll_earth_15_17 = compute_earthskimming_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_radio_15_17 = compute_radio_ll_decade(
        flavors, decade_15_17, fe0_p2, fmu0_p2, ftau0_p2, energies_mid, flux_mid, base_dir
    )
    ll_panel2 = (
        ll_mese21_15_17
        # + ll_mesecos_scaled
        + ll_icgen2_15_17
        + ll_earth_15_17
        + ll_radio_15_17
    )

    ll_earth_17_19 = compute_earthskimming_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_radio_17_19 = compute_radio_ll_decade(
        flavors, decade_17_19, fe0, fmu0, ftau0, energies_mid, flux_mid, base_dir
    )
    ll_earth_plus_radio_17_19 = ll_earth_17_19 + ll_radio_17_19

    plot_three_ternaries(
        [ll_mese21_13_15, ll_panel2, ll_earth_plus_radio_17_19],
        flavors=flavors,
        fe_arr=fe_arr,
        fmu_arr=fmu_arr,
        ftau_arr=ftau_arr,
        captions=(
            "(A) $10^{13}$-$10^{15}$ eV",
            "(B) $10^{15}$-$10^{17}$ eV",
            "(C) $10^{17}$-$10^{19}$ eV",
        ),
        savepath="MC_outputs/figure3.png",
    )


if __name__ == "__main__":
    main()
