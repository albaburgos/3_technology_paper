### Figure 2: Radio-only flavor sensitivity
### (1) Low Flux (2) Intermediate Flux (3) High Flux

import os

import numpy as np

import EventGeneration as eventgen
import Helpers as helpers
from EventGeneration import loglike_combined, obs_events_radio
from Helpers import (
    generate_flavor_grid,
    geometric_edges_from_centers,
    logx_interp,
    read_effarea_csv_radio,
)
from Plotting import plot_three_ternaries_95pairs

BIN_WIDTH_LOG10 = 0.4
T_EXPOSURE = 10 * 365.25 * 24 * 3600.0
OMEGA = 4 * np.pi
GRID_STEP = 1 / 40

E0 = 1e5

# Radio classifier performance (GeV).
EMT_GEV = np.array([1e17, 2e17, 3e17, 4e17, 5e17, 6e17, 9e17, 2e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
MU_EFF = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.28, 0.37, 0.42, 0.46], dtype=float)
TAU_EFF = np.array([0.035, 0.03, 0.03, 0.035, 0.04, 0.04, 0.05, 0.09, 0.14, 0.18, 0.23], dtype=float)
ETRUTH_GEV = np.array([1e17, 2e17, 4e17, 1e18, 2e18, 3e18, 4e18, 6e18, 9e18], dtype=float) * 1e-9
TRUTH_EFF = np.array([0.1, 0.22, 0.34, 0.5, 0.6, 0.63, 0.65, 0.67, 0.67], dtype=float)


def _flux_powerlaw(energy_gev: np.ndarray, phi0_per_flavor: float = 2.06e-18, gamma: float = 2.46, e0: float = E0):
    return phi0_per_flavor * ((energy_gev / e0) ** (-gamma))


def _loglog_interp_strict(x, y, x_new):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_new = np.asarray(x_new, float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("log-log interpolation needs x>0 and y>0.")
    idx = np.argsort(x)
    return np.exp(np.interp(np.log(x_new), np.log(x[idx]), np.log(y[idx])))


def _geometric_mid_flux(e_target, e_ta, f_ta, e_au, f_au):
    f_ta_i = _loglog_interp_strict(e_ta, f_ta, e_target)
    f_au_i = _loglog_interp_strict(e_au, f_au, e_target)
    return np.sqrt(f_ta_i * f_au_i)


def build_flux_triplet():
    # Optimistic (TA)
    e2phi_ta = np.array(
        [3 * 3e-8, 3 * 1.5e-8, 18e-9, 15e-9, 1e-8, 9e-9, 8e-9, 6e-9, 5e-9, 4.5e-9, 7e-9, 9e-9, 1e-8, 8e-9, 5e-9, 2e-9, 9e-10, 3e-10],
        dtype=float,
    )
    energies_ta = np.array(
        [1.6e13, 1e14, 1e15, 2e15, 5e15, 7e15, 1e16, 2.5e16, 5e16, 1.4e17, 4e17, 7e17, 1.5e18, 4e18, 7e18, 1.5e19, 2.5e19, 7e19],
        dtype=float,
    ) * 1e-9
    phi_ta = e2phi_ta / (energies_ta ** 2)

    # Pessimistic (Au)
    e2phi_au = np.array(
        [3 * 3e-8, 3 * 1.5e-8, 18e-9, 15e-9, 1e-8, 9e-9, 8e-9, 6e-9, 5e-9, 4e-9, 3.5e-9, 3.4e-9, 3e-9, 2.3e-9, 1.6e-9, 1.1e-9, 9e-10, 5e-10, 4e-10],
        dtype=float,
    )
    energies_au = np.array(
        [1.6e13, 1e14, 1e15, 2e15, 5e15, 7e15, 1e16, 2.5e16, 5e16, 8e16, 3e17, 6e17, 1e18, 2e18, 4e18, 6e18, 9e18, 2e19, 4e19],
        dtype=float,
    ) * 1e-9
    phi_au = e2phi_au / (energies_au ** 2)

    flux_interp_ta = _flux_powerlaw(energies_ta, 4.4e-18, 2.36, E0)
    flux_interp_au = _flux_powerlaw(energies_au, 4.4e-18, 2.36, E0)
    flux_mese_ta = _flux_powerlaw(energies_ta, 2.06e-18, 2.46, E0)
    flux_mese_au = _flux_powerlaw(energies_au, 2.06e-18, 2.46, E0)

    flux_ta = flux_mese_ta + np.abs(flux_interp_ta - phi_ta)
    flux_au = flux_mese_au + np.abs(flux_interp_au - phi_au)

    energies_mid = np.unique(np.concatenate([energies_ta, energies_au])).astype(float)
    flux_mid = _geometric_mid_flux(energies_mid, energies_ta, flux_ta, energies_au, flux_au)
    return energies_au, flux_au, energies_mid, flux_mid, energies_ta, flux_ta


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    energies_Au, fluxAu, energies_mid, fluxMid, energies_TA, fluxTA = build_flux_triplet()

    helpers.energies_mid = energies_mid
    helpers.fluxMid = fluxMid
    helpers.OMEGA = OMEGA
    helpers.T_EXPOSURE = T_EXPOSURE
    eventgen.energies_mid = energies_mid
    eventgen.fluxMid = fluxMid

    coarse_edges = np.logspace(8, 10, int((10 - 8) / BIN_WIDTH_LOG10) + 1, base=10)
    fe0, fmu0, ftau0 = 0.30, 0.36, 0.34

    flavors = generate_flavor_grid(step=GRID_STEP)
    fe_arr = np.array([g[0] for g in flavors], dtype=float)
    fmu_arr = np.array([g[1] for g in flavors], dtype=float)
    ftau_arr = np.array([g[2] for g in flavors], dtype=float)

    E, A_NC, A_mu, A_tau, A_e = read_effarea_csv_radio(os.path.join(base_dir, "effareas8.csv"))
    edges = geometric_edges_from_centers(E)

    e_mid = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    Mux = logx_interp(e_mid, EMT_GEV, MU_EFF)
    Taux = logx_interp(e_mid, EMT_GEV, TAU_EFF)
    Truthx = logx_interp(e_mid, ETRUTH_GEV, TRUTH_EFF)

    base_counts_Au, a0_Au, b0_Au, _ = obs_events_radio(
        E, edges, A_NC, A_e, A_mu, A_tau,
        coarse_edges, fe0, fmu0, ftau0, fe0, fmu0, ftau0,
        Truthx, Mux, Taux, energies_Au, fluxAu,
    )

    base_counts_mid, a0_mid, b0_mid, _ = obs_events_radio(
        E, edges, A_NC, A_e, A_mu, A_tau,
        coarse_edges, fe0, fmu0, ftau0, fe0, fmu0, ftau0,
        Truthx, Mux, Taux, energies_mid, fluxMid,
    )

    base_counts_TA, a0_TA, b0_TA, _ = obs_events_radio(
        E, edges, A_NC, A_e, A_mu, A_tau,
        coarse_edges, fe0, fmu0, ftau0, fe0, fmu0, ftau0,
        Truthx, Mux, Taux, energies_TA, fluxTA,
    )

    def compute_ll_grids(energies_in, flux_in, *, Nobs_total, Nobs_ML, Nobs_mult):
        ll_radio = np.full(len(flavors), -np.inf, dtype=float)
        ll_ch1 = np.full(len(flavors), -np.inf, dtype=float)
        ll_ch2 = np.full(len(flavors), -np.inf, dtype=float)

        for j, (fe, fmu, ftau) in enumerate(flavors):
            ll_radio[j] = loglike_combined(
                E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges,
                fe, fmu, ftau, fe0, fmu0, ftau0,
                Nobs_total=Nobs_total.astype(int),
                Nobs_ML=np.rint(Nobs_ML).astype(int),
                Nobs_mult=np.rint(Nobs_mult).astype(int),
                T_vec=Truthx,
                F_vec=0.02 * np.ones_like(Truthx),
                rmu_vec=Mux,
                rtau_vec=Taux,
                energies=energies_in,
                flux=flux_in,
            )

            ll_ch1[j] = loglike_combined(
                E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges,
                fe, fmu, ftau, fe0, fmu0, ftau0,
                Nobs_total=Nobs_total.astype(int),
                Nobs_ML=np.rint(Nobs_ML).astype(int),
                Nobs_mult=None,
                T_vec=Truthx,
                F_vec=0.02 * np.ones_like(Truthx),
                energies=energies_in,
                flux=flux_in,
            )

            ll_ch2[j] = loglike_combined(
                E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges,
                fe, fmu, ftau, fe0, fmu0, ftau0,
                Nobs_total=Nobs_total.astype(int),
                Nobs_ML=None,
                Nobs_mult=np.rint(Nobs_mult).astype(int),
                rmu_vec=Mux,
                rtau_vec=Taux,
                energies=energies_in,
                flux=flux_in,
            )

        return ll_radio, ll_ch1, ll_ch2

    obs_Au = (base_counts_Au, a0_Au, b0_Au)
    obs_mid = (base_counts_mid, a0_mid, b0_mid)
    obs_TA = (base_counts_TA, a0_TA, b0_TA)

    a1_radio, a1_ch1, a1_ch2 = compute_ll_grids(
        energies_Au, fluxAu,
        Nobs_total=obs_Au[0], Nobs_ML=obs_Au[1], Nobs_mult=obs_Au[2],
    )

    a2_radio, a2_ch1, a2_ch2 = compute_ll_grids(
        energies_mid, fluxMid,
        Nobs_total=obs_mid[0], Nobs_ML=obs_mid[1], Nobs_mult=obs_mid[2],
    )

    a3_radio, a3_ch1, a3_ch2 = compute_ll_grids(
        energies_TA, fluxTA,
        Nobs_total=obs_TA[0], Nobs_ML=obs_TA[1], Nobs_mult=obs_TA[2],
    )

    plot_three_ternaries_95pairs(
        ll_grids=[a1_radio, a1_ch1, a1_ch2, a2_radio, a2_ch1, a2_ch2, a3_radio, a3_ch1, a3_ch2],
        flavors=flavors,
        fe_arr=fe_arr,
        fmu_arr=fmu_arr,
        ftau_arr=ftau_arr,
        captions=(
            "(1) Radio & Radar Low Flux\n$10^{2}$–$10^{4}$ PeV",
            "(2) Radio & Radar Intermediate Flux\n$10^{2}$–$10^{4}$ PeV",
            "(3) Radio & Radar High Flux\n$10^{2}$–$10^{4}$ PeV",
        ),
        savepath="MC_outputs/figure2.png",
    )


if __name__ == "__main__":
    main()
