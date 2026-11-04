import numpy as np


E0 = 1e5

def flux(energy_gev: np.ndarray, phi0_per_flavor: float = 2.06e-18, gamma: float = 2.46, e0: float = E0):
    return phi0_per_flavor * ((energy_gev / e0) ** (-gamma))


def loglog_interp_strict(x, y, x_new):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x_new = np.asarray(x_new, float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("log-log interpolation needs x>0 and y>0.")
    idx = np.argsort(x)
    return np.exp(np.interp(np.log(x_new), np.log(x[idx]), np.log(y[idx])))


def geometric_mid_flux(e_target, e_ta, f_ta, e_au, f_au):
    f_ta_i = loglog_interp_strict(e_ta, f_ta, e_target)
    f_au_i = loglog_interp_strict(e_au, f_au, e_target)
    return np.sqrt(f_ta_i * f_au_i)


# Benchmark UHE flux hypotheses, digitised from Fig. 7 of Coleman et al.
# (arXiv:2402.02432).  Each digitised curve is the *total all-flavor* flux used
# there: the IceCube nu_mu 9.5-yr through-going-track power law, multiplied by 3
# for all flavors and extrapolated to ultra-high energies, plus a cosmogenic
# component from fits to UHECR data ("high"/"TA": Telescope Array fit;
# "low"/"Au": Auger fit with 10 % protons). 
# Units: energies in GeV; E^2 Phi in GeV cm^-2 s^-1 sr^-1.
def build_flux_triplet():
    # High-flux benchmark: IceCube tracks (x3) + TA cosmogenic, all flavors
    e2phi_ta = np.array(
        [3 * 3e-8, 3 * 1.5e-8, 18e-9, 15e-9, 1e-8, 9e-9, 8e-9, 6e-9, 5e-9, 4.5e-9, 7e-9, 9e-9, 1e-8, 8e-9, 5e-9, 2e-9, 9e-10, 3e-10],
        dtype=float,
    )
    energies_ta = np.array(
        [1.6e13, 1e14, 1e15, 2e15, 5e15, 7e15, 1e16, 2.5e16, 5e16, 1.4e17, 4e17, 7e17, 1.5e18, 4e18, 7e18, 1.5e19, 2.5e19, 7e19],
        dtype=float,
    ) * 1e-9
    flux_ta = e2phi_ta / (energies_ta ** 2)

    # Low-flux benchmark: IceCube tracks (x3) + Auger (10 % p) cosmogenic, all flavors
    e2phi_au = np.array(
        [3 * 3e-8, 3 * 1.5e-8, 18e-9, 15e-9, 1e-8, 9e-9, 8e-9, 6e-9, 5e-9, 4e-9, 3.5e-9, 3.4e-9, 3e-9, 2.3e-9, 1.6e-9, 1.1e-9, 9e-10, 5e-10, 4e-10],
        dtype=float,
    )
    energies_au = np.array(
        [1.6e13, 1e14, 1e15, 2e15, 5e15, 7e15, 1e16, 2.5e16, 5e16, 8e16, 3e17, 6e17, 1e18, 2e18, 4e18, 6e18, 9e18, 2e19, 4e19],
        dtype=float,
    ) * 1e-9
    flux_au = e2phi_au / (energies_au ** 2)

    # Intermediate: geometric mean of the two benchmarks on the union grid
    energies_mid = np.unique(np.concatenate([energies_ta, energies_au])).astype(float)
    flux_mid = geometric_mid_flux(energies_mid, energies_ta, flux_ta, energies_au, flux_au)
    return energies_au, flux_au, energies_mid, flux_mid, energies_ta, flux_ta


def build_flux_hypothesis(hypothesis: str = "mid"):
    energies_au, flux_au, energies_mid, flux_mid, energies_ta, flux_ta = build_flux_triplet()

    key = str(hypothesis).strip().lower()
    if key in {"low", "au", "pessimistic"}:
        return energies_au, flux_au
    if key in {"mid", "intermediate"}:
        return energies_mid, flux_mid
    if key in {"high", "ta", "optimistic"}:
        return energies_ta, flux_ta
