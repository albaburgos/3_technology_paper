import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import binom

from Helpers import _clip01, integrate_bin_I, integrate_bin_I_mese

energies_mid = np.array([1.0], dtype=float)
fluxMid = np.array([0.0], dtype=float)

# ---------------------------Event Generation --------------------------------------------

def obs_events_radio(E: np.ndarray, edges: np.ndarray,A_NC, A_e, A_mu, A_tau,
                          coarse_edges,  fe, fmu, ftau, fe0, fmu0, ftau0 , Truth, Mu, Tau,energies,flux):
    counts = []
    countsML = []
    countsmulti = []
    countsvmvt = []
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i])
        Emax = float(coarse_edges[i + 1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax, energies, flux)
        Imu  = integrate_bin_I(edges, E, A_mu, Emin, Emax,energies, flux)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax,energies, flux)
        INC = integrate_bin_I(edges, E, A_NC, Emin, Emax, energies, flux)

        counts_all = Ie*fe0+Imu*fmu0+Itau*ftau0+INC

        counts.append(counts_all)
        countsML.append(Ie*fe0* Truth[i] + ((1-Ie*fe0))*0.02)
        countsmulti.append(Imu*(fmu0)*Mu[i]+Itau*(ftau0)*Tau[i])
        countsvmvt.append(Imu*fmu0+Itau*ftau0)

    return np.asarray(counts), np.asarray(countsML), np.asarray(countsmulti), np.asarray(countsvmvt)

def obs_events_tau(E: np.ndarray, edges: np.ndarray, A_e: np.ndarray, A_mu: np.ndarray, A_tau: np.ndarray,
                          coarse_edges: np.ndarray, fe: float, fmu: float, ftau: float, fe0, fmu0, ftau0, p_glash, energies,flux):
    counts = []
    counts_glash=[]
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i])
        Emax = float(coarse_edges[i + 1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax, energies,flux)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax, energies,flux)
        counts.append(Itau*ftau0+ Ie*fe0)
        counts_glash.append(Ie*fe0)

    return np.asarray(counts), np.asarray(counts_glash)

def _expected_components_per_bin_tau(E, edges, A_e, A_mu, A_tau,
                                 coarse_edges, fe, fmu, ftau, fe0, fmu0, ftau0, energies,flux):

    N_all = []
    Ne = []; Nmu = []; Ntau = []
    for i in range(len(coarse_edges)-1):
        Emin = float(coarse_edges[i]); Emax = float(coarse_edges[i+1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax,energies, flux)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax,energies, flux)

        Ne_i   = Ie * fe 
        Ntau_i = Itau * ftau
        Nall_i = Itau*ftau + Ie * fe 

        Ne.append(Ne_i); Ntau.append(Ntau_i); N_all.append(Nall_i)

    return np.asarray(N_all), np.asarray(Ne), np.asarray(Nmu), np.asarray(Ntau)

def _expected_components_per_bin_gen2(E, edges, A_e, A_mu, A_tau,
                                 coarse_edges, fe, fmu, ftau):

    N_all = []
    for i in range(len(coarse_edges)-1):
        Emin = float(coarse_edges[i]); Emax = float(coarse_edges[i+1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax)
        Imu = integrate_bin_I(edges, E, A_mu, Emin, Emax)

        N_all.append(Ie*fe +Itau*ftau+Imu*fmu)

    return np.asarray(N_all)

def _expected_components_per_bin(E, edges, A_NC, A_e, A_mu, A_tau,
                                 coarse_edges, fe, fmu, ftau, fe0, fmu0, ftau0, energies,flux):
    """
    Returns (N_all, N_CC_e, N_CC_mu, N_CC_tau) for each coarse bin,
    using the same integrals your events_per_coarse_bin_radio uses.
    """
    N_all = []
    Ne = []; Nmu = []; Ntau = []
    for i in range(len(coarse_edges)-1):
        Emin = float(coarse_edges[i]); Emax = float(coarse_edges[i+1])
        Ie   = integrate_bin_I(edges, E, A_e,  Emin, Emax, energies, flux)
        Imu  = integrate_bin_I(edges, E, A_mu, Emin, Emax, energies, flux)
        Itau = integrate_bin_I(edges, E, A_tau, Emin, Emax, energies, flux)
        INC  = integrate_bin_I(edges, E, A_NC, Emin, Emax, energies, flux)

        Ne_i   = Ie   * fe
        Nmu_i  = Imu  * fmu
        Ntau_i = Itau * ftau
        Nall_i = Ne_i + Nmu_i + Ntau_i + INC

        Ne.append(Ne_i); Nmu.append(Nmu_i); Ntau.append(Ntau_i); N_all.append(Nall_i)

    return np.asarray(N_all), np.asarray(Ne), np.asarray(Nmu), np.asarray(Ntau)

def events_per_coarse_bin_single(E: np.ndarray, edges: np.ndarray,
                                 A_e: np.ndarray, A_mu: np.ndarray, A_tau: np.ndarray,
                                 coarse_edges: np.ndarray,
                                 fe: float, fmu: float, ftau: float, norm: float = 1.0):
    counts = []
    for i in range(len(coarse_edges) - 1):
        Emin = float(coarse_edges[i])
        Emax = float(coarse_edges[i + 1])
        Ie   = integrate_bin_I_mese(edges, E, A_e,  Emin, Emax)
        Imu  = integrate_bin_I_mese(edges, E, A_mu, Emin, Emax)
        Itau = integrate_bin_I_mese(edges, E, A_tau, Emin, Emax)
        counts.append( (Ie*fe + Imu*fmu + Itau*ftau))
    return np.asarray(counts)


# ---------------------------Likelihood Calculations ---------------------------------------

def loglike_totals_poisson(Nobs_total, Nexp_total):
    """ Sum_i log Pois(Nobs_i | Nexp_i). """
    lam = np.clip(np.asarray(Nexp_total, float), 1e-12, None)
    k   = np.asarray(Nobs_total, float)
    return float(np.sum(k*np.log(lam) - lam - gammaln(k+1.0)))

def poisson_loglike(k, lam):
    lam = np.asarray(lam, dtype=float)
    lam = np.clip(lam, 1e-12, None)
    k = np.asarray(k, dtype=float)
    return np.sum(k * np.log(lam) - lam - gammaln(k + 1.0))


def loglike_channel_mult(Nobs_total, Nobs_mult, Nexp_total,
                         Nexp_CC_mu, Nexp_CC_tau, rmu_vec, rtau_vec):
    Nobs_total = np.asarray(Nobs_total, int)
    Nobs_mult  = np.asarray(Nobs_mult,  int)
    Nexp_total = np.asarray(Nexp_total, float)

    frac_mu  = np.clip(Nexp_CC_mu  / np.clip(Nexp_total, 1e-12, None), 0.0, 1.0)
    frac_tau = np.clip(Nexp_CC_tau / np.clip(Nexp_total, 1e-12, None), 0.0, 1.0)

    p_mu  = _clip01(frac_mu  * np.asarray(rmu_vec,  float))
    p_tau = _clip01(frac_tau * np.asarray(rtau_vec, float))

    ll = 0.0
    for nobs, mlos, p1, p2 in zip(Nobs_total, Nobs_mult, p_mu, p_tau):
        k0 = max(0, mlos - nobs); k1 = min(mlos, nobs)
        ks = np.arange(k0, k1+1, dtype=int)
        terms = binom.logpmf(ks,  nobs, p1) + binom.logpmf(mlos-ks, nobs, p2)
        ll += logsumexp(terms) if ks.size else -np.inf
    return float(ll)


def loglike_channel_ML(Nobs_total, Nobs_ML, Nexp_total, Nexp_CC_e, T_vec, F_vec):

    Nobs_total = np.asarray(Nobs_total, int)
    Nobs_ML    = np.asarray(Nobs_ML,    int)
    Nexp_total = np.asarray(Nexp_total, float)
    frac_e     = np.clip(Nexp_CC_e / np.clip(Nexp_total, 1e-12, None), 0.0, 1.0)
    ptp = _clip01(frac_e * np.asarray(T_vec, float))
    pfp = _clip01((1.0 - frac_e) * np.asarray(F_vec, float))

    ll = 0.0
    for nobs, mlos, p1, p2 in zip(Nobs_total, Nobs_ML, ptp, pfp):
        # valid k so both binomials are defined
        k0 = max(0, mlos - nobs); k1 = min(mlos, nobs)
        ks = np.arange(k0, k1+1, dtype=int)
        terms = binom.logpmf(ks,  nobs, p1) + binom.logpmf(mlos-ks, nobs, p2)
        ll += logsumexp(terms) if ks.size else -np.inf
    return float(ll)

def loglike_glashow(
    Nobs_total,   
    N_glashow,    
    Nexp_total,  
    Nexp_e,  
    p_glashow       
):

    frac_ebar = np.clip(Nexp_e / np.clip(Nexp_total, 1e-12, None), 0.0, 1.0)
    pG = _clip01(frac_ebar * p_glashow)
    ll = np.sum(binom.logpmf(N_glashow, Nobs_total, pG))

    return float(ll)


def loglike_combined(
    E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges,
    fe, fmu, ftau, fe0, fmu0, ftau0,
    Nobs_total,
    Nobs_ML=None,
    Nobs_mult=None,
    T_vec=None, F_vec=None,
    rmu_vec=None, rtau_vec=None, energies = energies_mid, flux=fluxMid
):

    Nexp_total, Nexp_e, Nexp_mu, Nexp_tau = _expected_components_per_bin(
        E, edges, A_NC, A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, fe0, fmu0, ftau0, energies, flux
    )

    ll = loglike_totals_poisson(Nobs_total, Nexp_total)

    if Nobs_mult is not None:
        if rmu_vec is None or rtau_vec is None:
            raise ValueError("rmu_vec and rtau_vec must be provided when Nobs_mult is given.")
        ll += loglike_channel_mult(
            Nobs_total, Nobs_mult, Nexp_total, Nexp_mu, Nexp_tau, rmu_vec, rtau_vec
        )

    if Nobs_ML is not None:
        if T_vec is None or F_vec is None:
            raise ValueError("T_vec and F_vec must be provided when Nobs_ML is given.")
        ll += loglike_channel_ML(
            Nobs_total, Nobs_ML, Nexp_total, Nexp_e, T_vec, F_vec
        )
    return ll


def loglike_combined_tau(E, edges, A_e, A_mu, A_tau, coarse_edges,
                     fe, fmu, ftau, fe0, fmu0, ftau0,
                     Nobs_total, N_glashow, p_glashow, energies, flux):

    Nexp_total, Nexp_e, Nexp_mu, Nexp_tau = _expected_components_per_bin_tau(
        E, edges,  A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, fe0, fmu0, ftau0, energies,flux
    )

    ll  = loglike_totals_poisson(Nobs_total, Nexp_total)
    ll += loglike_glashow(Nobs_total, N_glashow, Nexp_total, Nexp_e, p_glashow)
    return ll


def loglike_combined_tau2(E, edges, A_e, A_mu, A_tau, coarse_edges,
                     fe, fmu, ftau, fe0, fmu0, ftau0,
                     Nobs_total, N_glashow, p_glashow, energies, flux):

    Nexp_total, Nexp_e, Nexp_mu, Nexp_tau = _expected_components_per_bin_tau(
        E, edges,  A_e, A_mu, A_tau, coarse_edges, fe, fmu, ftau, fe0, fmu0, ftau0, energies,flux
    )

    ll  = loglike_totals_poisson(Nobs_total, Nexp_total)
    #ll += loglike_glashow(Nobs_total, N_glashow, Nexp_total, Nexp_e, p_glashow)
    return ll
