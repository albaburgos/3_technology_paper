## Figure 9: Effective Areas Icecubegen2 radio 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DeltaOmega = 4*np.pi 
GEV_PER_PEV = 1e6
PLOT_LINEWIDTH = 3.0

PLOT_PNG   = "/Users/albaburgosmondejar/3_technology_paper/MC_outputs/figure9.png"
SEC_PER_YEAR = 365.25 * 24 * 3600.0

def _apply_plotting_text_style():
    """Match typography settings used in Plotting.py."""
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

_apply_plotting_text_style()

def sigmaCC(E): 
    sigma = 5.53*10**(-36)*((E)**0.363)
    return sigma 

def sigmaNC(E): 
    sigma = 2.31*10**(-36)*((E)**0.363)
    return sigma 

N_A = 6.2*(10**23)
rho = 0.92 
n = N_A*rho 
n= 5.51e23

def effective_area_from_vol(V, sigma, n):
    # Effective Volume given in Km^3
    A = V* 10**15 * n * sigma 
    return A

def log_interp(x, xp, fp):
    xp = np.asarray(xp); fp = np.asarray(fp)
    mask = (xp > 0) & (fp > 0)
    xp = xp[mask]; fp = fp[mask]
    x = np.asarray(x, dtype=float)
    if len(xp) < 2:
        return np.zeros_like(x)
    xlog  = np.log10(x)
    xplog = np.log10(xp)
    fplog = np.log10(fp)
    ylog = np.interp(xlog, xplog, fplog, left=np.nan, right=np.nan)
    y = 10**ylog
    y[np.isnan(y)] = 0.0
    return y

# ------------------------ Data ------------------------

## IC gen2 Radio
E = np.array([
    1e16, 2e16, 1e17, 2e17, 3e17, 6e17, 1e18, 2e18, 5e18
])
E = E*1e-9
V_NC  = np.array([1.5e-2, 4e-1, 5e0, 1.5e1, 3e1, 6.5e1, 1e2, 2e2, 4e2])
V_mu  = np.array([3.5e-2, 5e-1, 6.5e0, 2e1, 4.5e1, 1e2, 1.8e2, 3e2, 6e2])
V_tau = np.array([5e-2, 6e-1, 7e0, 2e1, 4e1, 8e1, 1.5e2, 2.8e2, 6e2])
V_e   = np.array([4e-1, 4e0, 3e1, 7e1, 1.2e2, 2e2, 3e2, 5e2, 7e2])

# ------------------------ Compute component A_eff ICgen 2 -------------------

sigma_NC = sigmaNC(E)
sigma_CC = sigmaCC(E)

A_NC = effective_area_from_vol (V_NC, sigma_NC, n)
A_e = effective_area_from_vol (V_e, sigma_CC, n)
A_mu = effective_area_from_vol (V_mu, sigma_CC, n)
A_tau = effective_area_from_vol (V_tau, sigma_CC,n )

E_interp = np.logspace(np.log10(E[0]), np.log10(E[-1]), 100)
A_NC_interp = log_interp(E_interp, E, A_NC)
A_e_interp = log_interp(E_interp, E, A_e)
A_mu_interp = log_interp(E_interp, E, A_mu)
A_tau_interp = log_interp(E_interp, E, A_tau)

# ------------------------ Plot ------------------------
E_interp_pev = E_interp / GEV_PER_PEV
plt.figure(figsize=(9,6), dpi=140)
plt.loglog(E_interp_pev, A_mu_interp, linewidth=PLOT_LINEWIDTH, label=r"$\nu_\mu$ CC")
plt.loglog(E_interp_pev, A_tau_interp, linewidth=PLOT_LINEWIDTH, label=r"$\nu_\tau$ CC")
plt.loglog(E_interp_pev, A_e_interp, linewidth=PLOT_LINEWIDTH, label=r"$\nu_e$ CC")
plt.loglog(E_interp_pev, A_NC_interp, ls="--", linewidth=PLOT_LINEWIDTH, label="NC")
plt.xlabel("Energy [PeV]")
plt.ylabel(r"IceCube-Gen2 Radio $A_{\rm eff}$ [cm$^2$]")
plt.grid(True, which="both", alpha=0.3)
plt.legend(loc="lower right", ncol=2)
plt.tight_layout()
plt.savefig(PLOT_PNG)
