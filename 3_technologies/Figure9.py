## Figure 9: Effective Areas Icecubegen2 radio + RNO-G + ARIANNA + ARA + RET-N (they have all-flavor sensitivity)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------ Config  ------------------------

DeltaOmega = 4*np.pi 
GEV_PER_PEV = 1e6
PLOT_LINEWIDTH = 3.0

OUTPUT_CSV = "Final/effareas8.csv"
PLOT_PNG   = "MC_outputs/figure9.png"
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

### RNO-G
E_rnogo   = np.array([2e16,1e17,1e18,1e19,5e19])
E2Phi_rnogo = np.array([2e-8,9e-9,5.5e-9,6e-9, 9e-9])

### ARIANNA
E_ARIANNA   = np.array([1e16, 1e17, 1e18, 1e19, 1e20, 4e20])
E2Phi_ARIANNA = np.array([9e-6, 2e-6, 1.7e-6, 3e-6, 8e-6, 1.7e-5])

### ARA
E_ARA   = np.array([2e16,1e17,1e18, 1e19, 1e20, 1e21])
E2Phi_ARA = np.array([1e-5,1.5e-6,6e-7, 4e-7, 8e-7, 2e-6])

### RET-N
E_RET   = np.array([2e15,5e15,1e16,3e16,6e16, 2e17, 5e17,2e18, 7e18, 2e19, 5e19])
E2Phi_RET = np.array([1e-8,9e-9,5.7e-9,4e-9,3.2e-9, 1.5e-9, 9e-10, 6e-10, 8e-10, 1.5e-9, 2e-9])

# ------------------------ Helpers ------------------------

def effective_area_from_vol(V, sigma, n):
    # Effective Volume given in Km^3
    A = V* 10**15 * n * sigma 
    return A

def log_interp(x, xp, fp):
    """Log-log linear interpolation. Returns 0 outside the positive range."""
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

def effective_area_from_sensitivity(E_pts, E2Phi_pts):
    T_years=10
    mu90 =2.44
    DeltaOmega=4*np.pi
    E_eV = E_pts
    E_GeV = np.asarray(E_pts, dtype=float) * 1e-9
    E2Phi = np.asarray(E2Phi_pts, dtype=float)

    order = np.argsort(E_GeV)
    E_GeV = E_GeV[order]
    E2Phi = E2Phi[order]

    edges = np.zeros(len(E_GeV) + 1)
    for i in range(1, len(E_GeV)):
        edges[i] = np.sqrt(E_GeV[i-1] * E_GeV[i])
    edges[0]  = E_GeV[0]  / np.sqrt(E_GeV[1] / E_GeV[0])
    edges[-1] = E_GeV[-1] * np.sqrt(E_GeV[-1] / E_GeV[-2])

    T_sec = T_years * SEC_PER_YEAR
    Aeff_cm2 = mu90 *E_GeV / (T_sec * DeltaOmega * E2Phi * np.log(10) )
    return E_GeV, Aeff_cm2

# ------------------------ Compute component A_eff ICgen 2 -------------------

# Interpolate

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

# ------------------------ Compute component A_eff RNO-G + ARA + ARIANNA + RET-n -------------------

E_rnogo, A_rnogo = effective_area_from_sensitivity(
    E_rnogo, E2Phi_rnogo
)

E_RET, A_RET = effective_area_from_sensitivity(
    E_RET, E2Phi_RET
)

E_ARA, A_ARA = effective_area_from_sensitivity(
    E_rnogo, E2Phi_rnogo
)

E_ARIANNA, A_ARIANNA = effective_area_from_sensitivity(
    E_ARIANNA, E2Phi_ARIANNA)

E_all_interp = np.logspace(np.log10(E_RET[0]), np.log10(E_ARA[-1]), 100)
A_all_interp = log_interp(E_all_interp, E_rnogo, A_rnogo) + log_interp(E_all_interp, E_RET, A_RET) + log_interp(E_all_interp, E_ARA, A_ARA) +log_interp(E_all_interp, E_ARIANNA, A_ARIANNA) 

A_tot_interp = A_NC_interp + A_e_interp + A_mu_interp + A_tau_interp
eps = 1e-300  

f_NC_E = A_NC_interp / (A_tot_interp + eps)
f_e_E  = A_e_interp  / (A_tot_interp + eps)
f_mu_E = A_mu_interp / (A_tot_interp + eps)
f_tau_E= A_tau_interp/ (A_tot_interp + eps)

f_NC = log_interp(E_all_interp, E_interp, np.clip(f_NC_E, 0.0, 1.0))
f_e  = log_interp(E_all_interp, E_interp, np.clip(f_e_E,  0.0, 1.0))
f_mu = log_interp(E_all_interp, E_interp, np.clip(f_mu_E, 0.0, 1.0))
f_tau= log_interp(E_all_interp, E_interp, np.clip(f_tau_E,0.0, 1.0))

f_stack = np.vstack([f_NC, f_e, f_mu, f_tau])
f_sum = f_stack.sum(axis=0) + eps
f_stack /= f_sum
f_NC, f_e, f_mu, f_tau = f_stack

A_rnogo_NC  = A_all_interp * f_NC
A_rnogo_e   = A_all_interp * f_e
A_rnogo_mu  = A_all_interp * f_mu
A_rnogo_tau = A_all_interp * f_tau

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
print(f"Saved plot: {PLOT_PNG}")
