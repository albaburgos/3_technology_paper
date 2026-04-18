#### Code for TAMBO + Earth-Skimming with Glashow 

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEC_PER_YEAR = 365.25 * 24 * 3600.0
GEV_PER_PEV = 1e6
PLOT_LINEWIDTH = 2.5
PLOT_PNG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MC_outputs",
    "figure5.png",
)

def _apply_plotting_text_style():
    plt.rcParams.update(
        {
            "font.size": 20,
            "font.family": "serif",
            "font.serif": ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "legend.fontsize": 10,
        }
    )


_apply_plotting_text_style()

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

# ------------------------------------------------------------------

### TAMBO from sensitivity 
E_tambo2   = np.array([2e14, 1e15,4e15,1e16,4e16,1e17, 2e17, 1e18])
E2Phi_tambo = np.array([1e-7,7.5e-9,3.7e-9,2.8e-9,2.3e-9,3e-9, 3.5e-9, 6e-9])

E_tambo2, A_tambo = effective_area_from_sensitivity(
    E_tambo2, E2Phi_tambo)

E_tambo_interp = np.logspace(np.log10(E_tambo2[0]), np.log10(E_tambo2[-1]), 100)
A_tambo_interp = log_interp(E_tambo_interp, E_tambo2, A_tambo)

# TAMBO from Aperture Plot 
tambo_angle = 4*np.pi/0.1
tambo_years = 1
tambo_E_GeV_base = np.array([3e5, 4e5, 1e6, 2e6, 4e6, 5e6, 6e6, 7e6, 8e6, 1e7, 3e7, 1e8, 4e8, 1e9], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 1000, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 600, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_tambo = np.logspace(np.log10(tambo_E_GeV_base.min()), np.log10(tambo_E_GeV_base.max()), 200)
AOm_tambo_tau = log_interp(E_tambo, tambo_E_GeV_base, tambo_tau_ap_m2sr_base)     
AOm_tambo_all = log_interp(E_tambo, tambo_E_GeV_base, tambo_all_ap_m2sr_base)       
AOm_tambo_e   = np.abs(AOm_tambo_all - AOm_tambo_tau)            

A_tambo_e = AOm_tambo_e *10e4 / (tambo_angle*tambo_years )
A_tambo_tau = AOm_tambo_tau*10e4  / (tambo_angle*tambo_years)


# ------------------------ Probability Glashow Resonance ------------------------

p_vebar = A_tambo_e / (A_tambo_e + A_tambo_tau)
p_vtau  = A_tambo_tau / (A_tambo_e + A_tambo_tau)

# ------------------------------Other Earth-Skimming Experiments ---------------------------------

### Trinity
E_trinity   = np.array([1e15,1e16,1e17,1e18,1e19])
E2Phi_trinity = np.array([3e-8,2e-9,1e-9,2e-9,1e-8])
E_trinity, A_trinity = effective_area_from_sensitivity(
    E_trinity, E2Phi_trinity)

E_trinity_interp = np.logspace(np.log10(E_trinity[0]), np.log10(E_trinity[-1]), 100)
A_trinity_interp = log_interp(E_trinity_interp, E_trinity, A_trinity)

# Glashow for trinity
p_vtau_trinity = log_interp(E_trinity_interp, E_tambo, p_vtau)
valid = p_vtau_trinity > 0
A_trinity_e = np.full_like(A_trinity_interp, np.nan)
A_trinity_e[valid] = A_trinity_interp[valid] / p_vtau_trinity[valid] - A_trinity_interp[valid]

### GRAND 200k
E_grand   = np.array([6e16,1e17,2e17,4.6e17,4e18,4e19,1e20])
E2Phi_grand = np.array([1.2e-9,8e-10,3e-10,2e-10,4.5e-10,3e-9,7e-9])

E_grand, A_grand = effective_area_from_sensitivity(
    E_grand, E2Phi_grand)

E_grand_interp = np.logspace(np.log10(E_grand[0]), np.log10(E_grand[-1]), 100)
A_grand_interp = log_interp(E_grand_interp, E_grand, A_grand)

### POEMMA - results were quoted for 5 years only !
E_poemma   = np.array([1.5e16,2e16, 3e16,1e17,5e17,3e18,3e19, 1e20])
E2Phi_poemma = np.array([1e-5,1.6e-6, 3e-7,1.1e-7,6e-8,8e-8,3e-7, 1e-6])*0.5
E_poemma, A_poemma = effective_area_from_sensitivity(
    E_poemma, E2Phi_poemma
)

E_poemma_interp = np.logspace(np.log10(E_poemma[0]), np.log10(E_poemma[-1]), 100)
A_poemma_interp = log_interp(E_poemma_interp, E_poemma, A_poemma)

## plot effective areas for TAMBO, POEMMA, Grand200k, Trinity on common grid 
E_tambo_pev = E_tambo / GEV_PER_PEV
E_tambo_interp_pev = E_tambo_interp / GEV_PER_PEV
E_trinity_interp_pev = E_trinity_interp / GEV_PER_PEV
E_grand_interp_pev = E_grand_interp / GEV_PER_PEV
E_poemma_interp_pev = E_poemma_interp / GEV_PER_PEV

line_tambo_tau, = plt.loglog(
    E_tambo_interp_pev, A_tambo_interp, linewidth=PLOT_LINEWIDTH, label="TAMBO ντ"
)
plt.loglog(
    E_tambo_pev, A_tambo_e, ls="--", linewidth=PLOT_LINEWIDTH, color=line_tambo_tau.get_color(), label="TAMBO νe"
)

# Trinity tau (solid) and e (dashed, same color)
line_trinity_tau, = plt.loglog(
    E_trinity_interp_pev, A_trinity_interp, linewidth=PLOT_LINEWIDTH, label="Trinity ντ"
)
plt.loglog(
    E_trinity_interp_pev, A_trinity_e, ls="--", linewidth=PLOT_LINEWIDTH, color=line_trinity_tau.get_color(), label="Trinity νe"
)

# GRAND and POEMMA (solid, distinct colors)
plt.loglog(E_grand_interp_pev, A_grand_interp, linewidth=PLOT_LINEWIDTH, label="GRAND 200k")
plt.loglog(E_poemma_interp_pev, A_poemma_interp, linewidth=PLOT_LINEWIDTH, label="POEMMA")

plt.xlabel("Energy [PeV]")
plt.ylabel("Effective area [cm$^2$]")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
os.makedirs(os.path.dirname(PLOT_PNG), exist_ok=True)
plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved plot: {PLOT_PNG}")
plt.show()
