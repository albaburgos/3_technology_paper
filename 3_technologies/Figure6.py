#### Figure 6: Effective Areas for Earth-Skimming Neutrino Detectors (TAMBO 22k & 5k, Trinity, GRAND 200k, POEMMA, AUGER)
## Source: Snowmass paper Figure 18 https://arxiv.org/pdf/2203.08096 at 90%CL
# QinRui for TAMBO and TRINITY https://arxiv.org/pdf/2607.26128

## (TAMBO cross-referenced against https://arxiv.org/abs/2507.08070 Figure 3 Aperture Plot) 
## (Updated Trinity Sensitivity estimates from https://arxiv.org/pdf/2509.18236 Figure 2 at 90%CL : Not Used)

# TAMBO FOV 120*30 deg = 0.2806sr. 120 in azimuth, 30 in zenith
# Trinity FOV 5deg below horizon x 60 azimuth = 0.09126

import os
import numpy as np
import matplotlib.pyplot as plt


SEC_PER_YEAR = 365.25 * 24 * 3600.0
GEV_PER_PEV = 1e6
PLOT_LINEWIDTH = 3.0
PLOT_PNG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MC_outputs",
    "figure6.png",
)

def _apply_plotting_text_style():
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

def log_interp(x, xp, fp):
    xp = np.asarray(xp); fp = np.asarray(fp)
    mask = (xp > 0) & (fp > 0)
    xp = xp[mask]; fp = fp[mask]
    x = np.asarray(x, dtype=float)
    if len(xp) < 2:
        return np.zeros_like(x)
    order = np.argsort(xp, kind="stable")
    xp = xp[order]; fp = fp[order]
    xp, keep_idx = np.unique(xp, return_index=True)
    fp = fp[keep_idx]
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

def effective_area_from_aperture(E_grid, Aperture_pts_tau, Aperture_pts_e, angle):

    E = np.logspace(np.log10(E_grid.min()), np.log10(E_grid.max()), 300)
    A_tau = log_interp(E, E_grid, Aperture_pts_tau)
    A_e = log_interp(E, E_grid, Aperture_pts_e)

    A_final_tau = A_tau *1e4 * angle / ( 4*np.pi ) 
    A_final_e = A_e*1e4 * angle / (  4*np.pi )

    return E, A_final_e, A_final_tau

# ------------------------------------------------------------------

### TAMBO from sensitivity 
E_tambo   = np.array([2e14, 1e15,4e15,1e16,4e16,1e17, 2e17, 1e18])
E2Phi_tambo = np.array([1e-7,7.5e-9,3.7e-9,2.8e-9,2.3e-9,3e-9, 3.5e-9, 6e-9])

E_tambo2, A_tambo = effective_area_from_sensitivity(
    E_tambo, E2Phi_tambo)

E_tambo_interp = np.logspace(np.log10(E_tambo2[0]), np.log10(E_tambo2[-1]), 20)
A_tambo_interp = log_interp(E_tambo_interp, E_tambo2, A_tambo)

# TAMBO from Aperture Plot 2025 paper (5k) 
tambo_E_PeV_base = np.array([0.3e0, 0.4e0, 1e0, 2e0, 4e0, 5e0, 6e0, 7e0, 8e0, 1e1, 2e1 , 3e1,  4e1, 1e2], dtype=float)
tambo_all_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 1000, 3000, 4000, 3000, 2000, 6000, 20000, 40000, 50000], dtype=float)
tambo_tau_ap_m2sr_base = np.array([1, 9, 50, 150, 500, 600, 800, 1000, 1200, 2000, 6000, 20000, 40000, 50000], dtype=float)

E_TAMBO_2025, Eff_TAMBO_tau_2025, Eff_TAMBO_e_2025 = effective_area_from_aperture(tambo_E_PeV_base,tambo_all_ap_m2sr_base, tambo_tau_ap_m2sr_base , 0.2806)

# TAMBO from Aperture Plot new paper (5k)
tambo_E_PeV_base = np.array([1e0, 5e0, 6.3e0, 8e0, 1.2e1, 3e1, 1e2], dtype=float)
tambo_tau_ap_m2sr_base_5k = np.array([ 1e2, 1.3e3, 1.7e3, 2e3, 4e3, 1.5e4, 4e4], dtype=float)
tambo_ve_ap_m2sr_base_5k = np.array([0, 1e2, 4e3, 1e3, 1e2, 0, 0], dtype=float)
E_TAMBO_5k, Eff_TAMBO_ve_5k, Eff_TAMBO_tau_5k = effective_area_from_aperture(tambo_E_PeV_base,tambo_tau_ap_m2sr_base_5k, tambo_ve_ap_m2sr_base_5k,0.2806)

# TAMBO from Aperture Plot new paper (22k)
tambo_E_PeV_base_22k = np.array([1e0, 4.4e0, 5e0, 6.3e0, 8e0, 1.8e1, 3e1, 1e2], dtype=float)
tambo_all_ap_m2sr_base_22k = np.array([4e2, 3e3, 4.5e3, 6e3, 7e3, 2e4, 4.5e4, 1.2e5], dtype=float)
tambo_ve_ap_m2sr_base_22k = np.array([0, 1e2, 5e2, 1.5e4,  2.8e3, 1e2, 0, 0], dtype=float)
E_TAMBO_22k, Eff_TAMBO_ve_22k, Eff_TAMBO_tau_22k = effective_area_from_aperture(tambo_E_PeV_base_22k,tambo_all_ap_m2sr_base_22k, tambo_ve_ap_m2sr_base_22k, 0.2806)

# ------------------------------Other Earth-Skimming Experiments ---------------------------------

### Trinity from aperture plots (new paper)
trinity_all_PeV = np.array([1.2e0, 3e0, 6e0, 8e0, 1e1, 2e1,4e1, 7e1, 1e2], dtype=float)
trinity_all_ap_m2sr = np.array([1e2, 1.6e3, 8e3, 1.5e4,2e4, 5.5e4, 1.5e5, 3e5, 4e5], dtype=float)
trinity_ve_PeV = np.array([5e0, 6e0, 1.3e1, 2.3e1], dtype=float)
trinity_ve_ap_m2sr = np.array([1e2, 6e2, 3e2, 1e2], dtype=float)

E_trinity_new = np.logspace(np.log10(trinity_all_PeV.min()), np.log10(trinity_all_PeV.max()), 200)
AOm_trinity_all = log_interp(E_trinity_new, trinity_all_PeV, trinity_all_ap_m2sr)
AOm_trinity_ve = log_interp(E_trinity_new, trinity_ve_PeV, trinity_ve_ap_m2sr)

E_trinity_new , A_trinity_ve_ap, A_trinity_tau_ap = effective_area_from_aperture(E_trinity_new, AOm_trinity_all , AOm_trinity_ve, 0.09126)

### Trinity from Sensitivity
E_trinity   = np.array([1e15,1e16,1e17,1e18,1e19])
E2Phi_trinity = np.array([3e-8,2e-9,1e-9,2e-9,1e-8])
E_trinity, A_trinity = effective_area_from_sensitivity(
    E_trinity, E2Phi_trinity)

E_trinity_interp = np.logspace(np.log10(E_trinity[0]), np.log10(E_trinity[-1]), 100)
A_trinity_interp = log_interp(E_trinity_interp, E_trinity, A_trinity)

### GRAND 200k
E_grand   = np.array([6e16,1e17,2e17,4.6e17,4e18,4e19,1e20])
E2Phi_grand = np.array([1.2e-9,8e-10,3e-10,2e-10,4.5e-10,3e-9,7e-9])

E_grand, A_grand = effective_area_from_sensitivity(
    E_grand, E2Phi_grand)

E_grand_interp = np.logspace(np.log10(E_grand[0]), np.log10(E_grand[-1]), 100)
A_grand_interp = log_interp(E_grand_interp, E_grand, A_grand)

### AUGER
E_auger   = np.array([1e17,1e18,1e19,1e20])
E2Phi_auger = np.array([7e-8,2e-8,7e-8,4e-7])

E_auger, A_auger = effective_area_from_sensitivity(
    E_auger, E2Phi_auger)

E_auger_interp = np.logspace(np.log10(E_auger[0]), np.log10(E_auger[-1]), 100)
A_auger_interp = log_interp(E_auger_interp, E_auger, A_auger)

### POEMMA - results were quoted for 5 years only !
E_poemma   = np.array([1.5e16,2e16, 3e16,1e17,5e17,3e18,3e19, 1e20])
E2Phi_poemma = np.array([1e-5,1.6e-6, 3e-7,1.1e-7,6e-8,8e-8,3e-7, 1e-6])*0.5
E_poemma, A_poemma = effective_area_from_sensitivity(
    E_poemma, E2Phi_poemma
)

E_poemma_interp = np.logspace(np.log10(E_poemma[0]), np.log10(E_poemma[-1]), 100)
A_poemma_interp = log_interp(E_poemma_interp, E_poemma, A_poemma)

## plot effective areas for TAMBO, POEMMA, Grand200k, Trinity on common grid
plt.figure(figsize=(11, 7))
E_grand_interp_pev = E_grand_interp / GEV_PER_PEV
E_auger_interp_pev = E_auger_interp / GEV_PER_PEV
E_poemma_interp_pev = E_poemma_interp / GEV_PER_PEV
E_tambo_interp_pev = E_tambo_interp / GEV_PER_PEV
E_trinity_interp_pev = E_trinity_interp / GEV_PER_PEV

# GRAND and POEMMA
plt.loglog(E_grand_interp_pev, A_grand_interp, linewidth=PLOT_LINEWIDTH, label="GRAND 200k")
plt.loglog(E_poemma_interp_pev, A_poemma_interp, linewidth=PLOT_LINEWIDTH, label="POEMMA")
plt.loglog(E_auger_interp_pev, A_auger_interp, linewidth=PLOT_LINEWIDTH, label="AUGER")

# TAMBO 5k and 22k (new paper)
line_tambo_tau_5k, = plt.loglog(
    E_TAMBO_5k, Eff_TAMBO_tau_5k,  linewidth=PLOT_LINEWIDTH, label=r"TAMBO 5k $\nu_\tau$  "
)
plt.loglog(
    E_TAMBO_5k, Eff_TAMBO_ve_5k, ls="--", linewidth=PLOT_LINEWIDTH, color=line_tambo_tau_5k.get_color(), label=r"TAMBO 5k ${\nu}_e$"
)
line_tambo_tau_22k, = plt.loglog(
    E_TAMBO_22k, Eff_TAMBO_tau_22k, linewidth=PLOT_LINEWIDTH, label=r"TAMBO 22k $\nu_\tau$  "
)
plt.loglog(
    E_TAMBO_22k, Eff_TAMBO_ve_22k, ls="--", linewidth=PLOT_LINEWIDTH, color=line_tambo_tau_22k.get_color(), label=r"TAMBO 22k ${\nu}_e$"
)

# Trinity (new paper)
line_trinity_tau_ap, = plt.loglog(
    E_trinity_new, A_trinity_tau_ap, linewidth=PLOT_LINEWIDTH, label=r"Trinity $\nu_\tau$ "
)
plt.loglog(
    E_trinity_new, A_trinity_ve_ap, ls="--", linewidth=PLOT_LINEWIDTH, color=line_trinity_tau_ap.get_color(), label=r"Trinity ${\nu}_e$"
)

# TAMBO and Trinity from sensitivity

# plt.loglog(E_TAMBO_2025, Eff_TAMBO_tau_2025, linewidth=PLOT_LINEWIDTH, label="TAMBO (old, aperture, tau)")
# plt.loglog(E_TAMBO_2025, Eff_TAMBO_e_2025, ls="--", linewidth=PLOT_LINEWIDTH, label="TAMBO (old, aperture, e)")

# plt.loglog(E_tambo_interp_pev, A_tambo_interp, linewidth=PLOT_LINEWIDTH, label="TAMBO (sensitivity)")
# plt.loglog(E_trinity_interp_pev, A_trinity_interp, linewidth=PLOT_LINEWIDTH, label="Trinity (sensitivity)")

plt.xlabel("Energy [PeV]")
plt.ylabel("Average Effective Area [cm$^2$]")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=14, borderaxespad=0)
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.ylim(1e3, 1e10)
os.makedirs(os.path.dirname(PLOT_PNG), exist_ok=True)
plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved plot: {PLOT_PNG}")
plt.show()
