## Source: Snowmass paper Figure 18 https://arxiv.org/pdf/2203.08096 at 90%CL
## QinRui Acceptance values for TAMBO and TRINITY https://arxiv.org/pdf/2607.26128

# TAMBO FOV 120*30 deg = 0.2806sr. 120 in azimuth, 30 in zenith
# Trinity FOV 5deg below horizon x 60 azimuth = 0.09126

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Helpers import log_interp


OUTPUT_CSV = "/Users/albaburgosmondejar/3_technology_paper/3_technologies/effareas_9.csv"
SEC_PER_YEAR = 365.25 * 24 * 3600.0
GEV_PER_PEV = 1e6

def effective_area_from_aperture(E_grid, Aperture_pts_tau, Aperture_pts_e, angle):

    E = np.logspace(np.log10(E_grid.min()), np.log10(E_grid.max()), 300)
    A_tau = log_interp(E, E_grid, Aperture_pts_tau)
    A_e = log_interp(E, E_grid, Aperture_pts_e)

    A_final_tau = A_tau *1e4 * angle / ( 4*np.pi ) 
    A_final_e = A_e*1e4 * angle / (  4*np.pi )

    return E, A_final_e, A_final_tau


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

# TAMBO from Acceptance (5k), energies in GeV
tambo_E_GeV_base_5k = np.array([1e0, 5e0, 6.3e0, 8e0, 1.2e1, 3e1, 1e2], dtype=float) * GEV_PER_PEV
tambo_tau_ap_m2sr_base_5k = np.array([1e2, 1.3e3, 1.7e3, 2e3, 4e3, 1.5e4, 4e4], dtype=float)
tambo_ve_ap_m2sr_base_5k = np.array([0, 1e2, 4e3, 1e3, 1e2, 0, 0], dtype=float)
E_TAMBO_5k, Eff_TAMBO_ve_5k, Eff_TAMBO_tau_5k = effective_area_from_aperture(
    tambo_E_GeV_base_5k, tambo_tau_ap_m2sr_base_5k, tambo_ve_ap_m2sr_base_5k, 0.2806)

# TAMBO from Acceptance (22k), energies in GeV
tambo_E_GeV_base_22k = np.array([1e0, 4.4e0, 5e0, 6.3e0, 8e0, 1.8e1, 3e1, 1e2], dtype=float) * GEV_PER_PEV
tambo_all_ap_m2sr_base_22k = np.array([4e2, 3e3, 4.5e3, 6e3, 7e3, 2e4, 4.5e4, 1.2e5], dtype=float)
tambo_ve_ap_m2sr_base_22k = np.array([0, 1e2, 5e2, 1.5e4, 2.8e3, 1e2, 0, 0], dtype=float)
E_TAMBO_22k, Eff_TAMBO_ve_22k, Eff_TAMBO_tau_22k = effective_area_from_aperture(
    tambo_E_GeV_base_22k, tambo_all_ap_m2sr_base_22k, tambo_ve_ap_m2sr_base_22k, 0.2806)

plt.figure(figsize=(7,5))
line_tambo_tau_5k, = plt.loglog(E_TAMBO_5k, Eff_TAMBO_tau_5k, label="TAMBO 5k τ channel")
plt.loglog(E_TAMBO_5k, Eff_TAMBO_ve_5k, ls="--", color=line_tambo_tau_5k.get_color(), label="TAMBO 5k e channel")
line_tambo_tau_22k, = plt.loglog(E_TAMBO_22k, Eff_TAMBO_tau_22k, label="TAMBO 22k τ channel")
plt.loglog(E_TAMBO_22k, Eff_TAMBO_ve_22k, ls="--", color=line_tambo_tau_22k.get_color(), label="TAMBO 22k e channel")
plt.loglog(E_tambo_interp, A_tambo_interp, "--", label="TAMBO (from sensitivity)")
plt.xlabel("Energy [GeV]")
plt.ylabel("Effective area [cm^2]")
plt.title("TAMBO Effective Area Comparison")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# Conclude: TAMBO from Snowmass paper is consistent with TAMBO paper https://arxiv.org/abs/2507.08070

# ------------------------------Other Earth-Skimming Experiments ---------------------------------

### Trinity from aperture plots (new paper), energies in GeV
trinity_all_GeV = np.array([1.2e0, 3e0, 6e0, 8e0, 1e1, 2e1, 4e1, 7e1, 1e2], dtype=float) * GEV_PER_PEV
trinity_all_ap_m2sr = np.array([1e2, 1.6e3, 8e3, 1.5e4, 2e4, 5.5e4, 1.5e5, 3e5, 4e5], dtype=float)
trinity_ve_GeV = np.array([5e0, 6e0, 1.3e1, 2.3e1], dtype=float) * GEV_PER_PEV
trinity_ve_ap_m2sr = np.array([1e2, 6e2, 3e2, 1e2], dtype=float)

E_trinity_new = np.logspace(np.log10(trinity_all_GeV.min()), np.log10(trinity_all_GeV.max()), 200)
AOm_trinity_all = log_interp(E_trinity_new, trinity_all_GeV, trinity_all_ap_m2sr)
AOm_trinity_ve = log_interp(E_trinity_new, trinity_ve_GeV, trinity_ve_ap_m2sr)

E_trinity_new, A_trinity_ve_ap, A_trinity_tau_ap = effective_area_from_aperture(
    E_trinity_new, AOm_trinity_all, AOm_trinity_ve, 0.09126)

# ### Trinity from sensitivity (kept for reference, not used)
# E_trinity   = np.array([1e15,2e15,1e16,4e16,1e17,1e18,1e19])
# E2Phi_trinity = np.array([1.2e-8,2.8e-9,1e-9,6e-10,5e-10,1.3e-9,8.5e-9])
# E_trinity, A_trinity = effective_area_from_sensitivity(
#     E_trinity, E2Phi_trinity)

# E_trinity_interp = np.logspace(np.log10(E_trinity[0]), np.log10(E_trinity[-1]), 100)
# A_trinity_interp = log_interp(E_trinity_interp, E_trinity, A_trinity)

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

# ------------------------  CSV -------------------------------------
emin =  min(E_TAMBO_5k)
emax = max(E_grand_interp)
master = np.logspace(np.log10(emin), np.log10(emax), 300)

## For All Earth-skimming exps 
A_mu_master = np.zeros(len(master))
A_tau_master = log_interp(master, E_trinity_new, A_trinity_tau_ap) + log_interp(master, E_poemma_interp, A_poemma_interp) + log_interp(master, E_TAMBO_5k,  Eff_TAMBO_tau_5k)  + log_interp(master, E_grand_interp, A_grand_interp) + log_interp(master, E_auger_interp, A_auger_interp)
A_e_master =  log_interp(master, E_TAMBO_5k,  Eff_TAMBO_ve_5k)  + log_interp(master, E_trinity_new,  A_trinity_ve_ap)

if os.path.dirname(OUTPUT_CSV):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df = pd.DataFrame({
    "E_GeV": master,
    "A_mu_m2": A_mu_master,
    "A_tau_m2": A_tau_master,
    "A_e_m2": A_e_master,
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved CSV:  {OUTPUT_CSV}")