
import os
import matplotlib.pyplot as plt
import numpy as np

from Helpers import log_interp

SEC_PER_YEAR = 365.25 * 24 * 3600.0
GEV_PER_PEV = 1e6

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

E_tambo_pev = E_tambo / GEV_PER_PEV

p_vebar = A_tambo_e / (A_tambo_e + A_tambo_tau)
p_vtau  = A_tambo_tau / (A_tambo_e + A_tambo_tau)

plt.figure(figsize=(7,5))
plt.semilogx(E_tambo_pev, p_vebar, linewidth=2.5, label=r"$p(\bar{\nu}_e)$")
plt.semilogx(E_tambo_pev, p_vtau, linewidth=2.5, label=r"$p(\nu_\tau)$")

plt.xlabel("Energy [PeV]")
plt.ylabel("Channel probability")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.ylim(0, 1)
plt.tight_layout()
out_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "MC_outputs",
    "figure6.png",
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
