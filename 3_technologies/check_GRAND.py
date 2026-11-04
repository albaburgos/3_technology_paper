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

### GRAND 200k
E_grand   = np.array([6e16,1e17,2e17,4.6e17,4e18,4e19,1e20])
E2Phi_grand = np.array([1.2e-9,8e-10,3e-10,2e-10,4.5e-10,3e-9,7e-9])

E_grand, A_grand = effective_area_from_sensitivity(
    E_grand, E2Phi_grand)

E_grand_interp = np.logspace(np.log10(E_grand[0]), np.log10(E_grand[-1]), 100)
A_grand_interp = log_interp(E_grand_interp, E_grand, A_grand)

### GRAND 200k check
E_grand_check   = np.array([1e8,1e9, 1e10, 1e11])
A_grand_check = np.array([1e10,1e11, 3e11, 5e11])

E_grand_interp_check = np.logspace(np.log10(E_grand_check[0]), np.log10(E_grand_check[-1]), 100)
A_grand_interp_check = log_interp(E_grand_interp_check, E_grand_check, A_grand_check)

FACTOR = (1 / 0.05233595624) * 3

fig, ax = plt.subplots(figsize=(11, 7))
E_grand_interp_pev = E_grand_interp / GEV_PER_PEV
E_grand_interp_pev_check = E_grand_interp_check / GEV_PER_PEV

ax.loglog(
    E_grand_interp_pev, A_grand_interp,
    linewidth=PLOT_LINEWIDTH, label=r"GRAND 200k $\nu_\alpha + \bar{\nu}_\alpha$ This Paper 2608.22556v1 (all-sky average)",
)
ax.loglog(
    E_grand_interp_pev, A_grand_interp * FACTOR,
    linewidth=PLOT_LINEWIDTH, ls="--", label=rf"GRAND 200k $\nu_\tau$ This Paper 2608.22556v1 (average over FOV)",
)
ax.loglog(
    E_grand_interp_pev_check, A_grand_interp_check,
    linewidth=PLOT_LINEWIDTH, label=r"GRAND 200k $\nu_\tau$ 1810.09994 (average over FOV)",
)

ax.set_xlabel("Energy [PeV]")
ax.set_ylabel("Average Effective Area [cm$^2$]")
ax.legend(loc="lower right", fontsize=14, frameon=False)
ax.grid(True, which="both", ls=":")
# ax.set_ylim(1e3, 1e10)

fig.tight_layout()
plt.show()
