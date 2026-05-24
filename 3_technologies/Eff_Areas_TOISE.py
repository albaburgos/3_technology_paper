"""
Extract IceCube-Gen2 total effective areas from TOISE by flavor and write to CSV.

"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from toise import factory

OUTPUT_CSV  = "effareasgen2_toise_noanti.csv"
OUTPUT_PLOT = "effAreaGen2.png"

ALL_TYPES = [
             "cascades", "double_cascades", "unshadowed_tracks", "shadowed_tracks"]

#  "unshadowed_tracks", "shadowed_tracks"
# shadowed tracks, unshadowed tracks, double cascades, strating tracks, cascades
# flavor index in aeff.values: 0=νe, 1=ν̄e, 2=νμ, 3=ν̄μ, 4=ντ, 5=ν̄τ

# ── TOISE setup ────────────────────────────────────────────────────────────────
factory.set_kwargs(psi_bins={k: [0, np.pi] for k in ("tracks", "cascades", "radio")})
optical = factory.get("Fictive-Optical")


def _aeff_1d(aeff_obj):
    """(E_upper_edges, A[6, n_E]) — zenith-averaged, reco-summed."""
    E = aeff_obj.bin_edges[0][1:]
    A = aeff_obj.values.mean(axis=2).sum(axis=(2, 3))  # (6, n_E)
    return E, A


# ── Sum all event types onto a common energy grid ─────────────────────────────
E_ref, A_tot = _aeff_1d(optical[ALL_TYPES[0]][0])
for etype in ALL_TYPES[1:]:
    E, A = _aeff_1d(optical[etype][0])
    A_tot += np.array([np.interp(E_ref, E, A[i]) for i in range(6)])

# Average ν + ν̄ per flavor
A_e   = (A_tot[0])
A_mu  = (A_tot[2])
A_tau = (A_tot[4])

# ── Save CSV ───────────────────────────────────────────────────────────────────
np.savetxt(OUTPUT_CSV,
           np.column_stack([E_ref, A_e, A_mu, A_tau]),
           delimiter=",", header="E_GeV,A_e,A_mu,A_tau", comments="")
print(f"Saved {OUTPUT_CSV}  ({len(E_ref)} energy bins)")


# ── Reader ─────────────────────────────────────────────────────────────────────
def read_eff_area_csv_toise(path: str):
    """
    Read TOISE GEN2 total effective-area CSV.
    Columns: E_GeV, A_e, A_mu, A_tau
    """
    df = pd.read_csv(path)
    return (df["E_GeV"].to_numpy(),
            df["A_e"].to_numpy(),
            df["A_mu"].to_numpy(),
            df["A_tau"].to_numpy())


# ── Verification plot: 5 subplots by event type + 1 total ────────────────────
FLAVOR_COLORS  = ["#e41a1c", "#377eb8", "#4daf4a"]
FLAVOR_LABELS  = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
TYPE_TITLES    = {
    "starting_tracks":   "Starting tracks",
    "cascades":          "Cascades",
    "double_cascades":   "Double cascades",
    "unshadowed_tracks": "Unshadowed tracks",
    "shadowed_tracks":   "Shadowed tracks",
}

fig, axes = plt.subplots(1, len(ALL_TYPES) + 1, figsize=(4 * (len(ALL_TYPES) + 1), 5),
                          sharey=True)

for ax, etype in zip(axes[:-1], ALL_TYPES):
    E, A = _aeff_1d(optical[etype][0])
    for idx, (label, color) in enumerate(zip(FLAVOR_LABELS, FLAVOR_COLORS)):
        A_flavor = 0.5 * (A[2 * idx] + A[2 * idx + 1])
        ax.loglog(E, A_flavor, label=label, color=color)
    ax.set_title(TYPE_TITLES.get(etype, etype), fontsize=10)
    ax.set_xlabel("Energy (GeV)")
    ax.set_xlim(1e4, 5e10)
    ax.set_ylim(1e-3, 1e6)
    ax.grid(True, which="both", alpha=0.2)

ax_tot = axes[-1]
for A_flavor, label, color in zip([A_e, A_mu, A_tau], FLAVOR_LABELS, FLAVOR_COLORS):
    ax_tot.loglog(E_ref, A_flavor, label=label, color=color)
ax_tot.set_title("Saved to CSV\n(ν only, all types)", fontsize=10)
ax_tot.set_xlabel("Energy (GeV)")
ax_tot.set_xlim(1e4, 5e10)
ax_tot.grid(True, which="both", alpha=0.2)
ax_tot.legend(fontsize=9)

axes[0].set_ylabel(r"Effective area (m$^2$)")
fig.suptitle("IceCube-Gen2 effective area by event type and flavor", y=1.01)
fig.tight_layout()
fig.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUTPUT_PLOT}")
