"""
Extract IceCube-Gen2 effective areas from TOISE per flavor and per channel, write to CSV.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from toise import factory

OUTPUT_CSV  = "effareasgen2_toise_perchannel.csv"
OUTPUT_PLOT = "effAreaGen2.png"

ALL_TYPES = [
    "cascades",
    "double_cascades",
    "unshadowed_tracks",
    "shadowed_tracks",
    "starting_tracks",
]

CHANNEL_SHORT = {
    "cascades":          "casc",
    "double_cascades":   "dc",
    "unshadowed_tracks": "utr",
    "shadowed_tracks":   "str",
    "starting_tracks":   "start",
}

# flavor index in aeff.values: 0=νe, 1=ν̄e, 2=νμ, 3=ν̄μ, 4=ντ, 5=ν̄τ
NU_IDX = {"e": (0, 1), "mu": (2, 3), "tau": (4, 5)}

factory.set_kwargs(psi_bins={k: [0, np.pi] for k in ("tracks", "cascades", "radio")})
optical = factory.get("Fictive-Optical")


def _aeff_1d(aeff_obj):
    """(E_upper_edges, A[6, n_E]) — zenith-averaged"""
    E = aeff_obj.bin_edges[0][1:]
    A = aeff_obj.values.mean(axis=2).sum(axis=(2, 3))  
    return E, A


E_ref, _ = _aeff_1d(optical[ALL_TYPES[0]][0])

channels_data = {}
for etype in ALL_TYPES:
    E, A = _aeff_1d(optical[etype][0])
    channels_data[etype] = {
        flav: np.interp(E_ref, E, A[nu_idx]) + np.interp(E_ref, E, A[nubar_idx])
        for flav, (nu_idx, nubar_idx) in NU_IDX.items()
    }

# ── Save CSV ───────────────────────────────────────────
df_dict = {"E_GeV": E_ref}
for etype in ALL_TYPES:
    short = CHANNEL_SHORT[etype]
    for flav in ("e", "mu", "tau"):
        df_dict[f"{short}_{flav}"] = channels_data[etype][flav]

df_out = pd.DataFrame(df_dict)
df_out.to_csv(OUTPUT_CSV, index=False, float_format="%.8e")
print(f"Saved {OUTPUT_CSV}  ({len(E_ref)} energy bins, {len(ALL_TYPES)} channels)")


def read_eff_area_csv_toise_perchannel(path: str):
    df = pd.read_csv(path)
    E = df["E_GeV"].to_numpy(dtype=float)
    order = np.argsort(E)
    E = E[order]
    channel_names = [c[:-2] for c in df.columns if c.endswith("_e")]
    triplets = []
    for name in channel_names:
        A_e   = df[f"{name}_e"].to_numpy(dtype=float)[order] * 1e4
        A_mu  = df[f"{name}_mu"].to_numpy(dtype=float)[order] * 1e4
        A_tau = df[f"{name}_tau"].to_numpy(dtype=float)[order] * 1e4
        triplets.append((A_e, A_mu, A_tau))
    return E, triplets, channel_names


FLAVOR_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
FLAVOR_LABELS = [r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"]
TYPE_TITLES = {
    "cascades":          "Cascades",
    "double_cascades":   "Double cascades",
    "unshadowed_tracks": "Unshadowed tracks",
    "shadowed_tracks":   "Shadowed tracks",
    "starting_tracks":   "Starting tracks",
}

n_panels = len(ALL_TYPES) + 1
fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=True)

for ax, etype in zip(axes[:-1], ALL_TYPES):
    for flav, label, color in zip(("e", "mu", "tau"), FLAVOR_LABELS, FLAVOR_COLORS):
        ax.loglog(E_ref, channels_data[etype][flav], label=label, color=color)
    ax.set_title(TYPE_TITLES[etype], fontsize=10)
    ax.set_xlabel("Energy (GeV)")
    ax.set_xlim(1e4, 5e10)
    ax.set_ylim(1e-3, 1e6)
    ax.grid(True, which="both", alpha=0.2)

A_tot = {
    flav: sum(channels_data[et][flav] for et in ALL_TYPES)
    for flav in ("e", "mu", "tau")
}
ax_tot = axes[-1]
for flav, label, color in zip(("e", "mu", "tau"), FLAVOR_LABELS, FLAVOR_COLORS):
    ax_tot.loglog(E_ref, A_tot[flav], label=label, color=color)
ax_tot.set_title("Total\n(all channels, ν only)", fontsize=10)
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
