import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OMEGA = 4 * np.pi
T_EXPOSURE = 10 * 365.25 * 24 * 3600.0
energies_mid = np.array([1.0], dtype=float)
fluxMid = np.array([0.0], dtype=float)

def read_eff_area_csv_gen(path: str):
    """
    Read GEN2 effective-area CSV with columns: E_GeV, track_mu, track_e, track_tau, casc_e, casc_mu, casc_tau

    Returns:
      E (centers),
      track_triplet = (A_mu, A_tau, A_e),
      casc_triplet  = (A_mu, A_tau, A_e)
    """
    df = pd.read_csv(path)
    required = [
        "E_GeV",
        "track_mu", "track_e", "track_tau",
        "casc_e", "casc_mu", "casc_tau",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    E = df["E_GeV"].to_numpy()

    # Reorder to (mu, tau, e) to match events_per_coarse_bin_single signature used elsewhere
    track_triplet = (
        df["track_mu"].to_numpy(),
        df["track_tau"].to_numpy(),
        df["track_e"].to_numpy(),
    )
    casc_triplet = (
        df["casc_mu"].to_numpy(),
        df["casc_tau"].to_numpy(),
        df["casc_e"].to_numpy(),
    )
    return E, track_triplet, casc_triplet


def read_effarea_csv_stacked(path: str):
    """
    CSV columns (10 total):
      0: E,
      1-3:  A_mu_1, A_tau_1, A_e_1,
      4-6:  A_mu_2, A_tau_2, A_e_2,
      7-9:  A_mu_3, A_tau_3, A_e_3
    Returns:
      E_sorted, [(A_mu_1, A_tau_1, A_e_1),
                 (A_mu_2, A_tau_2, A_e_2),
                 (A_mu_3, A_tau_3, A_e_3)]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f'CSV "{path}" not found.')

    try:
        df = pd.read_csv(path, comment="#", header=None)
    except Exception:
        df = pd.read_csv(path, comment="#")

    if df.shape[1] != 10:
        raise ValueError(
            f"Expected 10 columns (E + 9 effective areas). Got {df.shape[1]}.\n"
            f"Detected columns: {list(df.columns)}"
        )

    E = np.asarray(df.iloc[:, 0], dtype=float)
    triplets = []
    for s in range(3):
        base = 1 + 3*s
        A_mu  = np.asarray(df.iloc[:, base + 0], dtype=float)
        A_tau = np.asarray(df.iloc[:, base + 1], dtype=float)
        A_e   = np.asarray(df.iloc[:, base + 2], dtype=float)
        triplets.append((A_mu, A_tau, A_e))

    order = np.argsort(E)
    E_sorted = E[order]
    triplets_sorted = [(A_mu[order], A_tau[order], A_e[order]) for (A_mu, A_tau, A_e) in triplets]
    return E_sorted, triplets_sorted


def plot_histogram(coarse_edges: np.ndarray, counts: np.ndarray, fe: float, fmu: float, ftau: float, out_png: str):
    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(lefts, counts, width=widths, align="edge", edgecolor="black", linewidth=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Energy (GeV)")
    ax.set_ylabel("Number of events per bin")
    ax.set_title(f"Events per energy bin (10 years) — fe={fe:.1f}, fμ={fmu:.1f}, fτ={ftau:.1f}")
    ax.grid(True, which="both", axis="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_2x2_histograms(coarse_edges: np.ndarray,
                        counts_list: List[np.ndarray],
                        labels: List[str],
                        out_png: str):

    lefts = coarse_edges[:-1]
    rights = coarse_edges[1:]
    widths = rights - lefts

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, counts, lab in zip(axes, counts_list, labels):
        ax.bar(lefts, counts, width=widths, align="edge", edgecolor="black", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_title(lab)
        ax.grid(True, which="both", axis="both", alpha=0.3)

    axes[2].set_xlabel("Energy (GeV)")
    axes[3].set_xlabel("Energy (GeV)")
    axes[0].set_ylabel("Events / bin")
    axes[2].set_ylabel("Events / bin")

    fig.suptitle("Events per energy bin (10 years)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(out_png, dpi=150)
    plt.close(fig)


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


def logx_interp(x, xp, yp, *, left=None, right=None, assume_sorted=False):
    x  = np.asarray(x,  dtype=float)
    xp = np.asarray(xp, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if np.any(x <= 0) or np.any(xp <= 0):
        raise ValueError("x and xp must be > 0 for log-x interpolation.")
    if not assume_sorted:
        order = np.argsort(xp)
        xp, yp = xp[order], yp[order]
    return np.interp(np.log(x), np.log(xp), yp, left=left, right=right)

def bary_to_xy(fe: np.ndarray, fmu: np.ndarray, ftau: np.ndarray):

    x = fe + 0.5 * fmu
    y = (np.sqrt(3) / 2.0) * fmu
    return x, y

POINTS = {
    "0:1:0 at source": (0.17, 0.45, 0.37),
    "1:2:0 at source": (0.30, 0.36, 0.34),
    "1:0:0 at source": (0.55, 0.17, 0.28),
}

def _normalize_bary(fe, fmu, ftau):
    s = fe + fmu + ftau
    if s <= 0:
        return fe, fmu, ftau
    return fe/s, fmu/s, ftau/s


def read_effarea_csv_radio(path: str):

    df = pd.read_csv(path, comment="#", header=None)

    E = np.asarray(df.iloc[:, 0], dtype=float)
    A_NC = np.asarray(df.iloc[:, 1], dtype=float)
    A_mu = np.asarray(df.iloc[:, 2], dtype=float)
    A_tau = np.asarray(df.iloc[:, 3], dtype=float)
    A_e  = np.asarray(df.iloc[:, 4], dtype=float)

    order = np.argsort(E)
    return E[order], A_NC[order], A_mu[order], A_tau[order], A_e[order]


def geometric_edges_from_centers(E: np.ndarray) -> np.ndarray:
    edges = np.zeros(len(E) + 1, dtype=float)
    edges[1:-1] = np.sqrt(E[:-1] * E[1:])
    edges[0]  = E[0]  / np.sqrt(E[1] / E[0])
    edges[-1] = E[-1] * np.sqrt(E[-1] / E[-2])
    return edges

def make_log_bins(edges: np.ndarray, bin_width_log10: float) -> np.ndarray:
    lo = math.log10(edges[0])
    hi = math.log10(edges[-1])
    start = lo
    stop = lo + math.ceil((hi - lo))
    raw = 10.0 ** np.arange(start, stop + 1e-12, bin_width_log10)
    raw[0] = edges[0]
    b = np.unique(raw)
    return b

def integrate_bin_I_mese(edges: np.ndarray, centers: np.ndarray, A: np.ndarray, Emin: float, Emax: float) -> float:
    total = 0.0
    for j in range(len(centers)):

        total_flux = np.interp(centers[j], energies_mid, fluxMid)
        a = edges[j]
        b = edges[j+1]
        L = max(a, Emin)
        U = min(b, Emax)
        if L < U:
            total += A[j] * total_flux * (U-L) * OMEGA* T_EXPOSURE
    return float(total)

def integrate_bin_I(edges: np.ndarray, centers: np.ndarray, A: np.ndarray, Emin: float, Emax: float, energies_Au, flux) -> float:
    total = 0.0

    for j in range(len(centers)): 

        total_flux = np.interp(centers[j], energies_Au, flux)
        a = edges[j]
        b = edges[j+1]
        L = max(a, Emin)
        U = min(b, Emax)
        if L < U:
            total += A[j] * total_flux * (U-L) * OMEGA* T_EXPOSURE
    return float(total)

def generate_flavor_grid(step: float = 0.2) -> List[Tuple[float, float, float]]:
    vals = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    out = []
    for fe in vals:
        for fmu in vals:
            ftau = 1.0 - fe - fmu
            if ftau < -1e-9:
                continue
            ftau = float(round(ftau / step) * step)
            if ftau < 0 or ftau > 1.0:
                continue
            if abs(fe + fmu + ftau - 1.0) < 1e-6:
                out.append((float(fe), float(fmu), float(ftau)))
    out = sorted(set(out))
    return out

def save_counts_csv(coarse_edges: np.ndarray, counts: np.ndarray, path: str):
    centers = np.sqrt(coarse_edges[:-1] * coarse_edges[1:])
    df = pd.DataFrame({
        "Emin_GeV": coarse_edges[:-1],
        "Emax_GeV": coarse_edges[1:],
        "Ecenter_GeV": centers,
        "N_events": counts,
    })
    df.to_csv(path, index=False)

def read_effarea_csv(path: str):
    try:
        df = pd.read_csv(path, comment="#", header=None)
        if df.shape[1] < 4:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, comment="#")

    E = np.asarray(df.iloc[:, 0], dtype=float)
    A_mu = np.asarray(df.iloc[:, 1], dtype=float)
    A_tau = np.asarray(df.iloc[:, 2], dtype=float)
    A_e  = np.asarray(df.iloc[:, 3], dtype=float)

    order = np.argsort(E)
    return E[order], A_mu[order], A_tau[order], A_e[order]

def _clip01(p):
    return np.clip(p, 1e-12, 1.0 - 1e-12)
