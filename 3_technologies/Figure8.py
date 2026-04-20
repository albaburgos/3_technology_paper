## Figure 8: Radio Classification Probabilities from https://arxiv.org/pdf/2402.02432


import os

import matplotlib.pyplot as plt
import numpy as np

EV_PER_PEV = 1e15


def apply_plotting_text_style() -> None:
    """Match typography settings used in 3_technologies/Plotting.py."""
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


def smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 600,
    window: int = 11,
    logx: bool = True,
    logy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    if logx:
        mask &= x > 0
    if logy:
        mask &= y > 0
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("Need at least two valid points for interpolation.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    xu, inv = np.unique(x, return_inverse=True)
    if xu.size != x.size:
        y = np.bincount(inv, weights=y) / np.maximum(1, np.bincount(inv))
        x = xu

    xs = (
        np.logspace(np.log10(x.min()), np.log10(x.max()), n)
        if logx
        else np.linspace(x.min(), x.max(), n)
    )

    x_in = np.log10(x) if logx else x
    x_q = np.log10(xs) if logx else xs
    y_in = np.log10(y) if logy else y
    y_interp = np.interp(x_q, x_in, y_in)

    window = int(window)
    if window < 1:
        window = 1
    if window % 2 == 0:
        window += 1
    pad = window // 2

    y_pad = np.pad(y_interp, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    ys = 10.0**y_smooth if logy else y_smooth
    return xs, ys


def main() -> None:
    apply_plotting_text_style()

    # ---------- 1) nu_e CC classification vs. neutrino energy [PeV] ----------
    etruth_ev = np.array([1e17, 2e17, 4e17, 1e18, 2e18, 3e18, 4e18, 6e18, 9e18], dtype=float)
    truth = np.array([0.1, 0.22, 0.34, 0.5, 0.6, 0.63, 0.65, 0.67, 0.67], dtype=float)
    etruth_pev = etruth_ev / EV_PER_PEV
    xs1, ys1 = smooth_curve(etruth_pev, truth, n=500, logx=True, logy=False)

    # ---------- 2) Fraction of multishower events vs. neutrino energy [PeV] ----------
    emt_ev = np.array([1e17, 2e17, 3e17, 4e17, 5e17, 6e17, 9e17, 2e18, 4e18, 6e18, 9e18], dtype=float)
    mu = np.array([0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.28, 0.37, 0.42, 0.46], dtype=float)
    tau = np.array([0.035, 0.03, 0.03, 0.035, 0.04, 0.04, 0.05, 0.09, 0.14, 0.18, 0.23], dtype=float)
    emt_pev = emt_ev / EV_PER_PEV
    xsm, ysm_mu = smooth_curve(emt_pev, mu, n=500, logx=True, logy=False)
    _, ysm_tau = smooth_curve(emt_pev, tau, n=500, logx=True, logy=False)

    os.makedirs("Radio", exist_ok=True)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(17, 6.5), dpi=160)
    ax0.plot(xs1, ys1, linewidth=2, color="green", label="True positive")
    ax0.hlines(
        0.02,
        1e17 / EV_PER_PEV,
        1e19 / EV_PER_PEV,
        colors="red",
        linestyles="--",
        linewidth=2,
        label="False positive",
    )
    ax0.set_xscale("log")
    ax0.set_xlabel("Neutrino energy [PeV]")
    ax0.set_ylabel(r"$\nu_e$ CC Classification fraction")
    ax0.grid(True, which="both", linestyle=":")
    ax0.legend(loc="best")

    ax1.plot(xsm, ysm_mu, linewidth=2, label=r"$\nu_\mu$")
    ax1.plot(xsm, ysm_tau, linewidth=2, label=r"$\nu_\tau$")
    ax1.set_xscale("log")
    ax1.set_xlabel("Neutrino energy [PeV]")
    ax1.set_ylabel("Fraction of multishower events")
    ax1.grid(True, which="both", linestyle=":")
    ax1.legend(loc="best")

    fig.tight_layout()
    out_path = os.path.join("Radio", "figure11_pev_side_by_side.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
