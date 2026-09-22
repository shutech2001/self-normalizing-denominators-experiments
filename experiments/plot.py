import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "outputs"
FIG = ROOT / "figures"
Z = float(norm.ppf(0.975))
# Scale figure dimensions and absolute style sizes together.
PLOT_SCALE = 1.5
SUPPLEMENT_FONT_SCALE = 1.2
PLOT_STYLE = {
    "font.family": "Times New Roman",
    "font.size": 10 * PLOT_SCALE,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.bfit": "Times New Roman:bold:italic",
    "mathtext.cal": "Times New Roman:italic",
    "mathtext.sf": "Times New Roman",
    "mathtext.tt": "Times New Roman",
    "mathtext.fallback": "stix",
    "axes.titlesize": 12 * PLOT_SCALE,
    "axes.titlepad": 6 * PLOT_SCALE,
    "axes.labelsize": 10 * PLOT_SCALE,
    "axes.labelpad": 4 * PLOT_SCALE,
    "axes.linewidth": 0.8 * PLOT_SCALE,
    "legend.fontsize": 10 * PLOT_SCALE,
    "xtick.labelsize": 10 * PLOT_SCALE,
    "ytick.labelsize": 10 * PLOT_SCALE,
    "xtick.major.size": 3.5 * PLOT_SCALE,
    "ytick.major.size": 3.5 * PLOT_SCALE,
    "xtick.minor.size": 2 * PLOT_SCALE,
    "ytick.minor.size": 2 * PLOT_SCALE,
    "xtick.major.width": 0.8 * PLOT_SCALE,
    "ytick.major.width": 0.8 * PLOT_SCALE,
    "xtick.minor.width": 0.6 * PLOT_SCALE,
    "ytick.minor.width": 0.6 * PLOT_SCALE,
    "xtick.major.pad": 3.5 * PLOT_SCALE,
    "ytick.major.pad": 3.5 * PLOT_SCALE,
    "xtick.minor.pad": 3.4 * PLOT_SCALE,
    "ytick.minor.pad": 3.4 * PLOT_SCALE,
    "lines.markersize": 6 * PLOT_SCALE,
    "lines.markeredgewidth": 1 * PLOT_SCALE,
    "lines.linewidth": 1.5 * PLOT_SCALE,
}
SUPPLEMENT_STYLE = {
    key: PLOT_STYLE[key] * SUPPLEMENT_FONT_SCALE
    for key in (
        "font.size",
        "axes.labelsize",
        "legend.fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    )
}


def scaled_figsize(width, height):
    return width * PLOT_SCALE, height * PLOT_SCALE


def save(fig, name, figure_dir):
    fig.tight_layout()
    fig.savefig(figure_dir / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw the seven numerical figure panels as PDF.")
    parser.add_argument("--input-dir", type=Path, default=RES)
    parser.add_argument("--figure-dir", type=Path, default=FIG)
    args = parser.parse_args()
    input_dir = args.input_dir
    figure_dir = args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(PLOT_STYLE)
    block = pd.read_csv(input_dir / "block_results.csv")
    # 1(a): fixed n,r; preserve raw, unstudentized quantile scale.
    fig, ax = plt.subplots(figsize=scaled_figsize(3.7, 2.95))
    for name, marker in [("SN", "o"), ("stable", "s")]:
        a = block.query("r==3 and n==50 and denominator==@name").sort_values("rho")
        med = a.scaled_log_q50.to_numpy()
        lo = a.scaled_log_q025.to_numpy()
        hi = a.scaled_log_q975.to_numpy()
        ax.errorbar(
            a.rho,
            med,
            yerr=np.vstack((med - lo, hi - med)),
            marker=marker,
            capsize=4 * PLOT_SCALE,
            label=name,
            linestyle="-" if name == "SN" else "--",
        )
    ax.set_xlabel(r"Canonical correlation $\rho$")
    ax.set_ylabel(r"Quantiles of $\sqrt{n}\log R_{n,D}$")
    ax.set_xticks([0, 0.5, 0.9, 0.99])
    ax.set_xticklabels(["0", "0.5", "0.9", "0.99"], rotation=40)
    ax.legend(fontsize=9 * PLOT_SCALE)
    ax.set_title("(a)")
    save(fig, "1a_quantiles", figure_dir)
    # 1(b): no true-variance studentization is used for the transplant.
    fig, ax = plt.subplots(figsize=scaled_figsize(3.8, 2.95))
    for n, marker in [(1000, "s")]:
        for den, col, lab, ls in [
            ("SN", "coverage", "SN reference", "-"),
            ("stable", "coverage", "Stable reference", "--"),
            ("stable", "delta_coverage", "Stable plug-in", "-."),
        ]:
            a = block.query("r==3 and n==@n and denominator==@den").sort_values("rho")
            p = a[col].to_numpy()
            se = np.sqrt(p * (1 - p) / a.reps.to_numpy())
            ax.errorbar(
                a.rho,
                p,
                yerr=1.96 * se,
                marker=marker,
                linestyle=ls,
                markersize=4 * PLOT_SCALE,
                capsize=2 * PLOT_SCALE,
                label=lab,
            )
    rho = np.linspace(0, 0.99, 200)
    ax.plot(rho, 2 * norm.cdf(Z / np.sqrt(1 + rho * rho)) - 1, ":", label="Stable reference limit")
    ax.set_xlabel(r"Canonical correlation $\rho$")
    ax.set_ylabel("Central calibration / coverage")
    ax.set_ylim(0.815, 0.970)
    ax.set_xticks([0, 0.5, 0.9, 0.99])
    ax.set_xticklabels(["0", "0.5", "0.9", "0.99"], rotation=40)
    ax.legend(fontsize=8 * PLOT_SCALE, loc="lower left")
    ax.set_title("(b)")
    save(fig, "1b_calibration", figure_dir)
    # 2(a): coverage and precision live on different axes, without dual axes.
    med = pd.read_csv(input_dir / "mediation_results.csv")
    fig, ax = plt.subplots(figsize=scaled_figsize(3.7, 3.05))
    order = ["1", "n^-1/2", "n^-1", "n^-2"]
    for n, marker, ls in [(100, "o", "-"), (300, "s", "--"), (1000, "^", "-.")]:
        a = med.query('law=="G" and method=="emp" and n==@n').set_index("v_name").loc[order]
        ax.errorbar(
            a.se_q50,
            a.coverage,
            yerr=1.96 * a.mcse,
            marker=marker,
            linestyle=ls,
            capsize=3 * PLOT_SCALE,
            label=f"n={n}",
        )
    ax.axhline(0.95, linestyle=":", linewidth=1 * PLOT_SCALE)
    ax.set_xscale("log")
    ax.set_xlabel("Median empirical standard error (log scale)")
    ax.set_ylabel("Empirical-studentized Wald coverage")
    ax.legend(fontsize=8 * PLOT_SCALE)
    ax.set_ylim(0.917, 0.960)
    ax.set_title("(a)")
    save(fig, "2a_precision", figure_dir)
    # 2(b): genuinely finite-sample canonical grid, not its asymptotic law.
    c = pd.read_csv(input_dir / "canonical_results.csv").query('method=="emp"')
    h = np.sort(c.h_a.unique())
    xy = np.log1p(h)
    edges = np.r_[
        xy[0] - (xy[1] - xy[0]) / 2, (xy[1:] + xy[:-1]) / 2, xy[-1] + (xy[-1] - xy[-2]) / 2
    ]
    values = (
        c.pivot(index="h_b", columns="h_a", values="coverage")
        .reindex(index=h, columns=h)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=scaled_figsize(4.0, 3.1))
    mesh = ax.pcolormesh(edges, edges, values, shading="flat", rasterized=True)
    fig.colorbar(mesh, ax=ax, pad=0.02, label="Wald coverage")
    v = np.geomspace(1, 300**-2.0, 200)
    ha = 0.8 * np.sqrt(300 / v)
    hb = np.sqrt(300 * v) / np.sqrt(0.75)
    ax.plot(
        np.log1p(ha),
        np.log1p(hb),
        linestyle="--",
        color="#D55E00",
        linewidth=2 * PLOT_SCALE,
        label="Fixed-coefficient path",
    )
    ax.plot(
        np.log1p([2]),
        np.log1p([2]),
        marker="x",
        color="tab:orange",
        markersize=6 * PLOT_SCALE,
        linestyle="none",
    )
    ticks = np.array([0, 1, 4, 16, 64, 256, 1024, 4096])
    ax.set_xticks(np.log1p(ticks))
    ax.set_xticklabels(ticks, rotation=50, fontsize=7 * PLOT_SCALE)
    ax.set_yticks(np.log1p(ticks))
    ax.set_yticklabels(ticks, fontsize=7 * PLOT_SCALE)
    ax.set_xlim(0, xy[-1])
    ax.set_ylim(0, xy[-1])
    ax.set_xlabel(r"$h_a=\sqrt{n}\,\widetilde{a}$ (log(1+h) spacing)")
    ax.set_ylabel(r"$h_b=\sqrt{n}\,\widetilde{b}$")
    ax.legend(fontsize=7 * PLOT_SCALE, loc="upper left")
    ax.set_title("(b)")
    save(fig, "2b_canonical", figure_dir)
    # Supplement: finite log-pivot corrections.
    plt.rcParams.update(SUPPLEMENT_STYLE)
    fig, ax = plt.subplots(figsize=scaled_figsize(5.6, 3.6))
    a = block.query('r==3 and rho==0 and denominator=="SN"')
    for col, marker, lab, ls in [
        ("coverage", "o", "Exact pivot", "-"),
        ("delta_coverage", "s", "Log delta", "--"),
        ("corrected_normal_coverage", "^", "Exact log moments + normal", "-."),
    ]:
        p = a[col].to_numpy()
        ax.errorbar(
            a.n,
            p,
            yerr=1.96 * np.sqrt(p * (1 - p) / a.reps.to_numpy()),
            marker=marker,
            linestyle=ls,
            capsize=3 * PLOT_SCALE,
            label=lab,
        )
    ax.axhline(0.95, linestyle=":", linewidth=1 * PLOT_SCALE)
    ax.set_xscale("log")
    ax.set_xlabel("Sample size (log scale)")
    ax.set_ylabel("Generalized-variance interval coverage")
    ax.legend()
    save(fig, "log_corrections", figure_dir)
    # Supplement: slowly vanishing lower excursion at the support endpoint.
    nap = pd.read_csv(input_dir / "napkin_results.csv")
    fig, ax = plt.subplots(figsize=scaled_figsize(5.6, 3.7))
    for name, marker, ls in [("SN", "o", "-"), ("Napkin", "s", "--")]:
        a = nap.query("denominator==@name")
        ax.errorbar(
            a.n,
            a.escape_probability,
            yerr=1.96 * a.escape_mcse,
            marker=marker,
            linestyle=ls,
            capsize=3 * PLOT_SCALE,
            label=f"{name}: |R-1|>1/2",
        )
    a = nap.query('denominator=="Napkin"')
    if "escape_lower" in a:
        ax.plot(a.n, a.escape_lower, marker="^", linestyle="-.", label="Napkin: R<1/2")
        ax.plot(
            a.n,
            a.escape_upper,
            marker="v",
            linestyle=(0, (5, 1, 1, 1, 1, 1)),
            label="Napkin: R>3/2",
        )
    ax.axhline(
        norm.sf(np.sqrt(2) - 1) + norm.cdf(-np.sqrt(2) - 1), linestyle=":", label="Napkin limit"
    )
    ax.set_xscale("log")
    ax.set_xlabel("Sample size (log scale)")
    ax.set_ylabel("Excursion probability")
    ax.legend(
        fontsize=8 * PLOT_SCALE * SUPPLEMENT_FONT_SCALE,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncols=2,
    )
    save(fig, "napkin", figure_dir)
    # Supplement: QQ plot of the actual proved Napkin limit at n=5000.
    R = np.load(input_dir / "boundary_replicates.npz")["n5000"][:, 1]
    from scipy.stats import ncx2

    probs = np.linspace(0.005, 0.995, 199)
    theory = (1 + ncx2.ppf(probs, 1, 1)) / 2
    empirical = np.quantile(R, probs)
    fig, ax = plt.subplots(figsize=scaled_figsize(4.5, 3.7))
    ax.plot(theory, empirical, ".", label="n=5000")
    ax.plot([theory[0], theory[-1]], [theory[0], theory[-1]], "--", label="Equality")
    ax.set_xlabel(r"Quantiles of $\frac{1+(1+Z)^2}{2}$")
    ax.set_ylabel("Empirical relative-denominator quantiles")
    ax.legend()
    save(fig, "napkin_qq", figure_dir)
    print("Wrote seven individual figure panels as PDF to", figure_dir)


if __name__ == "__main__":
    main()
