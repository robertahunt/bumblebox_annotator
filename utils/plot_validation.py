"""
plot_validation.py
------------------
Generate residual-uncertainty plots from frame-level validation CSV outputs.
Importable by both the standalone script and the GUI validation worker.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde, norm, laplace
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ── ECCV-friendly rcParams ──────────────────────────────────────────────────
_ECCV_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLOURS = sns.color_palette("colorblind", n_colors=10)

DOSING_MAP = {
    "19_1": 0.5, "19_2": 10,
    "7_1": 100,  "7_2": 5,
    "18_1": 20,  "18_2": 2.5,
    "15_1": 1,   "15_2": 0,
}


# ── helpers ──────────────────────────────────────────────────────────────────


def _extract_bb_num(video_id: str) -> float:
    """Parse bumblebox number from a video_id string."""
    if not video_id.startswith("bumblebox"):
        return np.nan
    return float(video_id.split("/")[-1].split("-")[1].split("_")[0])


def _extract_bumblebox_datetime(video_id: str) -> pd.Timestamp:
    """Extract timestamp from bumblebox video_id.

    Expected format example: bumblebox-15_2026-02-28_12_00_00
    Returns NaT if parsing fails.
    """
    name = str(video_id).split("/")[-1]
    if not name.startswith("bumblebox-"):
        return pd.NaT
    try:
        # Keep everything after the first underscore
        dt_raw = name.split("_", 1)[1]
        # Convert HH_MM_SS -> HH:MM:SS
        date_part, hh, mm, ss = dt_raw.split("_")
        return pd.to_datetime(f"{date_part} {hh}:{mm}:{ss}", errors="coerce")
    except Exception:
        return pd.NaT


# ── main entry point ─────────────────────────────────────────────────────────

def generate_residual_plot(
    bee_csv: Path,
    summary_csv: Path,
    output_dir: Path,
    dosing_map: dict | None = None,
    xlim: int = 100,
    cutoff_date: str | None = None,
):
    """
    Generate the residual-uncertainty figure (PDF + PNG).

    Parameters
    ----------
    bee_csv : Path
        Path to bee_detections.csv from frame-level validation.
    summary_csv : Path
        Path to frame_summary.csv from frame-level validation.
    output_dir : Path
        Directory to write residual_uncertainty.pdf and .png into.
    dosing_map : dict | None
        Mapping ``"bb_ch"`` → dose (µg).  Falls back to the built-in map.
    xlim : int
        Symmetric x-axis limit for the forest plot (panel b).
    cutoff_date : str | None
        Optional inclusive datetime cutoff for bumblebox rows (e.g. ``"2026-02-23"``
        or ``"2026-02-23 12:00:00"``). Applied only when provided.
    """
    if dosing_map is None:
        dosing_map = DOSING_MAP

    plt.rcParams.update(_ECCV_RCPARAMS)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    b = pd.read_csv(bee_csv)
    summary = pd.read_csv(summary_csv)

    # Restrict to bumblebox videos
    b = b[b["video_id"].str.startswith("bumblebox")]
    summary = summary[summary["video_id"].str.startswith("bumblebox")]

    # Optional temporal cutoff (kept out of GUI by default unless passed explicitly).
    if cutoff_date is not None:
        b["video_dt"] = b["video_id"].apply(_extract_bumblebox_datetime)
        cutoff_dt = pd.to_datetime(cutoff_date, errors="coerce")
        if pd.isna(cutoff_dt):
            raise ValueError(f"Invalid cutoff_date: {cutoff_date}")
        b = b[
            b["video_dt"].notna()
            & (b["video_dt"] >= cutoff_dt)
        ].copy()

    b["bb_num"] = b["video_id"].apply(_extract_bb_num)
    b["pred_chamber_id"] = pd.to_numeric(b["pred_chamber_id"], errors="coerce")
    b["gt_chamber_id"] = pd.to_numeric(b["gt_chamber_id"], errors="coerce")

    # Include FP/FN effects by grouping with whichever chamber ID is available.
    eval_df = b.copy()
    eval_df["eval_chamber_id"] = eval_df["gt_chamber_id"].fillna(eval_df["pred_chamber_id"])
    eval_df = eval_df[eval_df["eval_chamber_id"].notna()].copy()
    eval_df["bc_id"] = (
        eval_df["bb_num"].apply(lambda x: str(int(x)) if not pd.isna(x) else "")
        + "_ch" + eval_df["eval_chamber_id"].astype(int).astype(str)
    )

    # ── compute per-group medians and residuals ──────────────────────────────
    records = []
    group_cols = ["video_id", "frame_idx", "eval_chamber_id"]
    for (vid, fidx, ch), grp in tqdm(
        eval_df.groupby(group_cols),
        total=eval_df.groupby(group_cols).ngroups,
        desc="Aggregating",
    ):
        gt_vals = grp["gt_nearest_bee_distance"].dropna().values
        pred_vals = grp["pred_nearest_bee_distance"].dropna().values
        n_gt = len(gt_vals)
        n_pred = len(pred_vals)
        if n_gt < 3 or n_pred < 3:
            continue
        gt_med = np.median(gt_vals)
        pred_med = np.median(pred_vals)
        records.append({
            "video_id": vid, "frame_idx": fidx, "chamber_id": int(ch),
            "bb_num": grp["bb_num"].iloc[0], "bc_id": grp["bc_id"].iloc[0],
            "n_bees": min(n_gt, n_pred),
            "n_gt": n_gt,
            "n_pred": n_pred,
            "gt_median": gt_med, "pred_median": pred_med,
            "residual": gt_med - pred_med,
        })

    medians = pd.DataFrame(records)
    if len(medians) == 0:
        print("plot_validation: no frame-level groups retained — skipping plot.")
        return

    medians["abs_residual"] = medians["residual"].abs()
    medians["bc_key"] = (
        medians["bb_num"].apply(lambda x: str(int(x)))
        + "_" + medians["chamber_id"].astype(str)
    )
    medians["dose"] = medians["bc_key"].map(dosing_map)

    bb_order = sorted(medians["bb_num"].unique())
    ch_order = sorted(medians["chamber_id"].unique())

    overall_mean_residual = medians["residual"].mean()
    overall_rmse = np.sqrt((medians["residual"] ** 2).mean())
    overall_mae = medians["abs_residual"].mean()

    # ── plot ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.9, 5.5), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4], hspace=0.35)

    # ---- panel (a): KDE + fits -------------------------------------------
    ax_a = fig.add_subplot(gs[0])
    resids = medians["residual"].dropna().values
    kde = gaussian_kde(resids)
    x_grid = np.linspace(resids.min() * 1.2, resids.max() * 1.2, 300)
    y_grid = kde(x_grid)

    # Light histogram in the background
    ax_a.hist(resids, bins=50, density=True, color="0.75", edgecolor="0.85",
              linewidth=0.3, alpha=0.5, zorder=0)

    ax_a.fill_between(x_grid, y_grid, alpha=0.25, color=COLOURS[0], edgecolor="none")
    ax_a.plot(x_grid, y_grid, color=COLOURS[0], lw=1.4, label="KDE")

    mu_gauss, sigma_gauss = norm.fit(resids)
    ax_a.plot(x_grid, norm.pdf(x_grid, mu_gauss, sigma_gauss),
              color="0.25", lw=1.0, ls="--", label="Gaussian fit")

    loc_lap, scale_lap = laplace.fit(resids)
    ax_a.plot(x_grid, laplace.pdf(x_grid, loc_lap, scale_lap),
              color="0.45", lw=1.0, ls=":", label="Laplace fit")

    # ── 95 % confidence intervals (printed, not plotted) ─────────────────
    def _kde_quantile(q):
        x_fine = np.linspace(x_grid[0], x_grid[-1], 2000)
        pdf_fine = kde(x_fine)
        cdf = np.cumsum(pdf_fine) * (x_fine[1] - x_fine[0])
        cdf /= cdf[-1]
        return x_fine[np.searchsorted(cdf, q)]

    kde_lo, kde_hi = _kde_quantile(0.025), _kde_quantile(0.975)
    gauss_lo, gauss_hi = norm.ppf([0.025, 0.975], mu_gauss, sigma_gauss)
    lap_lo, lap_hi = laplace.ppf([0.025, 0.975], loc_lap, scale_lap)

    # variances
    kde_var = np.var(resids, ddof=1)
    gauss_var = sigma_gauss ** 2
    lap_var = 2 * (scale_lap ** 2)

    stats_rows = [
        ("KDE",       kde_lo, kde_hi, kde_hi - kde_lo, kde_var),
        ("Gaussian",  gauss_lo, gauss_hi, gauss_hi - gauss_lo, gauss_var),
        ("Laplace",   lap_lo, lap_hi, lap_hi - lap_lo, lap_var),
    ]

    print("\nDistribution stats:")
    print(f"  Sample variance = {kde_var:.1f} px²   (σ = {np.sqrt(kde_var):.1f} px)")
    print(f"  {'':>10s}  {'95% CI':>22s}  {'width':>8s}  {'variance':>9s}")
    for name, lo, hi, w, v in stats_rows:
        print(f"  {name:>10s}  [{lo:+.1f}, {hi:+.1f}]  {w:6.1f}  {v:9.1f}")

    # Write CSV
    stats_df = pd.DataFrame(stats_rows, columns=["distribution", "ci_lo", "ci_hi", "ci_width", "variance"])
    stats_csv = output_dir / "distribution_stats.csv"
    stats_df.to_csv(stats_csv, index=False, float_format="%.2f")
    print(f"  → stats written to {stats_csv}")

    ax_a.legend(fontsize=6.5, loc="upper left", framealpha=0.7)

    # rug
    ax_a.scatter(resids, np.zeros_like(resids) - y_grid.max() * 0.03,
                 marker="|", s=12, color="0.3", alpha=0.4, linewidths=0.5)
    ax_a.axvline(0, color="0.4", ls="--", lw=0.8, zorder=0)

    stats_str = (
        f"Mean = {overall_mean_residual:+.0f} px\n"
        f"RMSE = {overall_rmse:.0f} px\n"
        f"n = {len(medians)} frames"
    )
    ax_a.text(0.97, 0.95, stats_str, transform=ax_a.transAxes,
              ha="right", va="top", fontsize=6.5, family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.85))
    ax_a.set_xlabel("Residual  (gt median − predicted median)  [px]")
    ax_a.set_ylabel("Density")
    ax_a.set_title("(a)  Overall residual distribution", loc="left", fontweight="bold")

    # ---- panel (b): forest plot ------------------------------------------
    ax_b = fig.add_subplot(gs[1])

    combo_labels = []
    present_groups = (
        medians[["bb_num", "chamber_id"]]
        .drop_duplicates()
        .sort_values(["bb_num", "chamber_id"])
        .itertuples(index=False, name=None)
    )
    for bb, ch in present_groups:
        key = f"{int(bb)}_{int(ch)}"
        dose_str = f"{dosing_map[key]:g}" if key in dosing_map else "?"
        combo_labels.append((bb, ch, f"BB {int(bb)}  ·  Ch {int(ch)}  ({dose_str} µg)"))

    combo_labels = combo_labels[::-1]
    combo_labels.append((None, None, "  All"))
    y_positions = list(range(len(combo_labels)))

    sep_y = len(combo_labels) - 1.5
    ax_b.axhline(sep_y, color="0.5", lw=0.7, ls="-", zorder=1)

    for i, (bb, ch, label) in enumerate(combo_labels):
        if bb is None:
            subset = medians
            c = "0.15"
            marker_size, ci_lw, swarm_alpha, swarm_size = 6.5, 2.4, 0.12, 4
        else:
            subset = medians[(medians["bb_num"] == bb) & (medians["chamber_id"] == ch)]
            if len(subset) == 0:
                continue
            c = COLOURS[bb_order.index(bb) % len(COLOURS)]
            marker_size, ci_lw, swarm_alpha, swarm_size = 5, 2.0, 0.18, 6

        mu = subset["residual"].mean()
        se_mean = subset["residual"].std(ddof=1) / np.sqrt(len(subset))
        ci_lo, ci_hi = mu - 1.96 * se_mean, mu + 1.96 * se_mean

        ax_b.plot(mu, i, "D" if bb is None else "o", color=c,
                  markersize=marker_size, zorder=5, markeredgewidth=0.8,
                  markeredgecolor="white" if bb is None else c)
        ax_b.plot([ci_lo, ci_hi], [i, i], color=c, lw=ci_lw, alpha=0.85, zorder=4)

        jitter = np.random.default_rng(42 + i).uniform(-0.25, 0.25, len(subset))
        ax_b.scatter(subset["residual"], i + jitter,
                     s=swarm_size, color=c, alpha=swarm_alpha,
                     edgecolors="none", zorder=2)

    ax_b.axvline(0, color="0.4", ls="--", lw=0.8, zorder=0)

    ax_b.set_yticks(y_positions)
    ytick_labels = [lbl for _, _, lbl in combo_labels]
    ax_b.set_yticklabels(ytick_labels, fontsize=6.5)
    ax_b.get_yticklabels()[-1].set_fontweight("bold")
    ax_b.set_xlabel("Residual  (gt median − predicted median)  [px]")
    ax_b.set_title(
        "(b)  Mean residual ± 95 % CI, by bumblebox and chamber",
        loc="left", fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=COLOURS[bb_order.index(bb) % len(COLOURS)],
              label=f"BB {int(bb)}")
        for bb in bb_order
    ]
    ax_b.legend(handles=legend_handles, loc="center right", ncol=2,
                framealpha=0.85, fontsize=6.5, title="Bumblebox", title_fontsize=7)

    # ── zoom & outlier markers ───────────────────────────────────────────
    ax_b.set_xlim(-xlim, xlim)
    for i, (bb, ch, label) in enumerate(combo_labels):
        if bb is None:
            subset = medians
            c = "0.15"
        else:
            subset = medians[(medians["bb_num"] == bb) & (medians["chamber_id"] == ch)]
            if len(subset) == 0:
                continue
            c = COLOURS[bb_order.index(bb) % len(COLOURS)]
        for data, x_edge, marker in [
            (subset[subset["residual"] < -xlim], -xlim, "<"),
            (subset[subset["residual"] > xlim], xlim, ">"),
        ]:
            if len(data) > 0:
                ax_b.scatter(
                    [x_edge] * len(data), [i] * len(data),
                    s=16, marker=marker, color=c, alpha=0.55,
                    edgecolors="white", linewidths=0.3, zorder=9,
                )

    ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax_b.grid(axis="x", which="major", color="0.85", lw=0.4, zorder=0)
    ax_b.grid(axis="x", which="minor", color="0.92", lw=0.2, zorder=0)

    # ── save ─────────────────────────────────────────────────────────────────
    pdf_path = output_dir / "residual_uncertainty.pdf"
    png_path = output_dir / "residual_uncertainty.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    plt.close(fig)
    print(f"plot_validation: → {pdf_path}")
    print(f"plot_validation: → {png_path}")
