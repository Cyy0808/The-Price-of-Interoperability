#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Net-flow visualizations (Schemes 1/2/3/5) for cross-chain bridge datasets.

Input (CSV) must contain at least:
  - date, source_chain, destination_chain, bridge
  - transfer_count
  - total_amount_usd (may be NaN)
  - avg_transfer_usd_value (used to impute NaN total_amount_usd)

Imputation rule:
  total_amount_usd := avg_transfer_usd_value * transfer_count  (when total_amount_usd is NaN)

"Official bridges removed" filter (same as your earlier scripts):
  drop rows where bridge contains 'native bridge' OR 'ronin bridge' (case-insensitive).
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm


# -----------------------
# Data loading & cleaning
# -----------------------
def load_data(path: str, cutoff: str):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()
    df = df[df["date"] <= pd.Timestamp(cutoff)].copy()

    # numeric + impute
    df["transfer_count"] = pd.to_numeric(df.get("transfer_count"), errors="coerce").fillna(0.0)
    df["avg_transfer_usd_value"] = pd.to_numeric(df.get("avg_transfer_usd_value"), errors="coerce")
    df["total_amount_usd"] = pd.to_numeric(df.get("total_amount_usd"), errors="coerce")
    m = df["total_amount_usd"].isna()
    df.loc[m, "total_amount_usd"] = df.loc[m, "avg_transfer_usd_value"].fillna(0.0) * df.loc[m, "transfer_count"]
    df["total_amount_usd"] = df["total_amount_usd"].fillna(0.0)

    # normalize strings
    for c in ["source_chain", "destination_chain", "bridge"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    # light alias fixes (optional)
    df["source_chain"] = df["source_chain"].str.replace("sei_network", "sei", regex=False)
    df["destination_chain"] = df["destination_chain"].str.replace("sei_network", "sei", regex=False)
    df["source_chain"] = df["source_chain"].str.replace("world chain", "worldchain", regex=False)
    df["destination_chain"] = df["destination_chain"].str.replace("world chain", "worldchain", regex=False)

    # drop unknown endpoints if present
    df = df[(df["source_chain"] != "unknown") & (df["destination_chain"] != "unknown")].copy()
    return df


def remove_official_bridges(df: pd.DataFrame) -> pd.DataFrame:
    b = df["bridge"]
    return df[~(b.str.contains("native bridge") | b.str.contains("ronin bridge"))].copy()


# -----------------------
# Net flow tables
# -----------------------
def net_pairs(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Build unordered-pair table with signed net:
      i,j chosen lexicographically.
      signed_net_i_to_j = flow(i->j) - flow(j->i)
      net_abs = |signed_net_i_to_j|
    """
    g = df.groupby(["source_chain", "destination_chain"], as_index=False)[value_col].sum()
    g = g[g[value_col] > 0].copy()

    a = g["source_chain"].values
    b = g["destination_chain"].values
    i = np.where(a <= b, a, b)
    j = np.where(a <= b, b, a)
    dir_ = np.where(a <= b, "i_to_j", "j_to_i")

    g2 = pd.DataFrame({"i": i, "j": j, "dir": dir_, "val": g[value_col].values})
    piv = g2.pivot_table(index=["i", "j"], columns="dir", values="val", aggfunc="sum", fill_value=0.0).reset_index()

    for col in ["i_to_j", "j_to_i"]:
        if col not in piv.columns:
            piv[col] = 0.0

    piv["signed_net_i_to_j"] = piv["i_to_j"] - piv["j_to_i"]
    piv["net_abs"] = piv["signed_net_i_to_j"].abs()
    piv["total"] = piv["i_to_j"] + piv["j_to_i"]
    return piv.sort_values("net_abs", ascending=False).reset_index(drop=True)


def align_pairs(p_all: pd.DataFrame, p_no: pd.DataFrame) -> pd.DataFrame:
    """
    Align unordered pairs between 'all bridges' and 'official removed'.
    Direction is defined by ALL-bridges net direction for labeling.
    """
    a = p_all.copy()
    b = p_no.copy()
    a["key"] = list(zip(a["i"], a["j"]))
    b["key"] = list(zip(b["i"], b["j"]))

    m = a.merge(
        b[["key", "signed_net_i_to_j", "net_abs", "total"]],
        on="key", how="left", suffixes=("_all", "_no")
    )
    m[["signed_net_i_to_j_no", "net_abs_no", "total_no"]] = m[["signed_net_i_to_j_no", "net_abs_no", "total_no"]].fillna(0.0)

    # Label direction by ALL-bridges net sign
    m["exporter"] = np.where(m["signed_net_i_to_j_all"] >= 0, m["i"], m["j"])
    m["importer"] = np.where(m["signed_net_i_to_j_all"] >= 0, m["j"], m["i"])

    # Project NO-official signed net onto ALL-direction (can go negative if direction flips)
    m["signed_net_no_in_all_dir"] = np.where(
        m["signed_net_i_to_j_all"] >= 0,
        m["signed_net_i_to_j_no"],
        -m["signed_net_i_to_j_no"]
    )
    m["flip_dir"] = m["signed_net_no_in_all_dir"] < 0
    return m


# -----------------------
# Plot helpers
# -----------------------
def mpl_color_map(keys):
    cmap = plt.get_cmap("tab20")
    return {k: cmap(i % 20) for i, k in enumerate(keys)}


# -----------------------
# Scheme 1: Ranked corridor bars (clean, paper-friendly)
# -----------------------
def plot_scheme1_ranked_corridors(aligned: pd.DataFrame, out_png: str, metric: str, top_k: int):
    dat = aligned.copy().head(top_k).copy()
    dat["corridor"] = dat["exporter"] + " \u2192 " + dat["importer"]

    if metric == "usd":
        dat["v_all"] = dat["net_abs_all"] / 1e9
        dat["v_no"] = dat["net_abs_no"] / 1e9
        xlabel = "|Net flow| (USD, billions)"
        title = "Top net-flow corridors (USD)"
    else:
        dat["v_all"] = dat["net_abs_all"] / 1e6
        dat["v_no"] = dat["net_abs_no"] / 1e6
        xlabel = "|Net flow| (transfers, millions)"
        title = "Top net-flow corridors (transfer count)"

    dat = dat.sort_values("v_all", ascending=True)
    y = np.arange(len(dat))
    h = 0.38

    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.barh(y - h / 2, dat["v_no"], height=h, label="Official bridges removed", color="#d0e2c0")
    ax.barh(y + h / 2, dat["v_all"], height=h, label="All bridges", color="#67a583")

    labels = []
    for c, flip in zip(dat["corridor"], dat["flip_dir"]):
        labels.append(c + (" *" if flip else ""))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, fontweight="bold", color="black")

    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Scheme 2: Net-flow matrix heatmaps (All / No-official / Delta)
# -----------------------
def top_nodes_by_gross(df: pd.DataFrame, value_col: str, top_n: int):
    out = df.groupby("source_chain")[value_col].sum()
    inn = df.groupby("destination_chain")[value_col].sum()
    gross = out.add(inn, fill_value=0.0).sort_values(ascending=False)
    return gross.head(top_n).index.tolist()


def net_matrix(df: pd.DataFrame, nodes: list[str], value_col: str) -> pd.DataFrame:
    g = df[df["source_chain"].isin(nodes) & df["destination_chain"].isin(nodes)]
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
    if not g.empty:
        agg = g.groupby(["source_chain", "destination_chain"])[value_col].sum()
        for (s, d), v in agg.items():
            mat.loc[s, d] = v
    net = mat - mat.T
    np.fill_diagonal(net.values, 0.0)
    return net


def plot_scheme2_heatmaps(df_all: pd.DataFrame, df_no: pd.DataFrame, nodes: list[str], value_col: str, out_png: str, metric: str):
    net_all = net_matrix(df_all, nodes, value_col)
    net_no = net_matrix(df_no, nodes, value_col)
    delta = net_no - net_all

    if metric == "usd":
        scale = 1e9
        unit = "USD (billions)"
        supt = "Scheme 2: Net flow matrix (USD)"
    else:
        scale = 1e6
        unit = "Transfers (millions)"
        supt = "Scheme 2: Net flow matrix (count)"

    A = net_all / scale
    B = net_no / scale
    D = delta / scale

    vmax = float(max(np.abs(A.values).max(), np.abs(B.values).max(), np.abs(D.values).max(), 1e-9))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, mat, t in zip(
        axes, [A, B, D],
        ["All bridges", "Official bridges removed", "Delta (removed - all)"]
    ):
        im = ax.imshow(mat.values, cmap=cmap, norm=norm, aspect="equal")
        ax.set_xticks(range(len(nodes)))
        ax.set_yticks(range(len(nodes)))
        ax.set_xticklabels(nodes, rotation=45, ha="right")
        ax.set_yticklabels(nodes)
        ax.set_title(t)

        ax.set_xticks(np.arange(-.5, len(nodes), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(nodes), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label(f"Net flow (row \u2192 col), {unit}")
    fig.suptitle(supt + " — positive means row exports to col", y=1.02, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Scheme 3: Bipartite top-corridor diagram (clean "mini-sankey" without crossings explosion)
# -----------------------
def plot_scheme3_bipartite(pairs: pd.DataFrame, out_png: str, metric: str, top_edges: int, title: str):
    p = pairs.copy()
    p["exporter"] = np.where(p["signed_net_i_to_j"] >= 0, p["i"], p["j"])
    p["importer"] = np.where(p["signed_net_i_to_j"] >= 0, p["j"], p["i"])
    p["value"] = p["net_abs"]
    p = p[p["value"] > 0].copy()
    p = p.sort_values("value", ascending=False).head(top_edges).copy()

    if metric == "usd":
        p["v"] = p["value"] / 1e9
        vlabel = "Net USD (B)"
    else:
        p["v"] = p["value"] / 1e6
        vlabel = "Net transfers (M)"

    exp_tot = p.groupby("exporter")["v"].sum().sort_values(ascending=False)
    imp_tot = p.groupby("importer")["v"].sum().sort_values(ascending=False)
    exporters = exp_tot.index.tolist()
    importers = imp_tot.index.tolist()

    colors = mpl_color_map(exporters + importers)

    y_exp = np.linspace(0.1, 0.9, len(exporters)) if len(exporters) > 1 else np.array([0.5])
    y_imp = np.linspace(0.1, 0.9, len(importers)) if len(importers) > 1 else np.array([0.5])
    pos_exp = {e: y for e, y in zip(exporters, y_exp)}
    pos_imp = {i: y for i, y in zip(importers, y_imp)}

    vmax = float(p["v"].max()) if float(p["v"].max()) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)

    # edges
    for _, r in p.iterrows():
        e, im, v = r["exporter"], r["importer"], float(r["v"])
        y1, y2 = pos_exp[e], pos_imp[im]
        rad = 0.25 * np.sign((y2 - y1)) * (abs(y2 - y1))
        lw = 1.5 + 8.0 * (v / vmax)
        patch = FancyArrowPatch(
            (0.12, y1), (0.88, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10 + 18 * (v / vmax),
            lw=lw, color=colors[e], alpha=0.55
        )
        ax.add_patch(patch)

    # nodes
    exp_max = float(exp_tot.max()) if float(exp_tot.max()) > 0 else 1.0
    imp_max = float(imp_tot.max()) if float(imp_tot.max()) > 0 else 1.0

    for e in exporters:
        y = pos_exp[e]
        size = 80 + 420 * (float(exp_tot[e]) / exp_max)
        ax.scatter([0.08], [y], s=size, color=colors[e], edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(0.02, y, e, ha="left", va="center", fontsize=12)

    for im in importers:
        y = pos_imp[im]
        size = 80 + 420 * (float(imp_tot[im]) / imp_max)
        ax.scatter([0.92], [y], s=size, color=colors[im], edgecolors="white", linewidths=1.0, zorder=3)
        ax.text(0.98, y, im, ha="right", va="center", fontsize=12)

    ax.text(
        0.5, 0.02,
        f"Top {top_edges} corridors by |net|. Edge width \u221d {vlabel}. Color = exporter.",
        ha="center", va="bottom", fontsize=11
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Scheme 5: Simplified chord (top edges only)
# -----------------------
def compute_directed_edges(pairs: pd.DataFrame, top_edges: int):
    p = pairs.copy()
    p["exporter"] = np.where(p["signed_net_i_to_j"] >= 0, p["i"], p["j"])
    p["importer"] = np.where(p["signed_net_i_to_j"] >= 0, p["j"], p["i"])
    p["value"] = p["net_abs"]
    p = p[p["value"] > 0].copy()
    p = p.sort_values("value", ascending=False).head(top_edges).copy()
    return list(zip(p["exporter"], p["importer"], p["value"]))


def plot_chord(ax, edges, title):
    nodes = sorted(set([u for u, _, _ in edges] + [v for _, v, _ in edges]))
    if not nodes:
        ax.axis("off")
        ax.set_title(title)
        return

    inc = {k: 0.0 for k in nodes}
    for u, v, w in edges:
        inc[u] += float(w)
        inc[v] += float(w)

    nodes = sorted(nodes, key=lambda x: inc[x], reverse=True)
    n = len(nodes)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {nodes[i]: (math.cos(theta[i]), math.sin(theta[i])) for i in range(n)}

    colors = mpl_color_map(nodes)
    wmax = max(float(w) for _, _, w in edges) if edges else 1.0

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)

    for u, v, w in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        iu, iv = nodes.index(u), nodes.index(v)
        delta = (iv - iu) if iv >= iu else (iv - iu + n)
        rad = 0.25 if delta < n / 2 else -0.25

        lw = 1.0 + 8.0 * (float(w) / wmax)
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10 + 18 * (float(w) / wmax),
            lw=lw, color=colors[u], alpha=0.65,
            shrinkA=12, shrinkB=12
        )
        ax.add_patch(patch)

    incmax = max(inc.values()) if inc else 1.0
    for node in nodes:
        x, y = pos[node]
        size = 180 + 700 * (inc[node] / incmax)
        ax.scatter([x], [y], s=size, color=colors[node], edgecolors="white", linewidths=1.2, zorder=3)
        ax.text(1.15 * x, 1.15 * y, node, ha="center", va="center", fontsize=11)

    ax.text(0, -1.25, f"Edges: top {len(edges)} by |net|. Color = exporter.", ha="center", va="center", fontsize=11)


def plot_scheme5_chords(pairs_all: pd.DataFrame, pairs_no: pd.DataFrame, out_png: str, metric: str, top_edges: int):
    edges_all = compute_directed_edges(pairs_all, top_edges=top_edges)
    edges_no = compute_directed_edges(pairs_no, top_edges=top_edges)

    if metric == "usd":
        edges_all = [(u, v, w / 1e9) for u, v, w in edges_all]
        edges_no = [(u, v, w / 1e9) for u, v, w in edges_no]
        unit = "USD (B)"
    else:
        edges_all = [(u, v, w / 1e6) for u, v, w in edges_all]
        edges_no = [(u, v, w / 1e6) for u, v, w in edges_no]
        unit = "Transfers (M)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_chord(axes[0], edges_all, f"All bridges ({unit})")
    plot_chord(axes[1], edges_no, f"Official removed ({unit})")
    fig.suptitle("Scheme 5: Simplified chord view of net flows", y=1.02, fontsize=16, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_png, dpi=260, bbox_inches="tight")
    plt.close(fig)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV path (e.g., final_bridge_data_layerzero_cleaned.csv)")
    ap.add_argument("--outdir", default="./netflow_options_1235_out")
    ap.add_argument("--cutoff", default="2025-10-31")
    ap.add_argument("--top_corridors", type=int, default=12, help="Scheme1: number of corridors")
    ap.add_argument("--top_nodes", type=int, default=12, help="Scheme2: heatmap size (top nodes by gross)")
    ap.add_argument("--top_edges", type=int, default=16, help="Scheme3: top edges")
    ap.add_argument("--top_edges_chord", type=int, default=12, help="Scheme5: top edges")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # style
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    df = load_data(args.input, cutoff=args.cutoff)
    df_no = remove_official_bridges(df)

    # Pair tables
    pairs_all_amt = net_pairs(df, "total_amount_usd")
    pairs_no_amt = net_pairs(df_no, "total_amount_usd")
    pairs_all_ct = net_pairs(df, "transfer_count")
    pairs_no_ct = net_pairs(df_no, "transfer_count")

    aligned_amt = align_pairs(pairs_all_amt, pairs_no_amt)
    aligned_ct = align_pairs(pairs_all_ct, pairs_no_ct)

    # Scheme 1
    plot_scheme1_ranked_corridors(aligned_amt, os.path.join(args.outdir, "scheme1_ranked_corridors_usd.png"), metric="usd", top_k=args.top_corridors)
    plot_scheme1_ranked_corridors(aligned_ct, os.path.join(args.outdir, "scheme1_ranked_corridors_count.png"), metric="count", top_k=args.top_corridors)

    # # Scheme 2
    # nodes_amt = top_nodes_by_gross(df, "total_amount_usd", top_n=args.top_nodes)
    # plot_scheme2_heatmaps(df, df_no, nodes_amt, "total_amount_usd", os.path.join(args.outdir, "scheme2_heatmap_usd.png"), metric="usd")

    # nodes_ct = top_nodes_by_gross(df, "transfer_count", top_n=args.top_nodes)
    # plot_scheme2_heatmaps(df, df_no, nodes_ct, "transfer_count", os.path.join(args.outdir, "scheme2_heatmap_count.png"), metric="count")

    # # Scheme 3
    # plot_scheme3_bipartite(pairs_all_amt, os.path.join(args.outdir, "scheme3_bipartite_usd_all.png"), metric="usd", top_edges=args.top_edges,
    #                        title="Scheme 3: Top net corridors (All bridges, USD)")
    # plot_scheme3_bipartite(pairs_no_amt, os.path.join(args.outdir, "scheme3_bipartite_usd_no_official.png"), metric="usd", top_edges=args.top_edges,
    #                        title="Scheme 3: Top net corridors (Official removed, USD)")

    # plot_scheme3_bipartite(pairs_all_ct, os.path.join(args.outdir, "scheme3_bipartite_count_all.png"), metric="count", top_edges=args.top_edges,
    #                        title="Scheme 3: Top net corridors (All bridges, count)")
    # plot_scheme3_bipartite(pairs_no_ct, os.path.join(args.outdir, "scheme3_bipartite_count_no_official.png"), metric="count", top_edges=args.top_edges,
    #                        title="Scheme 3: Top net corridors (Official removed, count)")

    # # Scheme 5
    # plot_scheme5_chords(pairs_all_amt, pairs_no_amt, os.path.join(args.outdir, "scheme5_chord_usd.png"), metric="usd", top_edges=args.top_edges_chord)
    # plot_scheme5_chords(pairs_all_ct, pairs_no_ct, os.path.join(args.outdir, "scheme5_chord_count.png"), metric="count", top_edges=args.top_edges_chord)

    print("Done. Outputs in:", args.outdir)


if __name__ == "__main__":
    main()
