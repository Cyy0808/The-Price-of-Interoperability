#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bridge count-share vs amount-share + per-transfer value distribution (with amount imputation).

Imputation:
  total_amount_usd := avg_transfer_usd_value * transfer_count  (when missing)
  then fill remaining NaN with 0

Outputs (to outdir):
  1) bridge_count_vs_amount_share.png
  2) bridge_share_gap_timeseries.png
  3) bridge_amount_share_timeseries.png
  4) bridge_count_share_timeseries.png
  5) per_transfer_value_ecdf_by_type.png
  6) bridge_share_and_value_analysis.xlsx

Notes:
- "all" is treated as a normal bridge name (NOT aggregation).
- Distribution uses avg_transfer_usd_value weighted by transfer_count.
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def weighted_ecdf(values: np.ndarray, weights: np.ndarray):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return v, cw

def weighted_quantiles(values: np.ndarray, weights: np.ndarray, ps=(0.5,0.9,0.99)):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    out = {}
    for p in ps:
        out[p] = v[np.searchsorted(cw, p)]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--topk", type=int, default=8, help="top K bridges to show in time-series plots")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in ["avg_transfer_usd_value", "transfer_count", "total_amount_usd"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- amount imputation ---
    fill_val = df["avg_transfer_usd_value"] * df["transfer_count"]
    missing_before = int(df["total_amount_usd"].isna().sum())
    imputable = int(((df["total_amount_usd"].isna()) & fill_val.notna()).sum())
    df["total_amount_usd"] = df["total_amount_usd"].fillna(fill_val).fillna(0.0)
    missing_after = int(df["total_amount_usd"].isna().sum())
    print(f"[impute] total_amount_usd missing before={missing_before}, imputable={imputable}, missing after={missing_after}")

    # monthly bucket
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # bridge-month totals
    bm = (df.groupby(["month", "bridge"], as_index=False)
            .agg(transfer_count=("transfer_count","sum"),
                 total_amount_usd=("total_amount_usd","sum")))

    # monthly totals across bridges
    tot_m = bm.groupby("month", as_index=False).agg(total_count=("transfer_count","sum"),
                                                   total_amount=("total_amount_usd","sum"))
    bm = bm.merge(tot_m, on="month", how="left")
    bm["count_share"]  = np.where(bm["total_count"]>0,  bm["transfer_count"]/bm["total_count"], 0.0)
    bm["amount_share"] = np.where(bm["total_amount"]>0, bm["total_amount_usd"]/bm["total_amount"], 0.0)
    bm["share_gap"] = bm["amount_share"] - bm["count_share"]

    # overall shares across full period
    b_over = (bm.groupby("bridge", as_index=False)
                .agg(total_count=("transfer_count","sum"),
                     total_amount=("total_amount_usd","sum")))
    b_over["count_share"]  = b_over["total_count"]/b_over["total_count"].sum()
    b_over["amount_share"] = b_over["total_amount"]/b_over["total_amount"].sum()
    b_over["share_gap"] = b_over["amount_share"] - b_over["count_share"]
    b_over = b_over.sort_values("total_amount", ascending=False)

    # ---------------- Plot 1: overall scatter ----------------
    fig = plt.figure(figsize=(10,8))
    ax = plt.gca()
    ax.scatter(b_over["count_share"], b_over["amount_share"])
    mx = float(max(b_over["count_share"].max(), b_over["amount_share"].max()))
    ax.plot([0, mx], [0, mx])
    ax.set_xlabel("Share of transfer count (overall)")
    ax.set_ylabel("Share of total amount USD (overall, imputed)")
    ax.set_title("Bridge: Count Share vs Amount Share")
    for _, r in b_over.iterrows():
        ax.annotate(str(r["bridge"]), (r["count_share"], r["amount_share"]),
                    fontsize=9, xytext=(4,4), textcoords="offset points")
    fig.savefig(os.path.join(args.outdir, "bridge_count_vs_amount_share.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Plot 2: share gap over time ----------------
    top_bridges = b_over.head(args.topk)["bridge"].tolist()
    gap = bm[bm["bridge"].isin(top_bridges)].pivot(index="month", columns="bridge", values="share_gap").sort_index()

    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.plot(gap.index, gap.values)
    ax.axhline(0)
    ax.set_title(f"Share gap over time (amount_share - count_share), top {args.topk} bridges by amount")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share gap")
    ax.legend(gap.columns, ncols=2, fontsize=9)
    fig.savefig(os.path.join(args.outdir, "bridge_share_gap_timeseries.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Plot 3: amount share over time ----------------
    amt = bm[bm["bridge"].isin(top_bridges)].pivot(index="month", columns="bridge", values="amount_share").sort_index()
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.plot(amt.index, amt.values)
    ax.set_title(f"Amount share over time, top {args.topk} bridges by amount")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount share")
    ax.legend(amt.columns, ncols=2, fontsize=9)
    fig.savefig(os.path.join(args.outdir, "bridge_amount_share_timeseries.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Plot 4: count share over time ----------------
    top_count_bridges = b_over.sort_values("total_count", ascending=False).head(args.topk)["bridge"].tolist()
    cnt = bm[bm["bridge"].isin(top_count_bridges)].pivot(index="month", columns="bridge", values="count_share").sort_index()
    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    ax.plot(cnt.index, cnt.values)
    ax.set_title(f"Count share over time, top {args.topk} bridges by count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count share")
    ax.legend(cnt.columns, ncols=2, fontsize=9)
    fig.savefig(os.path.join(args.outdir, "bridge_count_share_timeseries.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # ---------------- Experiment 2: per-transfer value distribution ----------------
    bridge_lower = df["bridge"].astype(str).str.lower()
    df["bridge_type"] = np.where(
        bridge_lower.str.contains("native bridge") | (bridge_lower=="ronin") | (bridge_lower=="ronin bridge"),
        "official_or_special",
        "third_party"
    )

    dist = df[(df["transfer_count"].fillna(0)>0) &
              (df["avg_transfer_usd_value"].notna()) &
              (df["avg_transfer_usd_value"]>0)].copy()

    fig = plt.figure(figsize=(10,7))
    ax = plt.gca()
    for t, g in dist.groupby("bridge_type"):
        v = g["avg_transfer_usd_value"].to_numpy(float)
        w = g["transfer_count"].to_numpy(float)
        if w.sum() <= 0:
            continue
        xv, yv = weighted_ecdf(v, w)
        ax.plot(xv, yv, label=t)
    ax.set_xscale("log")
    ax.set_title("Weighted ECDF of per-transfer value (avg_transfer_usd_value), by bridge type")
    ax.set_xlabel("Per-transfer value (USD, log scale)")
    ax.set_ylabel("Weighted cumulative share of transfers")
    ax.legend()
    fig.savefig(os.path.join(args.outdir, "per_transfer_value_ecdf_by_type.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    # tables: weighted mean & quantiles by bridge/type
    dist["w_sum"] = dist["avg_transfer_usd_value"] * dist["transfer_count"]
    bridge_tbl = (dist.groupby("bridge", as_index=False)
                    .agg(w_sum=("w_sum","sum"), w_cnt=("transfer_count","sum")))
    bridge_tbl["wmean_transfer_value"] = bridge_tbl["w_sum"] / bridge_tbl["w_cnt"]
    bridge_tbl = bridge_tbl.merge(b_over, on="bridge", how="left").sort_values("total_amount", ascending=False)

    type_tbl = (dist.groupby("bridge_type", as_index=False)
                  .agg(w_sum=("w_sum","sum"), w_cnt=("transfer_count","sum")))
    type_tbl["wmean_transfer_value"] = type_tbl["w_sum"] / type_tbl["w_cnt"]

    # extra: weighted quantiles by type
    rows=[]
    for t, g in dist.groupby("bridge_type"):
        v=g["avg_transfer_usd_value"].to_numpy(float)
        w=g["transfer_count"].to_numpy(float)
        qs=weighted_quantiles(v,w,ps=(0.5,0.9,0.99))
        rows.append({"bridge_type":t, "p50":qs[0.5], "p90":qs[0.9], "p99":qs[0.99]})
    type_q = pd.DataFrame(rows)

    out_xlsx = os.path.join(args.outdir, "bridge_share_and_value_analysis.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        b_over.to_excel(writer, sheet_name="overall_shares", index=False)
        bridge_tbl.to_excel(writer, sheet_name="bridge_wmean_transfer", index=False)
        type_tbl.to_excel(writer, sheet_name="type_wmean_transfer", index=False)
        type_q.to_excel(writer, sheet_name="type_weighted_quantiles", index=False)
        bm.sort_values(["month","bridge"]).to_excel(writer, sheet_name="bridge_month_shares", index=False)

    print("[done] outputs written to:", args.outdir)

if __name__ == "__main__":
    main()
