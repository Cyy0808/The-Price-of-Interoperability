#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def compute_bridge_order(gb, piv_cols, pivA_cols, other_label="Other"):
    # gb: long df with columns [time, bridge, count, amount]
    agg = gb.groupby("bridge", as_index=True)[["count", "amount"]].sum()

    # 只考虑会出现在图里的桥（topk+Other后的列集合）
    cols = [c for c in sorted(set(piv_cols) | set(pivA_cols)) if c != other_label]

    # 归一化后综合（避免 amount 压倒 count）
    sub = agg.reindex(cols).fillna(0.0)
    c = sub["count"].to_numpy()
    a = sub["amount"].to_numpy()
    c_norm = c / (c.max() + 1e-12)
    a_norm = a / (a.max() + 1e-12)

    score = 0.5 * c_norm + 0.5 * a_norm   # 你也可以改成 0.4/0.6 更偏向 amount
    order = [cols[i] for i in np.argsort(score)]  # 从小到大 => bottom 到 top

    # Other 放最底（或最顶都行，但两图必须一致）
    return [other_label] + order


def impute_total_amount_usd(df):
    # total_amount_usd missing -> avg_transfer_usd_value * transfer_count
    for c in ["transfer_count", "total_amount_usd", "avg_transfer_usd_value"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["transfer_count"] = df["transfer_count"].fillna(0.0)

    missing_before = int(df["total_amount_usd"].isna().sum())
    imputable = df["total_amount_usd"].isna() & df["avg_transfer_usd_value"].notna()
    imputable_n = int(imputable.sum())

    df.loc[imputable, "total_amount_usd"] = df.loc[imputable, "avg_transfer_usd_value"] * df.loc[imputable, "transfer_count"]
    df["total_amount_usd"] = df["total_amount_usd"].fillna(0.0)

    missing_after = int(df["total_amount_usd"].isna().sum())
    return df, (missing_before, imputable_n, missing_after)

def add_time_buckets(df):
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df

def topk_with_other(g, key_col, topk, label_other="Other"):
    # g: df grouped by time with category+value
    # returns pivoted df: time x (topk+Other)
    totals = g.groupby("bridge")[key_col].sum().sort_values(ascending=False)
    top = totals.head(topk).index.tolist()
    g2 = g.copy()
    g2["bridge2"] = np.where(g2["bridge"].isin(top), g2["bridge"], label_other)
    out = g2.groupby(["time","bridge2"], as_index=False)[key_col].sum()
    pivot = out.pivot(index="time", columns="bridge2", values=key_col).fillna(0.0)
    return pivot

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

def plot_weekly_totals(df, outdir):
    # Basic characteristics: weekly total transfer_count and weekly total amount (raw vs filled)
    g = df.groupby("week", as_index=False).agg(
        total_count=("transfer_count", "sum"),
        total_amount_filled=("total_amount_usd", "sum"),
    )

    # raw total_amount_usd before fill cannot be recovered now (already filled in df)
    # so we approximate "raw" by summing rows where original total_amount_usd existed:
    # We'll recompute using a copy without fill in main() before calling this, but to keep script simple:
    # we store raw in df["_raw_total_amount_usd"] if present.
    if "_raw_total_amount_usd" in df.columns:
        g_raw = df.groupby("week", as_index=False).agg(total_amount_raw=("_raw_total_amount_usd","sum"))
        g = g.merge(g_raw, on="week", how="left")
    else:
        g["total_amount_raw"] = np.nan

    plt.figure(figsize=(12,4))
    plt.plot(g["week"], g["total_count"])
    plt.title("Weekly total transfer_count")
    plt.xlabel("Week")
    plt.ylabel("Count")
    save_fig(os.path.join(outdir, "01_weekly_total_transfer_count.png"))

    plt.figure(figsize=(12,4))
    if g["total_amount_raw"].notna().any():
        plt.plot(g["week"], g["total_amount_raw"], label="raw total_amount_usd (NA ignored)")
        plt.legend()
    plt.plot(g["week"], g["total_amount_filled"], label="filled total_amount_usd")
    plt.title("Weekly total_amount_usd (raw vs filled)")
    plt.xlabel("Week")
    plt.ylabel("USD")
    if g["total_amount_raw"].notna().any():
        plt.legend()
    save_fig(os.path.join(outdir, "02_weekly_total_amount_raw_vs_filled.png"))

def get_professional_palette(all_bridges):
    # 1. 定义核心配色字典（手动指定重点桥梁的颜色）
    # 颜色选择：深蓝、钢蓝、青色、浅灰、以及一个暖色调作为点缀
    color_map = {                     
                 
            "wormhole": "#a1d8e8",
            "polygon native bridge": "#ffcca6",
            "optimism native bridge": "#c85e62", 
            "layerzero": "#f47254",
            "hyperliquid native bridge": "#f59c7c", 
            "debridge": "#d0e2c0",
            "cctp_v1": "#a2c986",
            "base standard bridge": "#7b95c6",  
            "arbitrum native bridge": "#67a583",  
            "across": "#49c2d9",          
            "Other": "#fded95",                
        }
    
    manual_colors = {
        "arbitrum": "#49c2d9", "base": "#67a583", "optimism": "#a2c986",
        "polygon": "#7b95c6", "avalanche": "#f47254", "bsc": "#d0e2c0",
        "ethereum": "#f59c7c", "aptos": "#c85e62", "linea": "#ffcca6",
        "hyperliquid": "#a1d8e8", 
        "other": "#fded95" 
    }

    # 将颜色分配给桥。如果桥的数量超过颜色数量，使用取模运算循环使用颜色。
    # color_map = {
    #     bridge: custom_hex_colors[i % len(custom_hex_colors)]
    #     for i, bridge in enumerate(all_bridges)
    # }

            
    return color_map


def plot_bridge_timeseries(df, outdir, topk, drop_bridges=("cctp_both","base native bridge")):
    # 1. 预处理数据
    gb = df.groupby(["week", "bridge"], as_index=False).agg(
        count=("transfer_count", "sum"),
        amount=("total_amount_usd", "sum"),
    ).rename(columns={"week": "time"})

    gb["time"] = pd.to_datetime(gb["time"])

    # --- NEW: 忽略指定 bridge（默认 cctp_both） ---
    gb["bridge"] = gb["bridge"].astype(str).str.strip().str.lower()
    drop_set = {b.strip().lower() for b in drop_bridges}
    gb = gb[~gb["bridge"].isin(drop_set)].copy()

    # 生成透视表
    piv = topk_with_other(gb, "count", topk)
    pivA = topk_with_other(gb, "amount", topk)

    # 2. 颜色映射
    all_bridges = sorted(list(set(piv.columns) | set(pivA.columns)))
    color_map = get_professional_palette(all_bridges)

    # --- 统一格式的绘图函数 ---
    def draw_stack_plot(data, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(16, 5.5))
        current_colors = [color_map[col] for col in data.columns]

        ax.stackplot(
            data.index, data.T.values,
            labels=data.columns,
            colors=current_colors,
            linewidth=0.0,
            alpha=1
        )

        ax.set_title(title, pad=25, fontsize=18, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)

        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.xticks(rotation=0, ha='center')

        ax.grid(axis="y", alpha=0.15, linestyle='--', color='gray')

        if ylabel.lower() == "share":
            ax.set_ylim(0, 1.0)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=14
        )

        plt.tight_layout()
        save_path = os.path.join(outdir, filename)
        plt.savefig(save_path, dpi=260, bbox_inches="tight")
        print(f"成功保存: {save_path}")
        plt.close(fig)

    # 3. 开始绘图
    draw_stack_plot(piv, f"Weekly transfer_count by bridge (top {topk} + Other)", "Count",
                    "03_bridge_weekly_count_stacked.png")

    share = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    draw_stack_plot(share, f"Weekly share of transfer_count by bridge (top {topk} + Other)", "Share",
                    "04_bridge_weekly_count_share_stacked.png")

    draw_stack_plot(pivA, f"Weekly total_amount_usd by bridge (top {topk} + Other)", "USD",
                    "05_bridge_weekly_amount_stacked.png")

    shareA = pivA.div(pivA.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    draw_stack_plot(shareA, f"Weekly share of amount by bridge (top {topk} + Other)", "Share",
                    "06_bridge_weekly_amount_share_stacked.png")

def plot_chain_endpoint_activity(df, outdir, topk=10):
    # Chains: endpoint activity = out + in per week
    out = df.groupby(["week","source_chain"], as_index=False).agg(
        out_count=("transfer_count","sum"),
        out_amount=("total_amount_usd","sum"),
    ).rename(columns={"source_chain":"chain"})
    inc = df.groupby(["week","destination_chain"], as_index=False).agg(
        in_count=("transfer_count","sum"),
        in_amount=("total_amount_usd","sum"),
    ).rename(columns={"destination_chain":"chain"})

    m = out.merge(inc, on=["week","chain"], how="outer").fillna(0.0)
    m["endpoint_count"] = m["out_count"] + m["in_count"]
    m["endpoint_amount"] = m["out_amount"] + m["in_amount"]

    # pick top chains by total endpoint_count
    topchains = m.groupby("chain")["endpoint_count"].sum().sort_values(ascending=False).head(topk).index.tolist()
    m2 = m[m["chain"].isin(topchains)].copy()

    # count stacked
    piv = m2.pivot(index="week", columns="chain", values="endpoint_count").fillna(0.0)
    plt.figure(figsize=(12,5))
    plt.stackplot(piv.index, piv.T.values, labels=piv.columns)
    plt.title(f"Top {topk} chains by endpoint activity (weekly transfer_count)")
    plt.xlabel("Week")
    plt.ylabel("Count")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    save_fig(os.path.join(outdir, "09_chain_endpoint_count_top10_2.png"))

    share = piv.div(piv.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)
    plt.figure(figsize=(12,5))
    plt.stackplot(share.index, share.T.values, labels=share.columns)
    plt.title(f"Top {topk} chains weekly share of endpoint transfer_count")
    plt.xlabel("Week")
    plt.ylabel("Share")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    save_fig(os.path.join(outdir, "10_chain_endpoint_count_share_top10_2.png"))

    # amount stacked
    pivA = m2.pivot(index="week", columns="chain", values="endpoint_amount").fillna(0.0)
    plt.figure(figsize=(12,5))
    plt.stackplot(pivA.index, pivA.T.values, labels=pivA.columns)
    plt.title(f"Top {topk} chains by endpoint activity (weekly total_amount_usd filled)")
    plt.xlabel("Week")
    plt.ylabel("USD")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    save_fig(os.path.join(outdir, "11_chain_endpoint_amount_top10_2.png"))

    shareA = pivA.div(pivA.sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0)
    plt.figure(figsize=(12,5))
    plt.stackplot(shareA.index, shareA.T.values, labels=shareA.columns)
    plt.title(f"Top {topk} chains weekly share of endpoint amount (filled)")
    plt.xlabel("Week")
    plt.ylabel("Share")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    save_fig(os.path.join(outdir, "12_chain_endpoint_amount_share_top10_2.png"))

def official_removed(df):
    # remove all "*native bridge*" and "*ronin bridge*"
    b = df["bridge"].astype(str).str.lower()
    mask = b.str.contains("native bridge") | b.str.contains("ronin bridge")
    return df[~mask].copy()

def compute_chain_flow_summary(df, out_csv):
    # aggregate by (source,dest)
    g = df.groupby(["source_chain","destination_chain"], as_index=False).agg(
        flow=("total_amount_usd","sum"),
        count=("transfer_count","sum")
    )
    # build symmetric summary table for top pairs by total flow
    # choose top 25 directed edges
    g_sorted = g.sort_values("flow", ascending=False)
    g_sorted.to_csv(out_csv, index=False)
    return g_sorted



def plot_layering(df, outdir, topk=8):
    # ---------------------------------------------------------
    # (1) Scatter: count share vs amount share (Full Sample)
    # ---------------------------------------------------------
    g = df.groupby("bridge", as_index=False).agg(
        count=("transfer_count", "sum"),
        amount=("total_amount_usd", "sum")
    )
    g["count_share"] = g["count"] / g["count"].sum() if g["count"].sum() > 0 else 0.0
    g["amount_share"] = g["amount"] / g["amount"].sum() if g["amount"].sum() > 0 else 0.0

    # 散点图通常推荐正方形比例以观察 y=x 线的偏离
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(g["count_share"], g["amount_share"], alpha=1, s=60, color='#7b95c6', edgecolors='white')
    
    # 标注 Top 桥梁 (分位数 0.9 以上)
    for _, r in g.iterrows():
        if r["amount_share"] > g["amount_share"].quantile(0.80) or r["count_share"] > g["count_share"].quantile(0.90):
            ax.text(r["count_share"] + 0.005, r["amount_share"] + 0.005, r["bridge"], fontsize=14)
    
    mx = max(g["count_share"].max(), g["amount_share"].max()) * 1.1
    ax.plot([0, mx], [0, mx], color='gray', linestyle='--', alpha=0.5, label="y=x (Balance)")
    
    ax.set_xlabel("Share of transfer count", fontsize=16, fontweight='bold')
    ax.set_ylabel("Share of USD notional", fontsize=16, fontweight='bold')
    ax.set_title("Bridge layering: Count share vs Amount share", pad=25, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(alpha=0.15, linestyle='--')
    
    plt.tight_layout()
    save_path_scatter = os.path.join(outdir, "13_bridge_count_share_vs_amount_share.png")
    plt.savefig(save_path_scatter, dpi=260, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------
    # (2) Share gap over time (Monthly), top bridges by amount
    # ---------------------------------------------------------
    m = df.groupby(["month", "bridge"], as_index=False).agg(
        count=("transfer_count", "sum"),
        amount=("total_amount_usd", "sum")
    )
    tot = m.groupby("month", as_index=False).agg(
        total_count=("count", "sum"),
        total_amount=("amount", "sum")
    )
    m = m.merge(tot, on="month", how="left")
    m["count_share"] = np.where(m["total_count"] > 0, m["count"] / m["total_count"], 0.0)
    m["amount_share"] = np.where(m["total_amount"] > 0, m["amount"] / m["total_amount"], 0.0)
    m["share_gap"] = m["amount_share"] - m["count_share"]

    topb = m.groupby("bridge")["amount"].sum().sort_values(ascending=False).head(topk).index.tolist()
    m2 = m[m["bridge"].isin(topb)].copy()

    # 线图采用参考程序的宽幅比例 (16x5.5)
    fig, ax = plt.subplots(figsize=(16, 5.5))
    
    # 建议配色方案
    manual_colors = {
        "arbitrum": "#49c2d9", "base": "#67a583", "optimism": "#a2c986",
        "polygon": "#7b95c6", "avalanche": "#f47254", "bsc": "#d0e2c0",
        "ethereum": "#f59c7c", "aptos": "#c85e62", "linea": "#ffcca6"
    }
    tab10 = plt.get_cmap("tab10").colors

    for i, b in enumerate(topb):
        d = m2[m2["bridge"] == b].sort_values("month")
        color = manual_colors.get(b.lower(), tab10[i % 10])
        ax.plot(d["month"], d["share_gap"], label=b, lw=1.5, marker='o', markersize=3, color=color)
    
    ax.axhline(0, lw=1.5, color='black', alpha=0.3) # 零基准线
    
    # 样式设置
    ax.set_title("Share gap over time (amount_share - count_share)", pad=25, fontsize=18, fontweight='bold')
    ax.set_xlabel("Month", fontsize=16, fontweight='bold')
    ax.set_ylabel("Share gap", fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # 年度坐标轴格式化
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=0, ha='center')

    ax.grid(axis="y", alpha=0.15, linestyle='--', color='gray')

    # --- 图例放在右侧外侧 ---
    ax.legend(
        loc="center left", 
        bbox_to_anchor=(1.01, 0.5), 
        frameon=False, 
        fontsize=14,
        ncol=1
    )

    plt.tight_layout()
    save_path_line = os.path.join(outdir, "14_share_gap_over_time_top_bridges.png")
    plt.savefig(save_path_line, dpi=260, bbox_inches="tight")
    print(f"成功保存分层分析图表至: {outdir}")
    plt.close(fig)

def chain_inflow_outflow_hhi(df, outdir):
    # For each week and each chain:
    # outflow HHI: distribution across destinations
    # inflow HHI: distribution across sources
    # Then aggregate across chains weighted by total outflow/inflow volume
    # We'll do count-based and amount-based, output both charts.

    # out
    out = df.groupby(["week","source_chain","destination_chain"], as_index=False).agg(
        count=("transfer_count","sum"),
        amount=("total_amount_usd","sum")
    ).rename(columns={"source_chain":"chain","destination_chain":"peer"})

    # inflow: swap role
    inn = df.groupby(["week","destination_chain","source_chain"], as_index=False).agg(
        count=("transfer_count","sum"),
        amount=("total_amount_usd","sum")
    ).rename(columns={"destination_chain":"chain","source_chain":"peer"})

    def per_chain_hhi(d, valcol):
        # compute per (week,chain) HHI across peers
        tot = d.groupby(["week","chain"], as_index=False).agg(total=(valcol,"sum"))
        d2 = d.merge(tot, on=["week","chain"], how="left")
        d2["share"] = np.where(d2["total"]>0, d2[valcol]/d2["total"], 0.0)
        h = d2.groupby(["week","chain"], as_index=False).agg(
            hhi=("share", lambda s: float(np.sum(np.square(s)))),
            total=("total","first")
        )
        return h

    out_hhi_c = per_chain_hhi(out, "count")
    in_hhi_c  = per_chain_hhi(inn, "count")
    out_hhi_a = per_chain_hhi(out, "amount")
    in_hhi_a  = per_chain_hhi(inn, "amount")

    def weighted_avg(h):
        # weighted by total
        g = h.groupby("week").apply(lambda x: float(np.average(x["hhi"], weights=x["total"])) if x["total"].sum()>0 else np.nan)
        return g.reset_index(name="weighted_hhi")

    wc_out = weighted_avg(out_hhi_c).rename(columns={"weighted_hhi":"outflow_hhi_count"})
    wc_in  = weighted_avg(in_hhi_c).rename(columns={"weighted_hhi":"inflow_hhi_count"})
    wa_out = weighted_avg(out_hhi_a).rename(columns={"weighted_hhi":"outflow_hhi_amount"})
    wa_in  = weighted_avg(in_hhi_a).rename(columns={"weighted_hhi":"inflow_hhi_amount"})

    panel = wc_out.merge(wc_in,on="week").merge(wa_out,on="week").merge(wa_in,on="week")
    panel.to_csv(os.path.join(outdir, "17_weekly_chain_inout_hhi.csv"), index=False)

    plt.figure(figsize=(12,5))
    plt.plot(panel["week"], panel["outflow_hhi_count"], label="outflow HHI (count)")
    plt.plot(panel["week"], panel["inflow_hhi_count"], label="inflow HHI (count)")
    plt.title("Chain concentration: outflow vs inflow HHI (count)")
    plt.xlabel("Week")
    plt.ylabel("Weighted HHI")
    plt.legend()
    save_fig(os.path.join(outdir, "17_chain_out_in_hhi_count.png"))

    plt.figure(figsize=(12,5))
    plt.plot(panel["week"], panel["outflow_hhi_amount"], label="outflow HHI (amount)")
    plt.plot(panel["week"], panel["inflow_hhi_amount"], label="inflow HHI (amount)")
    plt.title("Chain concentration: outflow vs inflow HHI (amount)")
    plt.xlabel("Week")
    plt.ylabel("Weighted HHI")
    plt.legend()
    save_fig(os.path.join(outdir, "18_chain_out_in_hhi_amount.png"))

def sankey_outputs(df, outdir, top_edges=40):
    # We output CSV tables always; if plotly is available we also output HTML sankey.
    g = df.groupby(["source_chain","destination_chain"], as_index=False).agg(
        amount=("total_amount_usd","sum"),
        count=("transfer_count","sum")
    )
    g_amt = g.sort_values("amount", ascending=False).head(top_edges)
    g_cnt = g.sort_values("count", ascending=False).head(top_edges)
    g_amt.to_csv(os.path.join(outdir, "21_top_chainpair_edges_by_amount.csv"), index=False)
    g_cnt.to_csv(os.path.join(outdir, "22_top_chainpair_edges_by_count.csv"), index=False)

    try:
        import plotly.graph_objects as go

        def make_sankey(edges, value_col, out_html, title):
            # map nodes
            nodes = pd.Index(pd.unique(edges[["source_chain","destination_chain"]].values.ravel("K")))
            node_index = {n:i for i,n in enumerate(nodes)}
            sources = edges["source_chain"].map(node_index).tolist()
            targets = edges["destination_chain"].map(node_index).tolist()
            values  = edges[value_col].astype(float).tolist()

            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=10, thickness=12, label=nodes.tolist()),
                link=dict(source=sources, target=targets, value=values)
            )])
            fig.update_layout(title_text=title, font_size=10, height=600)
            fig.write_html(out_html)

        make_sankey(g_amt, "amount", os.path.join(outdir, "21_sankey_top_edges_amount.html"),
                   f"Top chain-pair flows by amount (filled), top {top_edges}")
        make_sankey(g_cnt, "count", os.path.join(outdir, "22_sankey_top_edges_count.html"),
                   f"Top chain-pair flows by count, top {top_edges}")

    except Exception as e:
        # plotly not installed or failed; tables are still produced
        with open(os.path.join(outdir, "sankey_note.txt"), "w", encoding="utf-8") as f:
            f.write("Plotly not available; sankey HTML not generated. Install plotly to enable.\n")
            f.write(str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="cleaned csv, e.g. final_bridge_data_layerzero_cleaned.csv")
    ap.add_argument("--outdir", default="./out_rerun")
    # --- 新增起始日期参数 ---
    ap.add_argument("--start_date", default="2022-01-01") 
    ap.add_argument("--cutoff", default="2025-10-31")
    ap.add_argument("--topk_bridge", type=int, default=10)
    ap.add_argument("--topk_chain", type=int, default=10)
    ap.add_argument("--flow_circle_nodes", type=int, default=15)
    ap.add_argument("--steps", default="all",
                    help="comma-separated: basic,bridge,heatmap,chains,flowcircle,sankey,layering,hhi,chainhhi,attack,all")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    steps = set([s.strip() for s in args.steps.split(",")]) if args.steps!="all" else set(["all"])

    df = pd.read_csv(args.input)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    # --- 核心逻辑修改：双向时间过滤 ---
    start_date = pd.to_datetime(args.start_date)
    cutoff = pd.to_datetime(args.cutoff)
    
    # 同时过滤起始时间和截止时间
    df = df[(df["date"] >= start_date) & (df["date"] <= cutoff)].copy()

    # 如果过滤后数据为空，打印警告
    if df.empty:
        print(f"警告: 在 {args.start_date} 到 {args.cutoff} 之间没有发现数据！")
        return

    # 记录原始数据用于绘图（后续逻辑保持不变）
    df["_raw_total_amount_usd"] = pd.to_numeric(df.get("total_amount_usd", np.nan), errors="coerce").fillna(0.0)

    df, (mb, imp, ma) = impute_total_amount_usd(df)
    df = add_time_buckets(df)

    # # basic
    if "all" in steps or "basic" in steps:
        plot_weekly_totals(df, args.outdir)

    # bridge
    if "all" in steps or "bridge" in steps:
        plot_bridge_timeseries(df, args.outdir, args.topk_bridge)

    # chains
    if "all" in steps or "chains" in steps:
        plot_chain_endpoint_activity(df, args.outdir, args.topk_chain)

    # # flow circle + summaries
    if "all" in steps or "flowcircle" in steps:
        compute_chain_flow_summary(df, os.path.join(args.outdir, "23_chainpair_flow_table_all.csv"))
        plot_flow_circle(df, os.path.join(args.outdir, "24_flow_circle_all.png"), args.flow_circle_nodes)

        df_no = official_removed(df)
        compute_chain_flow_summary(df_no, os.path.join(args.outdir, "25_chainpair_flow_table_no_official.csv"))
        plot_flow_circle(df_no, os.path.join(args.outdir, "26_flow_circle_no_official.png"), args.flow_circle_nodes)

    # sankey tables (+ optional html)
    if "all" in steps or "sankey" in steps:
        sankey_outputs(df, args.outdir, top_edges=40)

    # layering
    if "all" in steps or "layering" in steps:
        plot_layering(df, args.outdir, topk=8)

    # chain inflow/outflow hhi
    if "all" in steps or "chainhhi" in steps:
        chain_inflow_outflow_hhi(df, args.outdir)

    with open(os.path.join(args.outdir, "RUN_LOG.txt"), "w", encoding="utf-8") as f:
        f.write(f"Input: {args.input}\n")
        f.write(f"Start Date: {args.start_date}\n") # 日志中也记录起始时间
        f.write(f"Cutoff: {args.cutoff}\n")
        f.write(f"Imputation: total_amount_usd missing_before={mb}, imputable={imp}, missing_after={ma}\n")

    print("Done. Outputs in:", args.outdir)

if __name__ == "__main__":
    main()
