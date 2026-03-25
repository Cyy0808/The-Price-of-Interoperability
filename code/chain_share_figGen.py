import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1. 全局配置与字体放大 ---
plt.rcParams.update({'font.size': 14}) # 基础字号

DATA_PATH = "bridge_data_final.csv"
CUTOFF = "2025-10-31"
TOP_N = 10

OUT_COUNT = "chain_share_weekly_count_top10_large_8.png"
OUT_AMOUNT = "chain_share_weekly_amount_top10_large_8.png"

# --- 2. 数据加载与预处理 ---
df = pd.read_csv(DATA_PATH)

def pick_col(candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}")

date_col = pick_col(["date", "day", "timestamp", "time"])
dst_col  = pick_col(["destination_chain", "dst_chain", "to_chain", "destination", "endpoint_chain", "endpoint"])
count_col = pick_col(["transfer_count", "count", "tx_count", "n_transfers"])
amount_col = pick_col(["total_amount_usd", "amount_usd", "total_usd"])
avg_col = pick_col(["avg_transfer_usd_value", "avg_usd", "avg_amount_usd", "avg_transfer_value_usd"])

df[date_col] = pd.to_datetime(df[date_col])
df = df[df[date_col] <= pd.Timestamp(CUTOFF)].copy()

# 填充缺失金额
df["transfer_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0.0)
raw_amount = pd.to_numeric(df[amount_col], errors="coerce")
avg_val = pd.to_numeric(df[avg_col], errors="coerce")
df["filled_total_amount_usd"] = raw_amount.where(raw_amount.notna(), avg_val * df["transfer_count"]).fillna(0.0)

# --- 3. 计算逻辑 (解决 NameError 的关键) ---
def weekly_pivot(value_col: str) -> pd.DataFrame:
    tmp = df[[date_col, dst_col, value_col]].copy()
    tmp = tmp.rename(columns={date_col: "date", dst_col: "endpoint", value_col: "value"})
    weekly = (
        tmp.groupby(["endpoint", pd.Grouper(key="date", freq="W")])["value"]
           .sum()
           .unstack("endpoint")
           .fillna(0.0)
           .sort_index()
    )
    return weekly

def shares_top_with_others(weekly: pd.DataFrame, top_cols: list[str]):
    total = weekly.sum(axis=1)
    shares = weekly.div(total.replace(0, np.nan), axis=0).fillna(0.0)
    top = shares.reindex(columns=top_cols, fill_value=0.0)
    others = (1.0 - top.sum(axis=1)).clip(lower=0.0)
    return top, others

# 获取周数据
weekly_count = weekly_pivot("transfer_count")
weekly_amount = weekly_pivot("filled_total_amount_usd")

# 定义 Top 10 链
top_totals = weekly_count.sum(axis=0).sort_values(ascending=False)
top_chains = top_totals.head(TOP_N).index.tolist()

# 【重要：定义报错中缺失的变量】
top_share_count, others_count = shares_top_with_others(weekly_count, top_chains)
top_share_amount, others_amount = shares_top_with_others(weekly_amount, top_chains)

# --- 4. 绘图函数 (大字号 + 年度坐标轴 + 灰色 Other) ---
def plot_stack_side_legend(top_share: pd.DataFrame, others: pd.Series, title: str, outpath: str):
    fig, ax = plt.subplots(figsize=(16, 5.5))
    
    plot_df = top_share.copy()
    plot_df["Other"] = others
    x = plot_df.index
    columns = plot_df.columns
    

    manual_colors = {
        "arbitrum": "#49c2d9", "base": "#67a583", "optimism": "#a2c986",
        "polygon": "#7b95c6", "avalanche": "#f47254", "bsc": "#d0e2c0",
        "ethereum": "#f59c7c", "aptos": "#c85e62", "linea": "#ffcca6",
        "hyperliquid": "#a1d8e8", 
        "other": "#fded95" # 灰色
    }

    # manual_colors = {
    #     "arbitrum": "#49c2d9", "base": "#67a583", "optimism": "#a2c986",
    #     "polygon": "#7b95c6", "avalanche": "#c85e62", "bsc": "#fded95",
    #     "ethereum": "#f47254", "aptos": "#f59c7c", "linea": "#d0e2c0",
    #     "hyperliquid": "#a1d8e8", 
    #     "other": "#ffcca6" # 灰色
    # }

    tab10_colors = plt.get_cmap("tab10").colors
    colors = [manual_colors.get(c.lower(), tab10_colors[i % 10]) for i, c in enumerate(columns)]

    ax.stackplot(x, plot_df.values.T, labels=columns, colors=colors, linewidth=0.0, alpha=1)

    # 字体放大设置
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Share", fontsize=16, fontweight='bold')
    ax.set_title(title, pad=25, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # 年度坐标轴
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=0, ha='center') 
    
    ax.grid(axis="y", alpha=0.15, linestyle='--', color='gray')

    # --- 修改后的图例代码 ---
    # 获取当前绘图的所有句柄和标签
    handles, labels = ax.get_legend_handles_labels()

    # 使用 [::-1] 将它们反转，这样图例顶部的名称就会对应图中顶部的色块
    ax.legend(
        handles[::-1], 
        labels[::-1], 
        loc="center left", 
        bbox_to_anchor=(1.02, 0.5), 
        frameon=False, 
        fontsize=14
    )
    # 图例放大
    # ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath, dpi=260, bbox_inches="tight")
    print(f"成功保存: {outpath}")
    plt.close(fig)

# --- 5. 执行绘图 ---
plot_stack_side_legend(
    top_share_count, others_count,
    "Top 10 endpoint chains — weekly share of transfer_count",
    OUT_COUNT
)
plot_stack_side_legend(
    top_share_amount, others_amount,
    "Top 10 endpoint chains — weekly share of total_amount_usd",
    OUT_AMOUNT
)