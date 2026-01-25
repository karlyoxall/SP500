#!/usr/bin/env python
import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import holidays
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from itertools import tee
from datetime import datetime, timedelta
from tqdm import tqdm

# --- Configuration ---
DATA_DIR = Path("./data")
DAILIES_DIR = Path("./dailies")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest GP model performance with local caching."
    )
    parser.add_argument("top_n", type=int, nargs="?", default=50)
    return parser.parse_args()


def get_date_pairs(start, end):
    us_holidays = holidays.financial_holidays("NYSE")
    daterange = pd.bdate_range(start=start, end=end)
    daterange = [c for c in daterange if c not in us_holidays]
    if len(daterange) < 2:
        return []
    a1, a2 = tee(daterange)
    next(a2)
    return [(z[0].to_pydatetime(), z[1].to_pydatetime()) for z in zip(a1, a2)]


def get_bulk_data(tickers, start, end):
    """Downloads all ticker data in one go to minimize API calls."""
    print(f"Downloading historical data for {len(tickers)} tickers...")
    # Download as a multi-index dataframe (Columns: Price Type -> Ticker)
    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=2),
        group_by="ticker",
        progress=True,
    )
    return data


def grab_performance_data(date_pairs, top_n, snp_df):
    model_type = "close"
    target = "Close"

    # 1. Identify all unique tickers needed for this backtest
    all_needed_syms = set(["^GSPC"])
    windows_to_process = []

    for start, end in date_pairs:
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        top_file = DAILIES_DIR / f"top_{model_type}_{top_n}_{s_str}_{e_str}.csv"
        bot_file = DAILIES_DIR / f"bottom_{model_type}_{top_n}_{s_str}_{e_str}.csv"

        if top_file.exists() and bot_file.exists():
            t_syms = pd.read_csv(top_file).Sym.tolist()
            b_syms = pd.read_csv(bot_file).Sym.tolist()
            all_needed_syms.update(t_syms)
            all_needed_syms.update(b_syms)
            windows_to_process.append((start, end, t_syms, b_syms))

    if not windows_to_process:
        return pd.DataFrame()

    # 2. Bulk Download
    global_start = windows_to_process[0][0]
    global_end = windows_to_process[-1][1]
    cache = get_bulk_data(list(all_needed_syms), global_start, global_end)

    # 3. Process Windows using Cache
    results = {
        "duration": [windows_to_process[0][0]],
        "allspy": [1.0],
        "alltoppercentages": [1.0],
        "allbottompercentages": [1.0],
        "alltopnumbers": [0.0],
        "allbottomnumbers": [0.0],
    }

    for start, end, t_syms, b_syms in tqdm(
        windows_to_process, desc="Calculating Returns"
    ):
        # Helper for vectorized return calculation
        def calc_mean_return(symbols):
            rets = []
            for s in symbols:
                try:
                    # Accessing multi-index dataframe: cache[Ticker][PriceType]
                    s_data = cache[s][target].loc[start : end + timedelta(days=1)]
                    change = s_data.pct_change().dropna() + 1
                    if not change.empty:
                        rets.append(change.values[0])
                except KeyError:
                    continue
            return np.mean(rets) if rets else 1.0

        spy_ret = calc_mean_return(["^GSPC"])
        top_ret = calc_mean_return(t_syms)
        bot_ret = calc_mean_return(b_syms)

        results["duration"].append(end)
        results["allspy"].append(spy_ret)
        results["alltoppercentages"].append(top_ret)
        results["allbottompercentages"].append(bot_ret)
        results["alltopnumbers"].append(len(t_syms))
        results["allbottomnumbers"].append(len(b_syms))

    return pd.DataFrame(results)


def plot_results(df, top_n):
    # (Plotting logic remains the same as previous version)
    df['duration'] = pd.to_datetime(df['duration'])

    # Reset to default then apply dark to ensure a clean state
    plt.rcParams.update(plt.rcParamsDefault)
    plt.style.use('dark_background')

    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        sharex=True,
        figsize=(15, 10),
        gridspec_kw={'height_ratios': [3, 1]},
        facecolor='black'
    )

    # 1. Cumulative Performance
    ax1.plot(df.duration, np.cumprod(df.allbottompercentages), label='Bottom Portfolio', color='#FF3131', alpha=0.8)
    ax1.plot(df.duration, np.cumprod(df.alltoppercentages), label='Top Portfolio', color='#39FF14', linewidth=2)
    ax1.plot(df.duration, np.cumprod(df.allspy), label='S&P 500 Index', color='#FFFFFF', linestyle='--', alpha=0.7)

    ax1.set_title(f"Cumulative Performance (top_n={top_n})", fontsize=16, pad=20, color='white')
    ax1.set_ylabel("Growth of $1", color='white')

    # Legend with visible border
    ax1.legend(loc='upper left', frameon=True, facecolor='#222222', edgecolor='white', labelcolor='white')
    ax1.grid(True, which='major', linestyle=':', alpha=0.3, color='grey')

    # 2. Consensus Ticker Count
    ax2.bar(df.duration, df.alltopnumbers, color='#00D4FF', width=0.8, alpha=0.8)
    ax2.set_ylabel("No. of Tickers", color='white')
    ax2.set_xlabel("Date", color='white')
    ax2.grid(True, axis='y', linestyle=':', alpha=0.2, color='grey')

    # Force tick colors to white
    ax1.tick_params(colors='white', which='both')
    ax2.tick_params(colors='white', which='both')

    # --- Fix X-Axis Ticks ---
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.offset_formats[3] = '%Y'  # Ensure year is visible in concise format

    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)

    # Rotate dates so they aren't cut off
    fig.autofmt_xdate()

    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.savefig(str(top_n) + ".png")


def main():
    args = parse_args()
    summary_file = DAILIES_DIR / f"latest_{args.top_n}.csv"

    latest_df = pd.read_csv(summary_file) if summary_file.is_file() else None
    start_date = (
        latest_df["duration"].iloc[-1] if latest_df is not None else "2024-11-01"
    )
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    date_pairs = get_date_pairs(start_date, datetime.today() - timedelta(days=1))
    if not date_pairs:
        if latest_df is not None:
            plot_results(latest_df, args.top_n)
        return

    new_data = grab_performance_data(date_pairs, args.top_n, None)

    if latest_df is not None:
        latest_df["duration"] = pd.to_datetime(latest_df["duration"])
        new_data["duration"] = pd.to_datetime(new_data["duration"])
        final_df = (
            pd.concat([latest_df, new_data])
            .drop_duplicates(subset=["duration"])
            .reset_index(drop=True)
        )
    else:
        final_df = new_data

    final_df.to_csv(summary_file, index=False)
    plot_results(final_df, args.top_n)


if __name__ == "__main__":
    main()

