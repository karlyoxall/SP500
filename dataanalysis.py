#!/usr/bin/env python
import argparse
import re
from datetime import datetime, timedelta
from itertools import tee
from pathlib import Path
from typing import List, Tuple, Optional

import holidays
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tqdm import tqdm

from gp import GPTiny

# --- Configuration & Paths ---
DATA_DIR = Path("./data")
DAILIES_DIR = Path("./dailies")
START_DATE_DEFAULT = '2024-11-01'


def parse_args():
    parser = argparse.ArgumentParser(description="Stock analysis script with Genetic Programming models.")
    parser.add_argument(
        "top_n",
        type=int,
        nargs='?',
        default=50,
        help="The number of top/bottom stocks to filter (default: 250)"
    )
    return parser.parse_args()


def get_date_pairs(start: str, end: str) -> List[Tuple[datetime, datetime]]:
    us_holidays = holidays.financial_holidays('NYSE')
    daterange = pd.bdate_range(start=start, end=end)
    daterange = [c for c in daterange if c not in us_holidays]
    a1, a2 = tee(daterange)
    next(a2)
    return [(z[0].to_pydatetime(), z[1].to_pydatetime()) for z in zip(a1, a2)]


def get_snp500_list() -> pd.DataFrame:
    filepath = DATA_DIR / "sp500_companies.csv"
    if filepath.is_file():
        return pd.read_csv(filepath)

    headers = {'User-Agent': 'Mozilla/5.0'}
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find_all('table')[0]

    companies = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if cols:
            symbol = cols[0].text.strip().replace('.', '-')
            companies.append({"Symbol": symbol})

    df = pd.DataFrame(companies)
    df.to_csv(filepath, index=False)
    return df


def munge_data(sym: str, spy_data: Optional[pd.DataFrame], start: str, end: str):
    # noinspection PyBroadException
    try:
        if spy_data is None:
            spy_raw = yf.Ticker('^GSPC').history(start=start, end=end)
            spy_data = spy_raw[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change(fill_method=None) + 1

        stock_raw = yf.Ticker(sym).history(start=start, end=end)
        x = stock_raw[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change(fill_method=None) + 1

        x = pd.DataFrame(np.log(x / spy_data))
        x.insert(0, 'Days', (x.index - x.index[0]).days)

        for col in ['Open', 'Close', 'High', 'Low', 'Volume']:
            x[f'{col}_Div'] = x[col].diff() / x['Days'].diff()
            x[f'{col}_Div2'] = x[f'{col}_Div'].diff() / x['Days'].diff()
            for window in [7, 14, 21, 28]:
                x[f'{col}_{window}_MN'] = x[col].rolling(pd.Timedelta(days=window)).mean()
                x[f'{col}_{window}_SD'] = x[col].rolling(pd.Timedelta(days=window)).std()

        x.drop(columns=['Days'], inplace=True)
        x.insert(0, "Sym", sym)
        x['Target'] = x['Close'].shift(-1).fillna(-999)
        x.replace([np.inf, -np.inf], np.nan, inplace=True)
        return x.dropna(), spy_data
    except Exception:
        return None, spy_data


# noinspection SpellCheckingInspection
def main():
    args = parse_args()
    top_n = args.top_n
    print(f"Running analysis with top_n = {top_n}")

    DATA_DIR.mkdir(exist_ok=True)
    DAILIES_DIR.mkdir(exist_ok=True)

    # Determine last processed date
    existing_files = sorted(DAILIES_DIR.glob(f"top_close_{top_n}*.csv"), key=lambda f: f.stat().st_mtime)
    start_point = START_DATE_DEFAULT
    if existing_files:
        match = re.search(r'(\d{4}-\d{2}-\d{2})', existing_files[-1].name)
        if match: start_point = match.group(0)

    date_pairs = get_date_pairs(start=start_point, end=datetime.today().strftime("%Y-%m-%d"))
    if not date_pairs:
        print("Data is up to date.")
        return

    snp500 = get_snp500_list()

    # Global data pull
    global_start = (date_pairs[0][0] - timedelta(weeks=6)).strftime("%Y-%m-%d")
    global_end = date_pairs[-1][1].strftime("%Y-%m-%d")

    spy = None
    all_data = []
    for sym in tqdm(snp500.Symbol.values, desc="Downloading Tickers"):
        processed, spy = munge_data(sym, spy, global_start, global_end)
        if processed is not None:
            all_data.append(processed)

    full_df = pd.DataFrame(pd.concat(all_data, axis=0))
    gp_model = GPTiny()

    # Dynamic method collection (GPI, GPII, ..., GPX)
    gp_methods = [getattr(gp_model, name) for name in [
        'GPI', 'GPII', 'GPIII', 'GPIV', 'GPV', 'GPVI', 'GPVII', 'GPVIII', 'GPIX', 'GPX'
    ]]

    for start, end in tqdm(date_pairs, desc="Processing Windows"):
        s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        out_top = DAILIES_DIR / f"top_close_{top_n}_{s_str}_{e_str}.csv"
        out_bot = DAILIES_DIR / f"bottom_close_{top_n}_{s_str}_{e_str}.csv"

        if out_top.is_file(): continue

        day_slice = full_df[(full_df.index >= s_str) & (full_df.index < e_str)].copy()
        if day_slice.empty: continue

        w_sets, l_sets = [], []
        for method in gp_methods:
            t = day_slice[['Sym']].copy()
            t['Target'] = method(day_slice)
            pivot = t.reset_index().pivot(index='Date', columns='Sym', values='Target').iloc[0]

            w_sets.append(set(pivot.nlargest(top_n).index))
            l_sets.append(set(pivot.nsmallest(top_n).index))

        top_inter = set.intersection(*w_sets)
        bot_inter = set.intersection(*l_sets)

        def save_results(sym_list, path):
            results = {}
            for s in sym_list:
                s_data = day_slice[day_slice['Sym'] == s]
                scores = [m(s_data) for m in gp_methods]
                results[s] = [np.std(scores), np.min(scores), np.mean(scores), np.max(scores)]
            df = pd.DataFrame.from_dict(results, orient='index', columns=['Std', 'Mi', 'Mn', 'Ma'])
            df.index.name = 'Sym'
            df.to_csv(path)

        save_results(top_inter, out_top)
        save_results(bot_inter, out_bot)

    # Final Summary Output
    last_s, last_e = date_pairs[-1]
    ls_str, le_str = last_s.strftime("%Y-%m-%d"), last_e.strftime("%Y-%m-%d")
    print(f"\n--- Final Results ({ls_str} to {le_str}) ---")
    for label, prefix in [("Top", "top"), ("Bottom", "bottom")]:
        p = DAILIES_DIR / f"{prefix}_close_{top_n}_{ls_str}_{le_str}.csv"
        if p.exists():
            res = pd.read_csv(p)
            if not res.empty and len(res.Sym) > 0:
                symbols = ','.join(res.Sym.astype(str))
                print(f"{label}: https://uk.finance.yahoo.com/quote/{symbols}/")
            else:
                print(f"{label}: No Recommendations")

        else:
            print(f"{label}: No consensus recommendations.")


if __name__ == "__main__":
    main()