#!/usr/bin/env python

import os
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from itertools import tee
from typing import Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import holidays
import yfinance as yf
from tqdm import tqdm

from gp import GPTiny


def get_date_pairs(start: str, end: str) -> list[tuple[datetime, datetime]]:
    """Generate pairs of consecutive business dates excluding NYSE holidays."""
    us_holidays = holidays.financial_holidays("NYSE")
    daterange = pd.bdate_range(start=start, end=end)
    daterange = [date for date in daterange if date not in us_holidays]

    iter1, iter2 = tee(daterange)
    next(iter2)
    pairs = [
        (date1.to_pydatetime(), date2.to_pydatetime())
        for date1, date2 in zip(iter1, iter2)
    ]
    return pairs


def get_latest_date_from_files(top_n: int) -> Optional[str]:
    """Extract the latest date from existing daily files."""
    existing_files = sorted(
        Path("./dailies").iterdir(), key=lambda f: f.stat().st_mtime
    )
    latest_files = [str(f) for f in existing_files if f"top_close_{top_n}_" in str(f)]

    if not latest_files:
        return None

    match = re.search(r"(\d+-\d+-\d+)", latest_files[-1])
    return match.group(0) if match else None


def fetch_sp500_companies(filepath: Path) -> pd.DataFrame:
    """Fetch S&P 500 companies from Wikipedia and cache to CSV."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36"
    }

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers,
        timeout=30,
    )
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")

    companies_data = []
    for row in tables[0].tbody.find_all("tr"):
        cols = row.find_all("td")
        if not cols:
            continue

        companies_data.append(
            {
                "Symbol": cols[0].text.strip().replace("\n", ""),
                "Security": cols[1].text.strip().replace("\n", ""),
                "Sector": cols[2].text.strip().replace("\n", ""),
                "Sub-Industry": cols[3].text.strip().replace("\n", ""),
                "Headquarters": cols[4].text.strip().replace("\n", ""),
                "Date-Added": cols[5].text.strip().replace("\n", ""),
                "CIK": cols[6].text.strip().replace("\n", ""),
                "Founded": cols[7].text.strip().replace("\n", ""),
            }
        )

    df = pd.DataFrame(data=companies_data)
    df["Symbol"] = df["Symbol"].str.replace(".", "-")
    df.to_csv(filepath, index=False)
    return df


def load_sp500_companies() -> pd.DataFrame:
    """Load S&P 500 companies from cache or fetch if not available."""
    filepath = Path("./data/sp500_companies.csv")

    if filepath.is_file():
        return pd.read_csv(filepath)

    return fetch_sp500_companies(filepath)


def munge_data(
    symbol: str, spy_data: Optional[pd.DataFrame], start: str, end: str
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Process stock data with technical indicators and normalization."""
    try:
        # Fetch SPY data if not provided
        if spy_data is None:
            spy_ticker = yf.Ticker("^GSPC")
            spy_hist = spy_ticker.history(start=start, end=end)
            spy_data = (
                spy_hist[["Open", "High", "Low", "Close", "Volume"]]
                .copy()
                .pct_change(fill_method=None)
                + 1
            )

        # Fetch stock data
        stock = yf.Ticker(symbol)
        stock_hist = stock.history(start=start, end=end)
        data = (
            stock_hist[["Open", "High", "Low", "Close", "Volume"]]
            .copy()
            .pct_change(fill_method=None)
            + 1
        )

        # Normalize by SPY and log transform
        data /= spy_data
        np.seterr(divide="ignore")
        data = np.log(data)
        np.seterr(divide="warn")
        # Add day count
        data.insert(0, "Days", (data.index - data.index[0]).days)

        # Calculate derivatives and rolling statistics
        columns = ["Open", "Close", "High", "Low", "Volume"]
        for col in columns:
            data[f"{col}_Div"] = data[col].diff() / data["Days"].diff()
            data[f"{col}_Div2"] = data[f"{col}_Div"].diff() / data["Days"].diff()

            for window in [7, 14, 21, 28]:
                data[f"{col}_{window}_MN"] = (
                    data[col].rolling(pd.Timedelta(days=window)).mean()
                )
                data[f"{col}_{window}_SD"] = (
                    data[col].rolling(pd.Timedelta(days=window)).std()
                )

        # Clean up and prepare final dataframe
        data.pop("Days")
        data.insert(0, "Sym", symbol)
        data["Target"] = data.groupby("Sym")["Close"].shift(-1).fillna(-999)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        return data, spy_data

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return None, spy_data


def calculate_gp_predictions(gp_model: GPTiny, data: pd.DataFrame) -> list[pd.Series]:
    """Calculate predictions using all GP models."""
    gp_methods = [
        gp_model.GPI,
        gp_model.GPII,
        gp_model.GPIII,
        gp_model.GPIV,
        gp_model.GPV,
        gp_model.GPVI,
        gp_model.GPVII,
        gp_model.GPVIII,
        gp_model.GPIX,
        gp_model.GPX,
    ]
    return [method(data) for method in gp_methods]


def process_daily_predictions(
    data: pd.DataFrame, start: datetime, end: datetime, gp_model: GPTiny, top_n: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process daily predictions and identify top/bottom performers."""
    filepath = Path(
        f"./dailies/top_close_{top_n}_{start.strftime('%Y-%m-%d')}_"
        f"{end.strftime('%Y-%m-%d')}.csv"
    )

    if filepath.is_file():
        return pd.DataFrame(), pd.DataFrame()

    # print(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    mask = (data.index >= start.strftime("%Y-%m-%d")) & (
        data.index < end.strftime("%Y-%m-%d")
    )
    daily_data = data[mask].copy()

    if daily_data.shape[0] == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Collect predictions from all GP models
    winners_set = []
    losers_set = []

    for predictions in calculate_gp_predictions(gp_model, daily_data):
        temp = daily_data[["Sym"]].copy()
        temp["Target"] = predictions

        pivoted = temp.reset_index().pivot(index="Date", columns="Sym", values="Target")

        winners = pd.DataFrame(
            {
                n: pivoted.T[col].nlargest(top_n).index.tolist()
                for n, col in enumerate(pivoted.T)
            }
        ).T

        losers = pd.DataFrame(
            {
                n: pivoted.T[col].nsmallest(top_n).index.tolist()
                for n, col in enumerate(pivoted.T)
            }
        ).T

        winners_set.append(winners.values[0])
        losers_set.append(losers.values[0])

    # Find intersection of top/bottom performers
    winners_subset = pd.DataFrame(
        data={"Sym": list(set.intersection(*map(set, winners_set)))}
    )
    losers_subset = pd.DataFrame(
        data={"Sym": list(set.intersection(*map(set, losers_set)))}
    )

    # Calculate statistics for each symbol
    winners_stats = calculate_symbol_statistics(
        winners_subset, data, start, end, gp_model
    )
    losers_stats = calculate_symbol_statistics(
        losers_subset, data, start, end, gp_model
    )

    # Save results
    save_predictions(winners_stats, start, end, "top", top_n)
    save_predictions(losers_stats, start, end, "bottom", top_n)

    return winners_subset, losers_subset


def calculate_symbol_statistics(
    symbols_df: pd.DataFrame,
    data: pd.DataFrame,
    start: datetime,
    end: datetime,
    gp_model: GPTiny,
) -> dict[str, list[float]]:
    """Calculate prediction statistics for each symbol."""
    stats = {}

    for symbol in symbols_df["Sym"].values:
        mask = (data.index >= start.strftime("%Y-%m-%d")) & (
            data.index < end.strftime("%Y-%m-%d")
        )
        symbol_data = data[mask].copy()
        symbol_data = symbol_data[symbol_data["Sym"] == symbol]

        predictions = [
            method(symbol_data)
            for method in [
                gp_model.GPI,
                gp_model.GPII,
                gp_model.GPIII,
                gp_model.GPIV,
                gp_model.GPV,
                gp_model.GPVI,
                gp_model.GPVII,
                gp_model.GPVIII,
                gp_model.GPIX,
                gp_model.GPX,
            ]
        ]

        stats[symbol] = [
            np.std(predictions),
            np.min(predictions),
            np.mean(predictions),
            np.max(predictions),
        ]

    return stats


def save_predictions(
    stats: dict[str, list[float]],
    start: datetime,
    end: datetime,
    prediction_type: str,
    top_n: int,
) -> None:
    """Save prediction statistics to CSV."""
    df = pd.DataFrame.from_dict(
        stats, orient="index", columns=["Std", "Mi", "Mn", "Ma"]
    )
    df.index.name = "Sym"

    filename = (
        f"./dailies/{prediction_type}_close_{top_n}_"
        f"{start.strftime('%Y-%m-%d')}_{end.strftime('%Y-%m-%d')}.csv"
    )
    df.to_csv(filename)


def print_recommendations(start: datetime, end: datetime, top_n: int) -> None:
    """Print stock recommendations from the latest predictions."""
    bottom = pd.read_csv(
        f"./dailies/bottom_close_{top_n}_{start.strftime('%Y-%m-%d')}_"
        f"{end.strftime('%Y-%m-%d')}.csv"
    )
    top = pd.read_csv(
        f"./dailies/top_close_{top_n}_{start.strftime('%Y-%m-%d')}_"
        f"{end.strftime('%Y-%m-%d')}.csv"
    )

    print("Top:")
    if len(top["Sym"]):
        symbols = ",".join(top["Sym"].values)
        print(f"https://uk.finance.yahoo.com/quote/{symbols}/")
    else:
        print("No Recommendations")

    print("Bottom:")
    if len(bottom["Sym"]):
        symbols = ",".join(bottom["Sym"].values)
        print(f"https://uk.finance.yahoo.com/quote/{symbols}/")
    else:
        print("No Recommendations")


def main() -> None:
    """Main execution function."""
    # Parse command line argument
    if len(sys.argv) != 2:
        print("Usage: python script.py <top_n>")
        print("Example: python script.py 200")
        sys.exit(1)

    try:
        top_n = int(sys.argv[1])
        if top_n <= 0:
            raise ValueError("top_n must be a positive integer")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please provide a valid positive integer")
        sys.exit(1)

    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./dailies", exist_ok=True)

    # Determine date range
    latest_date = get_latest_date_from_files(top_n)
    today = datetime.today().strftime("%Y-%m-%d")

    if latest_date is None:
        date_pairs = get_date_pairs(start="2024-11-01", end=today)
    else:
        date_pairs = get_date_pairs(start=latest_date, end=today)

    print(f"Start: {date_pairs[0][0]}, End: {date_pairs[-1][1]}")

    # Load S&P 500 companies
    sp500 = load_sp500_companies()
    sp500.head()

    # Process stock data
    start_date = date_pairs[0][0].strftime("%Y-%m-%d")
    end_date = date_pairs[-1][1].strftime("%Y-%m-%d")

    spy_data = None
    processed_data = []

    # Add 5 weeks buffer for rolling calculations
    buffer_start = (
        datetime.strptime(start_date, "%Y-%m-%d") - timedelta(weeks=5)
    ).strftime("%Y-%m-%d")

    for symbol in tqdm(list(sp500["Symbol"].values)):
        data, spy_data = munge_data(symbol, spy_data, buffer_start, end_date)
        if data is not None:
            processed_data.append(data)

    combined_data = pd.concat(processed_data, axis=0)

    # Generate predictions
    gp_model = GPTiny()
    for start, end in tqdm(date_pairs):
        process_daily_predictions(combined_data, start, end, gp_model, top_n)

    # Print recommendations
    final_start, final_end = date_pairs[-1]
    print(final_start, final_end)
    print_recommendations(final_start, final_end, top_n)


if __name__ == "__main__":
    main()
