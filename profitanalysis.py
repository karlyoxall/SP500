import sys
import time
import argparse
from itertools import tee
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import yfinance as yf
import matplotlib.pyplot as mp
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from numpy.typing import NDArray


def GetDatePairs(start: str, end: str) -> List[Tuple[datetime, datetime]]:
    us_holidays = holidays.financial_holidays("NYSE")
    daterange = pd.bdate_range(start=start, end=end)
    daterange_filtered = [c for c in daterange if c not in us_holidays]
    a1, a2 = tee(daterange_filtered)
    next(a2)
    pairs = list((z[0].to_pydatetime(), z[1].to_pydatetime()) for z in zip(a1, a2))
    return pairs


def GrabData(B: List[Tuple[datetime, datetime]], top_n: int, throttle: float = 0.2) -> Tuple[
    Optional[List[datetime]],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    modeltype = "close"
    target = "Close"
    allspy: List[float] = [1.0]
    alltoppercentages: List[float] = [1.0]
    alltopnumbers: List[float] = [0.0]
    allbottompercentages: List[float] = [1.0]
    allbottomnumbers: List[float] = [0.0]

    duration: Optional[List[datetime]] = None

    for start, end in B:
        spy: List[float] = []
        toppercentages: List[float] = []
        bottompercentages: List[float] = []
        topfileformat = f"./dailies/top_{{0}}_{top_n}_{{1}}_{{2}}.csv"
        bottomfileformat = f"./dailies/bottom_{{0}}_{top_n}_{{1}}_{{2}}.csv"
        topsyms_df = pd.read_csv(
            topfileformat.format(
                modeltype, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
        )
        topnumbers: float = float(topsyms_df.shape[0])
        topsyms: NDArray[np.str_] = topsyms_df.Sym.values

        stock = yf.Ticker("^GSPC")
        stockhist = stock.history(start=start, end=end + timedelta(days=1))
        x = stockhist[[target]].copy().pct_change(fill_method=None) + 1
        x = x.dropna()

        if x.shape[0] == 0:
            break

        if duration is None:
            duration = [start]
        duration.append(end)
        spy.append(float(x[target].values[0]))

        for sym in topsyms:
            stock = yf.Ticker(str(sym))
            stockhist = stock.history(start=start, end=end + timedelta(days=1))
            time.sleep(throttle)
            x = stockhist[[target]].copy().pct_change(fill_method=None) + 1
            x = x.dropna()
            if x.shape[0] > 0:
                toppercentages.append(float(x[target].values[0]))
            else:
                print(start, end, sym)

        bottomsyms_df = pd.read_csv(
            bottomfileformat.format(
                modeltype, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
            )
        )
        bottomnumbers: float = float(bottomsyms_df.shape[0])
        bottomsyms: NDArray[np.str_] = bottomsyms_df.Sym.values

        for sym in bottomsyms:
            stock = yf.Ticker(str(sym))
            stockhist = stock.history(start=start, end=end + timedelta(days=1))
            time.sleep(throttle)
            x = stockhist[[target]].copy().pct_change(fill_method=None) + 1
            x = x.dropna()
            if x.shape[0] > 0:
                bottompercentages.append(float(x[target].values[0]))
            else:
                print(start, end, sym)

        allspy.append(float(np.mean(spy)) if len(spy) > 0 else 1.0)

        if len(toppercentages) == 0:
            toppercentages = [1.0]
        if np.mean(toppercentages) < 0.95:
            print(start, end, topsyms, toppercentages)
        alltoppercentages.append(float(np.mean(toppercentages)))

        if len(bottompercentages) == 0:
            bottompercentages = [1.0]

        allbottompercentages.append(float(np.mean(bottompercentages)))
        alltopnumbers.append(topnumbers)
        allbottomnumbers.append(bottomnumbers)

    return (
        duration,
        allspy,
        alltoppercentages,
        allbottompercentages,
        alltopnumbers,
        allbottomnumbers,
    )


def main() -> None:
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

    latest: Optional[pd.DataFrame] = None
    B: List[Tuple[datetime, datetime]]
    x: Optional[pd.DataFrame] = None
    snp500 = pd.read_csv("./data/sp500_companies.csv")
    filepath = Path(f"./dailies/latest_{top_n}.csv")

    if filepath.is_file():
        latest = pd.read_csv(f"./dailies/latest_{top_n}.csv")
        latest_duration_str: str = str(latest.duration.values[-1])
        B = GetDatePairs(
            start=datetime.strptime(latest_duration_str, "%Y-%m-%d").strftime(
                "%Y-%m-%d"
            ),
            end=(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        )
        latest.duration = pd.to_datetime(latest.duration)
    else:
        B = GetDatePairs(
            start=datetime.strptime("2024-11-01", "%Y-%m-%d").strftime("%Y-%m-%d"),
            end=(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"),
        )

    if len(B) > 0:
        print(f"Start: {B[0][0]}, End:{B[-1][1]}")
        (
            duration,
            allspy,
            alltoppercentages,
            allbottompercentages,
            alltopnumbers,
            allbottomnumbers,
        ) = GrabData(B, top_n)

        if duration is not None:
            x = pd.DataFrame(
                data={
                    "duration": duration,
                    "allspy": allspy,
                    "alltoppercentages": alltoppercentages,
                    "allbottompercentages": allbottompercentages,
                    "alltopnumbers": alltopnumbers,
                    "allbottomnumbers": allbottomnumbers,
                }
            )

            if len(x) > 1:
                if latest is not None and len(latest) > 0:
                    x = pd.concat(
                        [
                            latest,
                            x.loc[x.duration.values > latest.duration.values[-1], :],
                        ]
                    ).reset_index(drop=True)
                    x = x.drop_duplicates(ignore_index=True)
                x.to_csv(f"./dailies/latest_{top_n}.csv", index=False)

                fig, (ax1, ax2) = mp.subplots(nrows=2, sharex=True, figsize=(15, 10))
                ax1.plot(x.duration, np.cumprod(x.allbottompercentages), label="worst")
                ax1.plot(x.duration, np.cumprod(x.alltoppercentages), label="best")
                ax1.plot(x.duration, np.cumprod(x.allspy), label="s&p")

                date_form = DateFormatter("%m-%y")
                ax1.xaxis.set_major_formatter(date_form)
                ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax2.bar(x.duration[1:], x.alltopnumbers[1:])
                ax2.xaxis.set_major_formatter(date_form)
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

                ax1.set(xlabel=None, ylabel=None, title="Cumulative Returns")
                ax2.set(xlabel=None, ylabel=None, title="No. of Tickers")

                ax1.legend()

                mp.savefig(str(top_n)+".png")


if __name__ == "__main__":
    main()