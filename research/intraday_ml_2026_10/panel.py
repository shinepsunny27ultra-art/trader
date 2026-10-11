"""Point-in-time hourly cross-sectional panel for the intraday ML ranker study.

Bars are Yahoo 1h bars, ts = bar START (IST): b0 09:15, b1 10:15, ..., b5 14:15, b6 15:15
(partial, 15:15-15:30).  A decision "after bar b" is taken at the close of bar b, i.e. at
wall-clock 10:15 (b=0), 11:15 (b=1), 12:15 (b=2), 13:15 (b=3), 14:15 (b=4).
Features use bars 0..b only plus daily context from sessions strictly before today.
Entry = open of bar b+1 (the next print after the decision, not the feature close).
Exit  = close of bar 5 (the price at 15:15, before MIS square-off) or close of bar b+1 (1h).

Yahoo's 09:15 hourly volume is 0 on ~85% of rows, so volume features use bars 1..b only.

Run: PYTHONPATH=src .venv/bin/python research/intraday_ml_2026_10/panel.py   (builds cache)
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nse_intraday_ai import nse_bhav

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "candles.sqlite3"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
PANEL = CACHE / "panel.parquet"
COST_BPS = 13.3
NB = 7            # hourly bars per session incl. the partial 15:15 bar
EXIT_BAR = 5      # close of the 14:15 bar = 15:15 price
DECISIONS = (0, 1, 2, 3, 4)
STOPS = (0.5, 1.0)  # ATR multiples

FEATURES = [
    "b", "r_open", "gap", "r_pc", "r_last1", "r_last2", "range_pos", "range_atr",
    "dist_twap", "dist_vwap", "relvol_cum", "relvol_last",
    "r_open_atr", "gap_atr", "r_pc_atr", "r_last1_atr", "atr_pct",
    "mkt_r_open", "mkt_r_pc", "mkt_r_last1", "breadth", "disp",
    "rel_r_open", "sec_r_open", "secrel_r_open",
    "rk_r_open", "rk_r_pc", "rk_r_last1", "rk_relvol", "rk_gap",
    "prev_ret", "ret5", "ret20", "prev_deliv", "prev_pos", "turn_rank", "prev_turn_ratio",
]


def load_bars():
    uni = pd.read_csv(ROOT / "data/nifty500_symbols.csv")
    uni["Symbol"] = uni["Symbol"].astype(str).str.strip()
    syms = sorted(uni["Symbol"] + ".NS")
    con = sqlite3.connect(DB)
    ph = ",".join("?" * len(syms))
    df = pd.read_sql(f"select symbol, ts, open, high, low, close, volume from candles "
                     f"where interval='1h' and symbol in ({ph})", con, params=syms)
    con.close()
    ts = pd.to_datetime(df.ts, utc=True).dt.tz_convert("Asia/Kolkata")
    df["day"] = ts.dt.date.astype(str)
    df["bar"] = ts.dt.hour - 9
    df = df[(df.bar >= 0) & (df.bar < NB) & (ts.dt.minute == 15)]
    today = pd.Timestamp.now(tz="Asia/Kolkata").date().isoformat()
    df = df[df.day < today]                       # never use the live session
    days_all = sorted(df.day.unique())
    symbols = sorted(df.symbol.unique())
    di = {d: i for i, d in enumerate(days_all)}
    si = {s: i for i, s in enumerate(symbols)}
    arr = {k: np.full((len(days_all), len(symbols), NB), np.nan) for k in ("open", "high", "low", "close", "volume")}
    for k in arr:
        arr[k][df.day.map(di).to_numpy(), df.symbol.map(si).to_numpy(), df.bar.to_numpy()] = df[k].to_numpy()
    sector = uni.set_index(uni["Symbol"] + ".NS")["Industry"].reindex(symbols).fillna("NA").to_numpy()
    return days_all, symbols, arr, sector


def daily_context(days, symbols):
    """Prior-session daily features from the official bhavcopy (split-robust: uses prev_close)."""
    start = date.fromisoformat(days[0]) - pd.Timedelta(days=60)
    bh = nse_bhav.load(start, date.fromisoformat(days[-1]),
                       columns=["symbol", "series", "high", "low", "close", "prev_close", "value", "deliv_pct", "open"])
    bh["symbol"] = bh["symbol"].astype(str) + ".NS"
    bh = bh[bh.symbol.isin(symbols)].drop_duplicates(["session", "symbol"])
    bh["session"] = pd.to_datetime(bh["session"])
    w = {c: bh.pivot(index="session", columns="symbol", values=c).reindex(columns=symbols).sort_index()
         for c in ("high", "low", "close", "prev_close", "value", "deliv_pct", "open")}
    pc = w["prev_close"]
    ret = w["close"] / pc - 1
    tr = pd.concat([(w["high"] - w["low"]), (w["high"] - pc).abs(), (w["low"] - pc).abs()]).groupby(level=0).max() / pc
    lr = np.log1p(ret)
    turn20 = w["value"].rolling(20, min_periods=10).mean()
    out = {
        "prev_ret": ret,
        "ret5": lr.rolling(5, min_periods=4).sum(),
        "ret20": lr.rolling(20, min_periods=15).sum(),
        "atr_pct": tr.rolling(14, min_periods=10).mean(),
        "prev_deliv": w["deliv_pct"],
        "prev_pos": (w["close"] - w["low"]) / (w["high"] - w["low"]),
        "turn20": turn20,
        "prev_turn_ratio": w["value"] / turn20.shift(1),
    }
    out["turn_rank"] = turn20.rank(axis=1, ascending=False)
    idx = pd.to_datetime(pd.Index(days))
    # shift(1) on the bhav calendar: the value known before session t is the one from session t-1.
    ctx = {k: v.shift(1).reindex(idx).to_numpy(float) for k, v in out.items()}
    # data check only (never a feature): today's official gap
    ctx["_bhav_gap"] = (w["open"] / pc - 1).reindex(idx).to_numpy(float)
    ctx["_bhav_cov"] = float(np.isin(idx, w["close"].index).mean())
    return ctx


def _xs_rank(x):
    return pd.DataFrame(x).rank(axis=1, pct=True).to_numpy()


def _sector_median(x, sector):
    df = pd.DataFrame(x.T)
    df["sec"] = sector
    med = df.groupby("sec").transform("median")
    return med.to_numpy().T


def _first_valid_close(c, lo, hi):
    """Last non-NaN close among bars lo..hi (inclusive), per (d, s)."""
    out = np.full(c.shape[:2], np.nan)
    for j in range(lo, hi + 1):
        v = c[:, :, j]
        out = np.where(np.isnan(v), out, v)
    return out


def _stop_sim(arr, b, side, k, atr_abs, entry, exit_bar):
    """Vectorised stop-first simulation from entry bar e=b+1 to exit_bar (inclusive)."""
    o, h, low, c = arr["open"], arr["high"], arr["low"], arr["close"]
    e = b + 1
    stop = entry - side * k * atr_abs
    res = np.full(entry.shape, np.nan)
    done = np.zeros(entry.shape, bool)
    for j in range(e, exit_bar + 1):
        hit = (low[:, :, j] <= stop) if side > 0 else (h[:, :, j] >= stop)
        hit &= ~done & ~np.isnan(h[:, :, j])
        if j == e:
            fill = stop
        else:
            oj = np.where(np.isnan(o[:, :, j]), stop, o[:, :, j])
            fill = np.minimum(stop, oj) if side > 0 else np.maximum(stop, oj)
        res = np.where(hit, side * (fill - entry) / entry * 1e4, res)
        done |= hit
    px = _first_valid_close(c, e, exit_bar)
    res = np.where(done, res, side * (px - entry) / entry * 1e4)
    return res, done


def build():
    days, symbols, arr, sector = load_bars()
    o, h, low, c, v = (arr[k] for k in ("open", "high", "low", "close", "volume"))
    # previous session close from the hourly series itself (same split adjustment as today's bars)
    last_close = _first_valid_close(c, 0, NB - 1)
    pclose = np.vstack([np.full((1, len(symbols)), np.nan), last_close[:-1]])
    ctx = daily_context(days, symbols)
    nsym = np.sum(~np.isnan(c[:, :, 1]), axis=1)
    good_day = nsym >= 300
    frames = []
    for b in DECISIONS:
        cb = c[:, :, b]
        o0 = o[:, :, 0]
        hi = np.nanmax(h[:, :, : b + 1], axis=2)
        lo = np.nanmin(low[:, :, : b + 1], axis=2)
        atr_pct = ctx["atr_pct"]
        f = {"b": np.full(cb.shape, b, float)}
        f["r_open"] = cb / o0 - 1
        f["gap"] = o0 / pclose - 1
        f["r_pc"] = cb / pclose - 1
        f["r_last1"] = cb / (c[:, :, b - 1] if b >= 1 else o0) - 1
        f["r_last2"] = cb / (c[:, :, b - 2] if b >= 2 else o0) - 1
        f["range_pos"] = (cb - lo) / (hi - lo)
        f["range_atr"] = (hi - lo) / pclose / atr_pct
        tp = (h[:, :, : b + 1] + low[:, :, : b + 1] + c[:, :, : b + 1]) / 3
        f["dist_twap"] = cb / np.nanmean(tp, axis=2) - 1
        if b >= 1:
            vv = v[:, :, 1: b + 1]
            vsum = np.nansum(vv, axis=2)
            f["dist_vwap"] = cb / (np.nansum(tp[:, :, 1:] * vv, axis=2) / np.where(vsum > 0, vsum, np.nan)) - 1
            cum = np.where(vsum > 0, vsum, np.nan)
            base = pd.DataFrame(cum).rolling(20, min_periods=10).mean().shift(1).to_numpy()
            f["relvol_cum"] = cum / base
            lastv = np.where(v[:, :, b] > 0, v[:, :, b], np.nan)
            f["relvol_last"] = lastv / pd.DataFrame(lastv).rolling(20, min_periods=10).mean().shift(1).to_numpy()
        else:
            for k in ("dist_vwap", "relvol_cum", "relvol_last"):
                f[k] = np.full(cb.shape, np.nan)
        for k in ("r_open", "gap", "r_pc", "r_last1"):
            f[k + "_atr"] = f[k] / atr_pct
        f["atr_pct"] = atr_pct
        f["mkt_r_open"] = np.nanmedian(f["r_open"], axis=1, keepdims=True) * np.ones_like(cb)
        f["mkt_r_pc"] = np.nanmedian(f["r_pc"], axis=1, keepdims=True) * np.ones_like(cb)
        f["mkt_r_last1"] = np.nanmedian(f["r_last1"], axis=1, keepdims=True) * np.ones_like(cb)
        valid = ~np.isnan(f["r_open"])
        f["breadth"] = (np.sum(f["r_open"] > 0, axis=1, keepdims=True) / np.maximum(valid.sum(1, keepdims=True), 1)) * np.ones_like(cb)
        f["disp"] = np.nanstd(f["r_open"], axis=1, keepdims=True) * np.ones_like(cb)
        f["rel_r_open"] = f["r_open"] - f["mkt_r_open"]
        f["sec_r_open"] = _sector_median(f["r_open"], sector)
        f["secrel_r_open"] = f["r_open"] - f["sec_r_open"]
        f["rk_r_open"] = _xs_rank(f["r_open"])
        f["rk_r_pc"] = _xs_rank(f["r_pc"])
        f["rk_r_last1"] = _xs_rank(f["r_last1"])
        f["rk_relvol"] = _xs_rank(f["relvol_cum"])
        f["rk_gap"] = _xs_rank(f["gap"])
        for k in ("prev_ret", "ret5", "ret20", "prev_deliv", "prev_pos", "turn_rank", "prev_turn_ratio"):
            f[k] = ctx[k]
        # targets (raw returns, bps) and stop simulations
        e = b + 1
        entry = o[:, :, e]
        px_eod = _first_valid_close(c, e, EXIT_BAR)
        px_1h = c[:, :, e]
        f["y_eod"] = (px_eod / entry - 1) * 1e4
        f["y_1h"] = (px_1h / entry - 1) * 1e4
        atr_abs = atr_pct * entry
        for k in STOPS:
            for side, nm in ((1, "L"), (-1, "S")):
                for hz, xb in (("eod", EXIT_BAR), ("1h", e)):
                    r, hit = _stop_sim(arr, b, side, k, atr_abs, entry, xb)
                    f[f"pnl_{nm}_{hz}_s{k}"] = r
        f["entry"] = entry
        f["d"] = np.repeat(np.arange(len(days))[:, None], len(symbols), axis=1).astype(float)
        f["s"] = np.repeat(np.arange(len(symbols))[None, :], len(days), axis=0).astype(float)
        f["_bhav_gap"] = ctx["_bhav_gap"]
        df = pd.DataFrame({k: np.asarray(x, float).ravel() for k, x in f.items()})
        keep = (good_day[df.d.astype(int)] & df.r_open.notna() & df.entry.notna() & df.y_eod.notna()
                & df.turn_rank.notna() & df.atr_pct.notna() & (df.gap.abs() < 0.25) & (df.entry >= 20))
        frames.append(df[keep.to_numpy()])
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel["day"] = np.asarray(days)[panel.d.astype(int)]
    gap_err = (panel.gap - panel._bhav_gap).abs()
    check = {"bhav_coverage": ctx["_bhav_cov"], "gap_mismatch_gt1pct": float((gap_err > 0.01).mean()),
             "rows": int(len(panel)), "sessions": int(panel.day.nunique()), "symbols": int(panel.s.nunique()),
             "first": panel.day.min(), "last": panel.day.max()}
    panel = panel.drop(columns=["_bhav_gap"])
    # only liquid names are traded or trained on; cross-sectional features above were
    # computed on the full NIFTY-500 cross-section before this filter
    panel = panel[panel.turn_rank <= 300].reset_index(drop=True)
    check["rows_liquid"] = int(len(panel))
    for col in panel.columns:
        if panel[col].dtype == float and col not in ("d", "s"):
            panel[col] = panel[col].astype(np.float32)
    return panel, symbols, check


def load_panel():
    if PANEL.exists():
        return pd.read_parquet(PANEL)
    panel, _, _ = build()
    return panel


if __name__ == "__main__":
    import json
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    CACHE.mkdir(exist_ok=True)
    panel, symbols, check = build()
    panel.to_parquet(PANEL, compression="zstd")
    (HERE / "panel_check.json").write_text(json.dumps(check, indent=1))
    print(json.dumps(check, indent=1))
    print(panel.describe().T.to_string())
