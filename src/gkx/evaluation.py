from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def panel_r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    denom = float(np.sum(y**2))
    return np.nan if denom == 0 else 1.0 - float(np.sum((y - p) ** 2)) / denom


def _valid_market_cap(df, cap="mve"):
    values = pd.to_numeric(df[cap], errors="coerce")
    return values.notna() & np.isfinite(values) & (values > 0)


def r2_table(pred, target="ret_excess", cap="mve", n=1000):
    rows = []
    for model, g in pred.groupby("model"):
        rows.append(
            {
                "model": model,
                "sample": "all",
                "n_obs": len(g),
                "r2_oos": panel_r2(g[target], g["prediction"]),
            }
        )
        valid = g.loc[_valid_market_cap(g, cap)].copy()
        rank = valid.groupby("date")[cap].rank(method="first", ascending=False)
        count = valid.groupby("date")[cap].transform("count")
        for sample, mask in [("top", rank <= n), ("bottom", rank > count - n)]:
            z = valid[mask]
            rows.append(
                {
                    "model": model,
                    "sample": f"{sample}_{n}",
                    "n_obs": len(z),
                    "r2_oos": panel_r2(z[target], z["prediction"]),
                }
            )
    return pd.DataFrame(rows)


def dm_test(a, b, target="ret_excess", lags=6):
    z = a[["date", "id", target, "prediction"]].merge(
        b[["date", "id", "prediction"]],
        on=["date", "id"],
        suffixes=("_a", "_b"),
    )
    z["d"] = (z[target] - z.prediction_a) ** 2 - (z[target] - z.prediction_b) ** 2
    d = z.groupby("date").d.mean().sort_index()
    fit = sm.OLS(d.values, np.ones((len(d), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(fit.tvalues[0]), float(fit.pvalues[0])


def dm_matrix(pred, target="ret_excess", lags=6):
    """Full ordered DM table, evaluating each unordered pair only once."""
    models = sorted(pred.model.unique())
    out = []
    for i, a in enumerate(models):
        out.append({"row_model": a, "column_model": a, "dm_t": 0.0, "p_value": 1.0})
        for b in models[i + 1 :]:
            t, p = dm_test(
                pred[pred.model == a],
                pred[pred.model == b],
                target,
                lags,
            )
            out.append({"row_model": a, "column_model": b, "dm_t": t, "p_value": p})
            out.append({"row_model": b, "column_model": a, "dm_t": -t, "p_value": p})
    return pd.DataFrame(out).sort_values(["row_model", "column_model"]).reset_index(drop=True)


def assign_prediction_bins(pred, bins=10):
    z = pred.copy()
    z["bin"] = z.groupby(["model", "date"])["prediction"].transform(
        lambda x: pd.qcut(x.rank(method="first"), bins, labels=False) + 1
    )
    return z


def portfolio_returns(pred, bins=10):
    z = assign_prediction_bins(pred, bins)
    ew = z.groupby(["model", "date", "bin"]).ret_excess.mean().rename("ew_ret").reset_index()
    zv = z.loc[_valid_market_cap(z)].copy()
    zv["vw_num"] = zv.ret_excess * zv.mve
    vw = (
        zv.groupby(["model", "date", "bin"])
        .agg(num=("vw_num", "sum"), den=("mve", "sum"))
        .assign(vw_ret=lambda x: x.num / x.den)
        .reset_index()
    )
    out = ew.merge(
        vw[["model", "date", "bin", "vw_ret"]],
        on=["model", "date", "bin"],
        how="left",
    )
    lo = out[out.bin == 1]
    hi = out[out.bin == bins]
    spread = hi.merge(lo, on=["model", "date"], suffixes=("_hi", "_lo"))
    spread["ew_ret"] = spread.ew_ret_hi - spread.ew_ret_lo
    spread["vw_ret"] = spread.vw_ret_hi - spread.vw_ret_lo
    spread["bin"] = "H-L"
    return pd.concat(
        [out, spread[["model", "date", "bin", "ew_ret", "vw_ret"]]],
        ignore_index=True,
    )


def portfolio_stats(rets, annualization=12):
    rows = []
    for (model, portfolio), g in rets.groupby(["model", "bin"]):
        for col, weighting in [("ew_ret", "ew"), ("vw_ret", "vw")]:
            x = g[col].dropna()
            mu = x.mean() * annualization
            vol = x.std(ddof=1) * np.sqrt(annualization)
            rows.append(
                {
                    "model": model,
                    "portfolio": portfolio,
                    "weighting": weighting,
                    "ann_mean": mu,
                    "ann_vol": vol,
                    "sharpe": mu / vol if vol else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _spread_target_weights(g: pd.DataFrame, bins: int, weighting: str) -> pd.DataFrame:
    top = g[g.bin == bins].copy()
    bottom = g[g.bin == 1].copy()
    if weighting == "ew":
        top["w"] = 1.0 / len(top)
        bottom["w"] = -1.0 / len(bottom)
    elif weighting == "vw":
        top = top.loc[_valid_market_cap(top)].copy()
        bottom = bottom.loc[_valid_market_cap(bottom)].copy()
        if top.empty or bottom.empty:
            return pd.DataFrame(columns=["id", "ret_total", "w"])
        top["w"] = top.mve / top.mve.sum()
        bottom["w"] = -bottom.mve / bottom.mve.sum()
    else:
        raise ValueError(weighting)
    return pd.concat([top, bottom], ignore_index=True)[["id", "ret_total", "w"]]


def turnover_series(pred, bins=10):
    """GKX spread turnover with return-induced drift, using stock total returns."""
    if "ret_total" not in pred.columns:
        raise ValueError("turnover_series requires ret_total; excess returns are not interchangeable")
    z = assign_prediction_bins(pred, bins)
    rows = []
    for model, gm in z.groupby("model"):
        by_date = {d: g for d, g in gm.groupby("date", sort=True)}
        dates = sorted(by_date)
        for weighting in ("ew", "vw"):
            targets = {
                date: _spread_target_weights(by_date[date], bins, weighting).set_index("id")
                for date in dates
            }
            for prev_date, date in zip(dates[:-1], dates[1:]):
                prev = targets[prev_date]
                cur = targets[date]
                if prev.empty or cur.empty:
                    rows.append(
                        {
                            "model": model,
                            "date": date,
                            "weighting": weighting,
                            "turnover": np.nan,
                        }
                    )
                    continue
                ids = prev.index.union(cur.index)
                prev_w = prev["w"].reindex(ids, fill_value=0.0)
                prev_r = prev["ret_total"].reindex(ids, fill_value=0.0)
                cur_w = cur["w"].reindex(ids, fill_value=0.0)
                gross = 1.0 + float((prev_w * prev_r).sum())
                if not np.isfinite(gross) or abs(gross) < 1e-12:
                    turn = np.nan
                else:
                    drifted = prev_w * (1.0 + prev_r) / gross
                    # GKX report the sum of absolute trades; there is no 1/2 convention here.
                    turn = float((cur_w - drifted).abs().sum())
                rows.append(
                    {
                        "model": model,
                        "date": date,
                        "weighting": weighting,
                        "turnover": turn,
                    }
                )
    return pd.DataFrame(rows)


def turnover_from_holdings(pred, bins=10):
    monthly = turnover_series(pred, bins)
    return (
        monthly.groupby(["model", "weighting"], as_index=False)
        .agg(monthly_turnover=("turnover", "mean"), months=("turnover", "count"))
    )
