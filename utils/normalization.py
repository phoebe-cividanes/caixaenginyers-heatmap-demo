# utils/normalization.py
from __future__ import annotations
import numpy as np, pandas as pd

def minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    vmin, vmax = float(s.min()), float(s.max())
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
        return (s - vmin) / (vmax - vmin)
    return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)

def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu, sd = float(s.mean()), float(s.std())
    if sd == 0 or not np.isfinite(sd):
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - mu) / sd

def winsorize(s: pd.Series, q: float = 0.01) -> pd.Series:
    s = s.astype(float).copy()
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)

def robust_minmax(s: pd.Series, q: float = 0.01) -> pd.Series:
    return minmax(winsorize(s, q=q))

def normalize_dataframe(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    spec[col] = {"method": "minmax|robust|zscore", "invert": bool}
    Crea columnas *_norm aplicando método y, si invert=True, invierte (1-x).
    """
    out = df.copy()
    for col, cfg in spec.items():
        if col not in out.columns: 
            continue
        method = cfg.get("method", "minmax")
        invert = bool(cfg.get("invert", False))
        if method == "zscore":
            v = zscore(out[col])
            v = 0.5 * (1 + (v / np.sqrt(1 + v**2)))  # map z ~ [0,1] (suave)
        elif method == "robust":
            v = robust_minmax(out[col])
        else:
            v = minmax(out[col])
        if invert:
            v = 1.0 - v
        out[f"{col}_norm"] = v
    return out
