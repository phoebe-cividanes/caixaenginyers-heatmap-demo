# utils/projections.py
from __future__ import annotations
import numpy as np, pandas as pd

def cagr(v0: float, v1: float, years: float) -> float:
    if years <= 0 or v0 is None or v0 <= 0 or not np.isfinite(v0) or not np.isfinite(v1):
        return 0.0
    return (v1 / v0) ** (1.0 / years) - 1.0

def apply_cagr_series(s: pd.Series, years_ahead: int, rate: float) -> pd.Series:
    return s.astype(float) * ((1.0 + rate) ** years_ahead)

def scenario_default_rates(scenario: str) -> dict:
    s = scenario.lower()
    if "optim" in s:
        return dict(acceso=-0.03, demanda=0.02, pib_pc=0.02, inclusion=0.02, digital=0.03,
                    alquiler_m2=0.01, competencia=0.01)
    if "pesim" in s:
        return dict(acceso=-0.005, demanda=0.005, pib_pc=0.005, inclusion=0.003, digital=0.005,
                    alquiler_m2=0.03, competencia=0.02)
    return dict(acceso=-0.02, demanda=0.01, pib_pc=0.012, inclusion=0.01, digital=0.015,
                alquiler_m2=0.02, competencia=0.01)

def blend_rate(empirical: float | None, scenario_rate: float, alpha: float = 0.6) -> float:
    if empirical is None or not np.isfinite(empirical):
        return scenario_rate
    return alpha * empirical + (1 - alpha) * scenario_rate

def project_dataframe(df: pd.DataFrame, years_ahead: int, scenario: str = "Base",
                      empirical_rates: dict[str, pd.Series] | None = None,
                      alpha: float = 0.6) -> pd.DataFrame:
    rates = scenario_default_rates(scenario)
    out = df.copy()

    def col_rate(col: str):
        scen = rates.get(col, 0.0)
        if empirical_rates and col in empirical_rates:
            return empirical_rates[col].apply(lambda r: blend_rate(r, scen, alpha))
        return scen

    for col in ["acceso", "demanda", "pib_pc", "inclusion", "digital", "alquiler_m2", "competencia"]:
        if col not in out.columns: 
            continue
        r = col_rate(col)
        if isinstance(r, pd.Series):
            out[f"{col}_t"] = out[col].astype(float) * (1.0 + r) ** years_ahead
        else:
            out[f"{col}_t"] = apply_cagr_series(out[col], years_ahead, r)
    return out

