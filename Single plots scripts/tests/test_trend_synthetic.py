import numpy as np
import pandas as pd
from sen_slope_mann_kendall_trend_analysis import TrendAnalyzer


def _make_series(n=30, slope=0.01, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp('2000-01-01')
    dates = pd.to_datetime([t0 + pd.Timedelta(days=int(i * 365.25)) for i in range(n)])
    y = slope * np.arange(n, dtype=float) + noise * rng.normal(size=n)
    return pd.DataFrame({
        'date': dates,
        'Albedo': y
    })


def test_trend_positive_mk_sen():
    df = _make_series(n=40, slope=0.02, noise=0.01, seed=42)
    cfg = {
        'trend_analysis': {
            'alpha': 0.05,
            'prewhitening': False,
            'min_years': 5
        }
    }
    ta = TrendAnalyzer(cfg)
    res = ta.analyze_time_series(df, 'Albedo')
    assert 'annual' in res
    assert res['annual']['mann_kendall']['p_value'] < 0.05
    assert res['annual']['sen_slope']['slope_per_year'] > 0


def test_trend_none_mk_sen():
    df = _make_series(n=40, slope=0.0, noise=0.02, seed=1)
    cfg = {
        'trend_analysis': {
            'alpha': 0.05,
            'prewhitening': False,
            'min_years': 5
        }
    }
    ta = TrendAnalyzer(cfg)
    res = ta.analyze_time_series(df, 'Albedo')
    assert 'annual' in res
    assert res['annual']['sen_slope']['slope_per_year'] == 0 or abs(res['annual']['sen_slope']['slope_per_year']) < 0.01


def test_prewhitening_effect_ar1():
    n = 60
    phi = 0.6
    rng = np.random.default_rng(0)
    e = rng.normal(size=n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i-1] + e[i]
    t0 = pd.Timestamp('2000-01-01')
    dates = pd.to_datetime([t0 + pd.Timedelta(days=int(i * 365.25)) for i in range(n)])
    df = pd.DataFrame({'date': dates, 'Albedo': x})

    cfg_no = {'trend_analysis': {'alpha': 0.05, 'prewhitening': False, 'min_years': 5}}
    cfg_pw = {'trend_analysis': {'alpha': 0.05, 'prewhitening': True, 'min_years': 5}}
    ta_no = TrendAnalyzer(cfg_no)
    ta_pw = TrendAnalyzer(cfg_pw)
    res_no = ta_no.analyze_time_series(df, 'Albedo')
    res_pw = ta_pw.analyze_time_series(df, 'Albedo')

    p_no = res_no['annual']['mann_kendall']['p_value']
    p_pw = res_pw['annual']['mann_kendall']['p_value']
    assert p_pw >= p_no or abs(p_pw - p_no) < 1e-6
