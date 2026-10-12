"""validation.py için sentetik testler. Çalıştırma: python tests/test_validation.py"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402


def _synthetic(n_per_day=400, seed=0):
    """Ocak ve Temmuz (62 gün) için sentetik doğrulama verisi."""
    rng = np.random.default_rng(seed)
    days = [(1, d) for d in range(1, 32)] + [(7, d) for d in range(1, 32)]
    months = np.repeat([m for m, _ in days], n_per_day)
    day_ids = np.repeat(np.arange(len(days)), n_per_day)
    y = rng.normal(40, 370, size=len(months))
    return y, months, day_ids, rng


def test_weighted_rmse():
    y = np.zeros(6)
    pred = np.array([1, 1, 1, 2, 2, 2], float)
    months = np.array([1, 1, 1, 7, 7, 7])
    res = V.weighted_rmse(y, pred, months, V.month_weights((1, 7)))
    assert math.isclose(res["weighted_rmse"], math.sqrt(config.W_FIRST * 1 + config.W_SECOND * 4))
    assert math.isclose(res["months"][7]["rmse"], 2.0)


def test_compare_identical_rejected():
    y, months, day_ids, rng = _synthetic()
    p = y + rng.normal(0, 300, size=len(y))
    res = V.compare(y, p, p.copy(), months, day_ids, (1, 7), nm_diff=None, n_boot=300)
    assert res["delta_weighted_rmse"] == 0 and not res["accepted"]
    assert res["days_won"] == 0 and not res["rules"]["r4_guven"]


def test_compare_clearly_better_accepted():
    y, months, day_ids, rng = _synthetic()
    noise = rng.normal(0, 1, size=len(y))
    base, new = y + 300 * noise, y + 250 * noise + rng.normal(0, 20, size=len(y))
    res = V.compare(y, base, new, months, day_ids, (1, 7), nm_diff=None, n_boot=300)
    assert res["accepted"], res["rules"]


def test_compare_tiny_gain_rejected():
    y, months, day_ids, rng = _synthetic()
    noise = rng.normal(0, 300, size=len(y))
    res = V.compare(y, y + noise, y + 0.9995 * noise, months, day_ids, (1, 7), nm_diff=None, n_boot=300)
    assert res["delta_weighted_rmse"] < 0 and not res["rules"]["r1_delta"] and not res["accepted"]


def test_floor_matches_taxi_space():
    """Ofset uzayındaki alt sınır, taksi uzayındaki max(60, ·) ile aynı hatayı vermeli."""
    rng = np.random.default_rng(3)
    nm = rng.normal(900, 2000, size=2000)
    y = rng.normal(0, 800, size=2000)
    pred = rng.normal(0, 800, size=2000)
    taxi_err = np.maximum(V.TAXI_FLOOR, nm - pred) - (nm - y)
    assert np.allclose(np.abs(V.floor_offset(pred, nm) - y), np.abs(taxi_err))
    months = np.repeat([1, 7], 1000)
    day_ids = np.repeat(np.arange(40), 50)
    res = V.compare(y, pred, pred, months, day_ids, (1, 7), nm_diff=nm, n_boot=50)
    assert res["alt_sinir"] and math.isclose(res["base"]["overall_rmse"], float(np.sqrt(np.mean(taxi_err ** 2))))


def _naive_te(keys, y, groups, apply_keys, m, stat):
    """Saf pandas karşılaştırması: her grup için grup dışı satırlardan TE."""
    df = pd.DataFrame({"k": keys, "y": y, "g": groups})
    out = np.empty(len(df))
    for g in df["g"].unique():
        rest, own = df[df["g"] != g], df["g"] == g
        out[own.to_numpy()] = _naive_map(rest, df.loc[own, "k"], m, stat)
    return out, _naive_map(df, pd.Series(apply_keys), m, stat)


def _naive_map(train, keys, m, stat):
    prior_mean, prior_std = train["y"].mean(), train["y"].std()
    st = train.groupby("k")["y"].agg(["count", "mean", "std"])
    if stat == "mean":
        val = (st["count"] * st["mean"] + m * prior_mean) / (st["count"] + m)
        return keys.map(val).fillna(prior_mean).to_numpy()
    std = st["std"].fillna(prior_std)
    val = (st["count"] * std + m * prior_std) / (st["count"] + m)
    return keys.map(val).fillna(prior_std).to_numpy()


def test_te_matches_naive():
    rng = np.random.default_rng(1)
    n = 5000
    keys = rng.choice([f"k{i}" for i in range(60)], size=n)
    groups = rng.integers(1, 11, size=n)
    y = rng.normal(0, 100, size=n) + pd.Series(keys).str[1:].astype(int).to_numpy() * 3
    apply_keys = rng.choice([f"k{i}" for i in range(70)], size=800)   # k60–k69 eğitimde yok
    for stat, m in (("mean", 20.0), ("std", 20.0)):
        tr, ap = V.grouped_oof_te(keys, y, groups, apply_keys, m=m, stat=stat)
        tr_ref, ap_ref = _naive_te(keys, y, groups, apply_keys, m, stat)
        assert np.allclose(tr, tr_ref, rtol=1e-4, atol=1e-3), stat
        assert np.allclose(ap, ap_ref, rtol=1e-4, atol=1e-3), stat


def test_te_no_self_leak():
    """Her anahtar tek bir grupta geçiyorsa eğitim satırının TE'si yalnızca grup dışı öncül olmalı."""
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(1, 6), 200)
    keys = np.array([f"g{g}_{i % 7}" for g, i in zip(groups, range(len(groups)))])
    y = rng.normal(0, 50, size=len(groups)) + groups * 100
    tr, _ = V.grouped_oof_te(keys, y, groups, keys[:5], m=20.0)
    for g in np.unique(groups):
        assert np.allclose(tr[groups == g], y[groups != g].mean(), rtol=1e-5)


def test_te_missing_keys():
    """Eksik anahtar 'nan' grubu gibi davranmalı; uygulamadaki eksik anahtar da aynı gruba düşmeli."""
    keys = np.array(["a", None, "a", None, "b", None] * 50, dtype=object)
    groups = np.repeat([1, 2, 3], 100)
    y = np.where(pd.isna(keys), 500.0, 0.0)
    tr, ap = V.grouped_oof_te(keys, y, groups, np.array([None, "nan", "zzz"], dtype=object), m=1.0)
    assert np.isfinite(tr).all() and ap[0] == ap[1] and ap[0] > 400 > ap[2]


def test_splits_and_es():
    months = np.repeat(np.arange(1, 13), 10)
    tr, va, vm = V.split_masks(months, "split_1")
    assert vm == (1, 7) and set(months[va]) == {1, 7} and not (tr & va).any()
    folds = V.quarter_folds(months)
    assert sum(f[2].sum() for f in folds) == len(months)
    day_ids = np.repeat(np.arange(300), 5)
    es = V.inner_es_mask(day_ids, frac=0.1)
    assert len(np.unique(day_ids[es])) == 30
    assert not (np.isin(day_ids[~es], day_ids[es])).any()      # gün bölünmüyor


if __name__ == "__main__":
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"OK  {name}")
    print(f"{len(tests)} test geçti")
