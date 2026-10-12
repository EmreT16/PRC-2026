"""A1 — Model A doğrulama düzeneği.

- Hakem: split_1 (Ocak+Temmuz 2025 doğrulama, diğer 10 ay eğitim). Onay: split_2 (Şubat+Ağustos).
- Çeyrek CV (Q1–Q4, ardışık 3'er ay) yalnızca OOF gereken işler için (harman ağırlığı, hata ayrıştırması).
- Birincil metrik: ay ağırlıklı RMSE = sqrt(Σ w_m · MSE_m); ağırlıklar testteki Model A satırlarından.
- Kabul: önceden tanımlı 4 kural (ilk kod tabanımızdaki çerçeve). Ondan iki bilinçli fark:
  ΔRMSE ağırlıklı metrikte ölçülür; bootstrap günleri her ayın içinde yeniden örnekler.
- Hedef kodlama yalnızca eğitim katından. Eğitim satırlarının değeri kendi grubu (varsayılan: ay)
  dışındaki satırlardan gelir; testte TE başka bir dönemden (2025) geldiği için bu onu taklit eder.
- Skor gönderilen taksi üzerinden: taksi = max(60, NM_TAXI_DIFF − ofset). Ofset uzayında bu, tahminin
  NM_TAXI_DIFF − 60'ta üstten kesilmesine denk (floor_offset). compare() bunu nm_diff verilince uygular.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SPLITS = {"split_1": (1, 7), "split_2": (2, 8), "split_3": (12, 6), "split_4": (11, 3), "split_5": (4, 10), "split_6": (5, 9)}  # split_3 (30.09): ADS-B düzelticisinin eğitimi için Aralık+Haziran dışarıda bırakılmış tahminleri; split_4 (02.10): Kasım+Mart, aynı amaçla
QUARTERS = {"Q1": (1, 2, 3), "Q2": (4, 5, 6), "Q3": (7, 8, 9), "Q4": (10, 11, 12)}
MIN_DAYS_RATIO = 40 / 62        # 62 günün en az 40'ında iyileşme
MIN_DELTA_RMSE = -0.5           # ΔRMSE ≤ −0,5 sn
TAXI_FLOOR = 60.0               # gönderimdeki alt sınır (PRC_4\src\train_and_build_v126_leaves200.py:316,331)


# ----------------------------------------------------------------------------- bölmeler
def split_masks(months, name):
    """split_1 / split_2 için (eğitim maskesi, doğrulama maskesi, doğrulama ayları)."""
    val_months = SPLITS[name]
    val = np.isin(np.asarray(months), val_months)
    return ~val, val, val_months


def quarter_folds(months):
    """Çeyrek CV katları: [(ad, eğitim maskesi, doğrulama maskesi), ...]."""
    months = np.asarray(months)
    return [(q, ~np.isin(months, ms), np.isin(months, ms)) for q, ms in QUARTERS.items()]


def day_of_year(months, days, year=2025):
    """Yıl içindeki gün (1–365). Tabloya sütun olarak eklenmez; özellik listesine sızmasın."""
    return pd.to_datetime(pd.DataFrame({"year": year, "month": np.asarray(months), "day": np.asarray(days)})
                          ).dt.dayofyear.to_numpy()


def inner_es_mask(day_ids, frac=0.10, seed=config.SEED):
    """Erken durdurma için eğitim günlerinden rastgele gün bloku (doğrulama ayları hiç girmez)."""
    day_ids = np.asarray(day_ids)
    days = np.unique(day_ids)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(days, size=max(1, round(frac * len(days))), replace=False)
    return np.isin(day_ids, chosen)


# ----------------------------------------------------------------------------- metrik
def floor_offset(pred_offset, nm_diff, floor=TAXI_FLOOR):
    """Gönderimdeki alt sınırın ofset uzayındaki karşılığı: min(ofset, NM_TAXI_DIFF − floor).

    Gönderilen taksi max(floor, NM_TAXI_DIFF − ofset); taksi hatası bu etkili ofsetin hatasının eksi işaretlisi.
    """
    return np.minimum(np.asarray(pred_offset, float), np.asarray(nm_diff, float) - floor)


def month_weights(val_months):
    """Doğrulama aylarına test ağırlıkları: ilk ay ↔ Ocak, ikinci ay ↔ Temmuz."""
    first, second = val_months
    return {first: config.W_FIRST, second: config.W_SECOND}


def weighted_rmse(y, pred, months, weights):
    """Ay ağırlıklı RMSE; ay bazında RMSE ve ağırlıksız genel RMSE ile birlikte."""
    err2 = (np.asarray(pred, float) - np.asarray(y, float)) ** 2
    months = np.asarray(months)
    out = {"months": {}}
    wmse = 0.0
    for m, w in weights.items():
        sel = months == m
        mse = float(err2[sel].mean())
        out["months"][int(m)] = {"n": int(sel.sum()), "rmse": math.sqrt(mse)}
        wmse += w * mse
    out["weighted_rmse"] = math.sqrt(wmse)
    out["overall_rmse"] = float(np.sqrt(err2.mean()))
    return out


def days_won(err2_base, err2_new, day_ids):
    """Adayın günlük MSE'si tabandan düşük olan gün sayısı ve toplam gün."""
    g = pd.DataFrame({"d": day_ids, "b": err2_base, "n": err2_new}).groupby("d")[["b", "n"]].mean()
    return int((g["n"] < g["b"]).sum()), int(len(g))


def stratified_day_bootstrap(err2_base, err2_new, months, day_ids, weights, n_boot=2000, seed=config.SEED):
    """Günleri her ayın içinde yeniden örnekler; ağırlıklı MSE ve RMSE farkının %95 aralığı.

    Taban ve aday aynı örneklerle hesaplanır (eşleştirilmiş).
    """
    g = (pd.DataFrame({"m": months, "d": day_ids, "b": err2_base, "n": err2_new})
         .groupby(["m", "d"], sort=True)
         .agg(b=("b", "sum"), n=("n", "sum"), c=("b", "size"))
         .reset_index())
    rng = np.random.default_rng(seed)
    wmse_b = np.zeros(n_boot)
    wmse_n = np.zeros(n_boot)
    for m, w in weights.items():
        gm = g[g["m"] == m]
        idx = rng.integers(0, len(gm), size=(n_boot, len(gm)))
        cnt = gm["c"].to_numpy()[idx].sum(axis=1)
        wmse_b += w * gm["b"].to_numpy()[idx].sum(axis=1) / cnt
        wmse_n += w * gm["n"].to_numpy()[idx].sum(axis=1) / cnt
    d_mse = wmse_n - wmse_b
    d_rmse = np.sqrt(wmse_n) - np.sqrt(wmse_b)
    return {
        "dmse_ci": [float(np.percentile(d_mse, 2.5)), float(np.percentile(d_mse, 97.5))],
        "drmse_ci": [float(np.percentile(d_rmse, 2.5)), float(np.percentile(d_rmse, 97.5))],
        "p_not_better": float(np.mean(d_mse >= 0)),
    }


def compare(y, pred_base, pred_new, months, day_ids, val_months, *, nm_diff, n_boot=2000, seed=config.SEED):
    """Adayı tabanla 4 kabul kuralına göre karşılaştırır.

    nm_diff: NM_TAXI_DIFF_SEC; verilirse iki tahmine de gönderimdeki 60 sn alt sınır uygulanır (floor_offset).
    Model A karşılaştırmalarında her zaman verilmeli; None yalnızca sentetik testler için.
    """
    if nm_diff is not None:
        pred_base, pred_new = floor_offset(pred_base, nm_diff), floor_offset(pred_new, nm_diff)
    y = np.asarray(y, float)
    months = np.asarray(months)
    day_ids = np.asarray(day_ids)
    w = month_weights(val_months)
    e_b = (np.asarray(pred_base, float) - y) ** 2
    e_n = (np.asarray(pred_new, float) - y) ** 2
    base = weighted_rmse(y, pred_base, months, w)
    new = weighted_rmse(y, pred_new, months, w)
    delta = new["weighted_rmse"] - base["weighted_rmse"]
    won, n_days = days_won(e_b, e_n, day_ids)
    min_days = math.ceil(MIN_DAYS_RATIO * n_days)
    boot = stratified_day_bootstrap(e_b, e_n, months, day_ids, w, n_boot, seed)
    rules = {
        "r1_delta": delta <= MIN_DELTA_RMSE,
        "r2_iki_ay": all(new["months"][m]["rmse"] < base["months"][m]["rmse"] for m in new["months"]),
        "r3_gunler": won >= min_days,
        "r4_guven": boot["dmse_ci"][1] < 0,
    }
    return {
        "base": base, "new": new, "delta_weighted_rmse": delta,
        "days_won": won, "n_days": n_days, "min_days": min_days,
        "bootstrap": boot, "rules": rules, "accepted": all(rules.values()), "alt_sinir": nm_diff is not None,
    }


def format_comparison(res, title=""):
    """compare() sonucunun kısa Türkçe özeti."""
    b, n = res["base"], res["new"]
    lines = [(f"== {title}" if title else "==") + ("" if res.get("alt_sinir") else "  [60 sn alt sınır UYGULANMADI]"),
             f"  ağırlıklı RMSE: taban {b['weighted_rmse']:.3f} | aday {n['weighted_rmse']:.3f} | "
             f"Δ {res['delta_weighted_rmse']:+.3f} sn "
             f"[%95: {res['bootstrap']['drmse_ci'][0]:+.3f} … {res['bootstrap']['drmse_ci'][1]:+.3f}]"]
    for m in n["months"]:
        lines.append(f"  ay {m:2d}: taban {b['months'][m]['rmse']:.3f} | aday {n['months'][m]['rmse']:.3f} "
                     f"(n={n['months'][m]['n']:,})")
    r = res["rules"]
    lines.append(f"  kurallar: Δ≤−0,5 {'✓' if r['r1_delta'] else '✗'} · iki ay {'✓' if r['r2_iki_ay'] else '✗'} · "
                 f"gün {res['days_won']}/{res['n_days']} (≥{res['min_days']}) {'✓' if r['r3_gunler'] else '✗'} · "
                 f"güven aralığı < 0 {'✓' if r['r4_guven'] else '✗'}  →  "
                 f"{'KABUL' if res['accepted'] else 'RET'}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- hedef kodlama
def grouped_oof_te(keys_train, y_train, groups_train, keys_apply, m=20.0, stat="mean"):
    """Yumuşatılmış hedef kodlama; TE v2 formülü: (n·ort + m·öncül) / (n + m).

    Eğitim satırları: kendi grubu dışındaki eğitim satırlarından (gruplu OOF; öncül de grup dışından).
    keys_apply satırları (doğrulama/test): eğitim katının tamamından.
    stat='std': yumuşatılmış standart sapma (ddof=1; n < 2 ise öncül std), TE v2'deki STAND_RWY_STD gibi.
    Dönüş: (eğitim değerleri, uygulama değerleri), float32.
    """
    codes, uniques = pd.factorize(_as_keys(keys_train))
    gcodes, _ = pd.factorize(np.asarray(groups_train), use_na_sentinel=False)
    y = np.asarray(y_train, float)
    n_k, n_g = len(uniques), int(gcodes.max()) + 1
    comb = codes.astype(np.int64) * n_g + gcodes
    size = n_k * n_g
    cnt_kg = np.bincount(comb, minlength=size).reshape(n_k, n_g).astype(float)
    sum_kg = np.bincount(comb, weights=y, minlength=size).reshape(n_k, n_g)
    ssq_kg = np.bincount(comb, weights=y * y, minlength=size).reshape(n_k, n_g)
    cnt_k, sum_k, ssq_k = cnt_kg.sum(1), sum_kg.sum(1), ssq_kg.sum(1)
    cnt_g, sum_g, ssq_g = cnt_kg.sum(0), sum_kg.sum(0), ssq_kg.sum(0)
    N, S, SS = cnt_k.sum(), sum_k.sum(), ssq_k.sum()

    # eğitim satırları: anahtar ve öncül istatistikleri kendi grubu çıkarılarak
    n = cnt_k[codes] - cnt_kg[codes, gcodes]
    s = sum_k[codes] - sum_kg[codes, gcodes]
    ss = ssq_k[codes] - ssq_kg[codes, gcodes]
    pn, ps, pss = N - cnt_g[gcodes], S - sum_g[gcodes], SS - ssq_g[gcodes]
    tr = _smooth(n, s, ss, pn, ps, pss, m, stat)

    # uygulama satırları: eğitim katının tamamı; görülmemiş anahtar → öncül
    idx = pd.Index(uniques).get_indexer(_as_keys(keys_apply))
    seen = idx >= 0
    n_a = np.where(seen, cnt_k[np.maximum(idx, 0)], 0.0)
    s_a = np.where(seen, sum_k[np.maximum(idx, 0)], 0.0)
    ss_a = np.where(seen, ssq_k[np.maximum(idx, 0)], 0.0)
    ap = _smooth(n_a, s_a, ss_a, N, S, SS, m, stat)
    return tr.astype(np.float32), ap.astype(np.float32)


def _as_keys(keys):
    """Anahtarları metne çevirir; eksik değer 'nan' olur.

    pandas 3'te astype(str) eksik değeri NaN bırakıyor; eski betikler (pandas < 3) 'nan' metni üretiyordu.
    """
    s = pd.Series(keys).astype(object)
    return s.where(s.notna(), "nan").astype(str).to_numpy(dtype=object)


def _smooth(n, s, ss, pn, ps, pss, m, stat):
    prior_mean = ps / pn
    if stat == "mean":
        return (s + m * prior_mean) / (n + m)
    if stat == "std":
        prior_std = np.sqrt(np.maximum(pss - ps * ps / pn, 0.0) / (pn - 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.maximum(ss - s * s / n, 0.0) / (n - 1)
        std = np.where(n >= 2, np.sqrt(var), prior_std)
        return (n * std + m * prior_std) / (n + m)
    raise ValueError(f"bilinmeyen stat: {stat}")


# ----------------------------------------------------------------------------- kütük
def code_fingerprint():
    """src altındaki .py dosyalarının ortak MD5'i (git yokken sürüm izi)."""
    h = hashlib.md5()
    for p in sorted((config.ROOT / "src").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


CODE_MD5 = code_fingerprint()   # süreç başındaki kod; çalışma sırasında dosyalar değişse de doğru kalır


def log_experiment(record, path=None):
    """Deney kaydını kutuk\\deneyler.jsonl dosyasına bir satır olarak ekler."""
    path = Path(path or config.LOG_DIR / "deneyler.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"zaman": time.strftime("%Y-%m-%d %H:%M:%S"), "kod_md5": CODE_MD5, **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=_to_builtin) + "\n")


def _to_builtin(o):
    if isinstance(o, (np.bool_, np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
