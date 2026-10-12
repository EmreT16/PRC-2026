"""11.10: EUROCONTROL'ün havalimanı-günü toplam kalkış öncesi gecikmesi
(all_pre_departure_delays, DLY_ALL_PRE_2, dk) 2025 etiketlerinde Σ max(0, off-block − tarife) ile birebir (off-block UTC günü;
oran medyanı 0,9999, günlerin %95'i %1 içinde). Bu dosyanın 2026 değerleri v208'den beri özellik olarak kullanılıyor (GS_*).
Fikir: toplamı kısıt olarak kullan — her tahmini, gün toplamı verildiğinde koşullu
beklenen değeriyle değiştir.
Yöntem (Gauss yaklaşımı): y_i = ŷ_i + e_i, e_i ~ N(0, s_i²); δ_i = d_i − y_i (d = kalkış − tarife), katkı c_i = max(0, δ_i).
E[c_i] = s φ(z) + δ̂ Φ(z), z = δ̂ / s; dc/de ≈ −Φ(z) = −g_i. Toplam T (sn) gözlenince
  ŷ_i ← ŷ_i − α · g_i s_i² (T − Σ E[c_j]) / (Σ g_j² s_j² + σ_T²),  σ_T = a · T + b.
s_i: havalimanı × (A izli / A izsiz / B) kalıntı std'si (aynı aylardan; yalnız ölçek). Gün = tahmini off-block UTC günü.
OOF: A = taban harman (kqgpPeT2) + düzeltici CV (Oca / Tem: ADSB_duz_cv_parkb_tabm2_lgbnn_planli; Şub / Haz:
ADSB_duz_cv_T2_sh_lgbnn_pl); B = BM_cv_tum_guncel_oof (tüm B satırları; LIRF / EDDF için yaklaşık).
Önceden tanımlı kabul (ilk sınama, ızgara α ∈ {0,5, 1}, (a, b) ∈ {(0,01, 300), (0,03, 600), (0,1, 1800)}):
  iç içe ay seçimi (3 ayda en iyi ayar → 4. ay) toplamda iyi VE en iyi ayarda 4/4 ay iyi.
Kullanım: python scripts/gunluk_kisit_analiz.py [yalniz_a=1]
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

os.environ["PRC_HARMAN"] = "kqgpPeT2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config, model_b as MB  # noqa: E402

P = config.DATA_DIR / "preds"
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
AYLAR = [1, 2, 6, 7]
IZGARA = [(al, a, b) for al in (0.5, 1.0) for a, b in ((0.01, 300.0), (0.03, 600.0), (0.1, 1800.0))]
# 11.10 ikinci tur (izgara=2): daha sıkı σ_T ve yineleme; s_b_ust=<sn> B std tavanı
IZGARA2 = [(1.0, a, b) for a, b in ((0.003, 60.0), (0.005, 120.0), (0.01, 300.0), (0.02, 300.0))]


def veri():
    fs = sorted(glob.glob(str(config.PRC2_DIR / "data" / "training_2025-*.parquet")))
    r = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                                                "TAXITIME_SEC_mvt"]) for f in fs])
    r = r[r.ADEP_mvt.isin(APT) & r.PHASE_mvt.eq("DEP")].copy()
    r["ay"] = r.MVT_TIME_UTC_mvt.dt.month
    r = r[r.ay.isin(AYLAR)].reset_index(drop=True)
    r["d"] = (r.MVT_TIME_UTC_mvt - r.SCHED_TIME_UTC_mvt).dt.total_seconds()
    # A tahminleri
    parca = []
    for sp, aylar, duz in (("split_1", (1, 7), "ADSB_duz_cv_parkb_tabm2_lgbnn_planli"), ("split_2", (2,), "ADSB_duz_cv_T2_sh_lgbnn_pl"),
                           ("split_3", (6,), "ADSB_duz_cv_T2_sh_lgbnn_pl")):
        f = A.d_dw_split(sp)
        f = f[f.MVT_MONTH.isin(aylar)][["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", "ofset_taban"]]
        f = f.merge(pd.read_parquet(P / f"{duz}.parquet")[["MVT_ID_mvt", "rhat"]], on="MVT_ID_mvt", how="left")
        f["iz"] = f.rhat.notna() & (f.rhat.abs() > 0)
        f["p"] = np.maximum(60.0, f.NM_TAXI_DIFF_SEC - f.ofset_taban + f.rhat.fillna(0.0))
        parca.append(f[["MVT_ID_mvt", "p", "iz"]].assign(tur="A"))
    b = pd.read_parquet(P / "BM_cv_tum_guncel_oof.parquet").rename(columns={"tum_mu_yt_tavan3500": "p"})
    parca.append(b[["MVT_ID_mvt", "p"]].assign(iz=False, tur="B"))
    t = pd.concat(parca, ignore_index=True)
    r = r.merge(t, on="MVT_ID_mvt", how="left", validate="one_to_one")
    eks = r.p.isna()
    print(f"kalkış {len(r):,} | tahminsiz {int(eks.sum()):,} (havalimanı-ay medyanıyla dolduruldu)")
    med = r[~eks].groupby(["ADEP_mvt", "ay"]).p.median()
    r.loc[eks, "p"] = [med[(a, m)] for a, m in zip(r.ADEP_mvt[eks], r.ay[eks])]
    r.loc[eks, "tur"] = "eksik"
    r["iz"] = r.iz.fillna(False).astype(bool)
    return r


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    r = veri()
    y = r.TAXITIME_SEC_mvt.to_numpy(float)
    p0 = r.p.to_numpy(float)
    grp = r.ADEP_mvt + "|" + np.where(r.tur.eq("B"), "B", np.where(r.iz, "Ai", "Az"))
    s = (pd.Series(y - p0).groupby(grp.to_numpy()).std()).clip(upper=3000.0)
    if "s_b_ust" in opt:
        s = pd.Series({k: (min(v, float(opt['s_b_ust'])) if k.endswith('|B') else v) for k, v in s.items()})
    sv = grp.map(s).to_numpy(float)
    if opt.get("sigma_yaz") == "1":  # 11.10: üretim (scripts/g_gunluk.py) için kalıntı std tablosu
        import json
        (config.ROOT / "kutuk" / "gunluk_kisit_sigma.json").write_text(json.dumps({k: round(float(v), 2) for k, v in s.items()},
                                                                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print("kutuk/gunluk_kisit_sigma.json yazıldı")
    print("kalıntı std (sn):", s.round(0).to_dict())
    g = pd.read_csv(config.EK_VERI / "ansperf" / "all_pre_departure_delays_2025.csv")
    g = g[g.APT_ICAO.isin(APT)]
    T = g.set_index([g.APT_ICAO, pd.to_datetime(g.FLT_DATE)]).DLY_ALL_PRE_2 * 60.0
    gun = (r.MVT_TIME_UTC_mvt.dt.tz_localize(None) if r.MVT_TIME_UTC_mvt.dt.tz is not None else r.MVT_TIME_UTC_mvt) - pd.to_timedelta(p0, unit="s")
    key = pd.MultiIndex.from_arrays([r.ADEP_mvt, gun.dt.normalize()])
    Tv = T.reindex(key).to_numpy(float)
    ok = np.isfinite(Tv)
    print(f"gün toplamı bulunan satır payı {ok.mean():.4f}")
    yalniz_a = opt.get("yalniz_a") == "1"
    ay = r.ay.to_numpy()
    tur = r.tur.to_numpy()
    apt = r.ADEP_mvt.to_numpy()
    kid = pd.factorize(key)[0]
    gn = gun.dt.normalize()
    eksik_gun = ((gn + pd.Timedelta(days=1)).dt.day.eq(1) | ~gn.dt.month.isin(AYLAR)).to_numpy()
    print(f"ay sonu / test ayı dışı off-block günündeki satır: {int(eksik_gun.sum()):,}")

    def duzelt(al, a, b, pb):
        d = r.d.to_numpy(float)
        dh = d - pb
        z = dh / sv
        ec = sv * norm.pdf(z) + dh * norm.cdf(z)
        gi = norm.cdf(z)
        # 11.10 bern=1: B satırlarının katkısı iki uçlu (yer tutucu ≈ 0 / normal ≈ d − 900) → Bernoulli varyansı paydaya ve
        # B düzeltmesine; yoksa B'nin belirsizliği A satırlarına yüklenir (LIRF A kötüleşmesi)
        vb = np.where(tur == "B", np.clip(dh, 0, None) * np.clip(np.maximum(d - 900.0, dh) - np.clip(dh, 0, None), 0, None), 0.0)
        vb = vb if opt.get("bern") == "1" else np.zeros(len(dh))
        V = np.bincount(kid, weights=(gi * sv) ** 2 + vb)  # payda tüm satırların belirsizliği (düzeltilmeyen B'ler dahil)
        if yalniz_a:
            gi = np.where(tur == "A", gi, 0.0)
        if "haric" in opt:  # 11.10: bu havalimanlarında hiçbir satır düzeltilmez (LIRF: A satırlarında da iki uçlu yankı rejimi)
            gi = np.where(np.isin(apt, opt["haric"].split(",")), 0.0, gi)
        elif "b_apt" in opt:  # düzeltilen B satırları yalnız bu havalimanlarında (öbürleri yalnız paydada)
            gi = np.where((tur != "B") | np.isin(apt, opt["b_apt"].split(",")), gi, 0.0)
        if opt.get("ay_sonu") == "0":  # 12.10: ay sonu / test ayı dışı günler düzeltilmez (veride günün son saatleri yok)
            gi = np.where(eksik_gun, 0.0, gi)
        S = np.bincount(kid, weights=ec)
        Tk = pd.Series(Tv).groupby(kid).first().reindex(range(kid.max() + 1)).to_numpy()
        sT = a * np.nan_to_num(Tk) + b
        k = np.where(np.isfinite(Tk), (Tk - S) / (V + sT ** 2), 0.0)
        dl = al * (gi * sv ** 2 + np.where(gi > 0, vb, 0.0)) * k[kid]
        if "tavan" in opt:  # 11.10: satır başına düzeltme tavanı (sn; veri hatalı günlere karşı güvenlik)
            dl = np.clip(dl, -float(opt["tavan"]), float(opt["tavan"]))
        return np.maximum(60.0, pb - dl)

    def olc(p):
        e = (p - y) ** 2
        o = {f"ay{m}": np.sqrt(e[ay == m].mean()) for m in AYLAR}
        o["hepsi"] = np.sqrt(e.mean())
        o["A"] = np.sqrt(e[tur == "A"].mean())
        o["B"] = np.sqrt(e[tur == "B"].mean())
        o["ay_sonu"] = np.sqrt(e[eksik_gun].mean())
        return o, e

    sat, E = {}, {}
    sat["taban"], E["taban"] = olc(p0)
    yin = int(opt.get("yineleme", 1))
    for al, a, b in (IZGARA2 if opt.get("izgara") == "2" else IZGARA):
        k = f"α{al} σT={a}T+{b:.0f}"
        p_ = p0.copy()
        for _ in range(yin):  # yineleme: doğrusallaştırmayı güncel tahmin etrafında yenile
            p_ = duzelt(al, a, b, p_)
        sat[k], E[k] = olc(p_)
    t = pd.DataFrame(sat).T
    print(t.round(2).to_string())
    ad = [k for k in E if k != "taban"]
    sse = 0.0
    for m in AYLAR:
        diger = ay != m
        en = min(ad, key=lambda k: E[k][diger].sum())
        sse += E[en][ay == m].sum()
        print(f"  ay {m}: diğer aylarla seçilen {en}")
    ic = np.sqrt(sse / len(y)) - t.loc["taban", "hepsi"]
    en = t.loc[ad, "hepsi"].idxmin()
    iyi = int(sum(t.loc[en, f"ay{m}"] < t.loc["taban", f"ay{m}"] for m in AYLAR))
    print(f"iç içe ay seçimi {ic:+.3f} | en iyi {en} ({t.loc[en, 'hepsi'] - t.loc['taban', 'hepsi']:+.3f}), iyi ay {iyi}/4 → "
          f"{'KABUL' if ic < 0 and iyi == 4 else 'RET'}")
    fark = (E[en] - E["taban"])
    print("havalimanı × tür (en iyi ayar) RMSE farkı:")
    x = pd.DataFrame({"apt": r.ADEP_mvt, "tur": tur, "e0": E["taban"], "e1": E[en]})
    print(x.groupby(["apt", "tur"]).apply(lambda q: np.sqrt(q.e1.mean()) - np.sqrt(q.e0.mean())).unstack().round(2).to_string())
    print(f"SSE payı: A {fark[tur == 'A'].sum() / fark.sum():.2f}, B {fark[tur == 'B'].sum() / fark.sum():.2f}")


if __name__ == "__main__":
    main()
