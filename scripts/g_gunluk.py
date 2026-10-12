"""11.10: gönderim dosyasına günlük kalkış öncesi gecikme kısıtı (son işlem; Y1 ve
LOBT'den sonra). EUROCONTROL all_pre_departure_delays (DLY_ALL_PRE_2, dk) 2025'te Σ max(0, off-block − tarife) ile birebir
(off-block UTC günü). Her havalimanı-günü için Gauss koşullu beklenen değer (scripts/gunluk_kisit_analiz.py ile aynı formül):
  δ̂ = d − ŷ, z = δ̂ / s, E[c] = s φ(z) + δ̂ Φ(z), g = Φ(z);  ŷ ← ŷ − α g s² (T − Σ E[c]) / (Σ g² s² + σ_T²),  σ_T = a T + b.
Payda tüm satırların belirsizliğini içerir; düzeltilen satırlar: A (NM'li) satırlar (duz=A) ya da A + belirtilen havalimanlarının
B satırları (duz=A b_apt=LIRF,LTFM). s: kutuk/gunluk_kisit_sigma.json (2025 OOF; havalimanı × A izli / A izsiz / B, B ≤ 1.000).
İzli = A düzeltici dosyasında rhat ≠ 0. Gün = tahmini off-block UTC günü (kalkış − ŷ). Sonra 60 sn alt sınır ve LOBT penceresi.
Ayarlar (2025 Oca / Şub / Haz / Tem, iç içe ay seçimi): α = 1, a = 0,003, b = 60.
Kullanım: python scripts/g_gunluk.py <kaynak gönderim> [a_dosya=D_kqgpPeT2_adsb_42p9T] [b_apt=LIRF,LTFM] "<not>"
  (PRC_FINAL=1 → final testi, gonderim_final/)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # noqa: E402
import g_gonderim as G  # noqa: E402

AL, A_T, B_T, TAVAN = 1.0, 0.003, 60.0, 300.0  # TAVAN: satır başına düzeltme sınırı (sn; 11.10 CV 187,37 → 187,36)
APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def duzelt(mvt_id, p0, a_dosya, b_apt, haric=()):
    s_tab = json.loads((config.ROOT / "kutuk" / "gunluk_kisit_sigma.json").read_text(encoding="utf-8"))
    r = pd.read_parquet(config.RANK_RAW, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    r = pd.DataFrame({"MVT_ID_mvt": mvt_id}).merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    nm = pd.DataFrame({"MVT_ID_mvt": mvt_id}).merge(pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"]),
                                                    on="MVT_ID_mvt", how="left", validate="one_to_one")
    iz = pd.DataFrame({"MVT_ID_mvt": mvt_id}).merge(pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{a_dosya}.parquet",
                                                                    columns=["MVT_ID_mvt", "rhat"]), on="MVT_ID_mvt", how="left")
    B = nm.NM_TAXI_DIFF_SEC.isna().to_numpy()
    izli = (iz.rhat.fillna(0).abs() > 0).to_numpy() & ~B
    apt = r.ADEP_mvt.to_numpy()
    grp = np.char.add(np.char.add(apt.astype(str), "|"), np.where(B, "B", np.where(izli, "Ai", "Az")))
    sv = np.array([s_tab[g] for g in grp], float)
    mt = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.tz_localize(None)
    d = (mt - pd.to_datetime(r.SCHED_TIME_UTC_mvt, utc=True).dt.tz_localize(None)).dt.total_seconds().to_numpy(float)
    gun = (mt - pd.to_timedelta(p0, unit="s")).dt.normalize()
    g = pd.read_csv(config.EK_VERI / "ansperf" / "all_pre_departure_delays_2026.csv")
    g = g[g.APT_ICAO.isin(APT)]
    T = g.set_index([g.APT_ICAO, pd.to_datetime(g.FLT_DATE)]).DLY_ALL_PRE_2 * 60.0
    key = pd.MultiIndex.from_arrays([apt, gun])
    Tv = T.reindex(key).to_numpy(float)
    kid = pd.factorize(key)[0]
    dh = d - p0
    z = dh / sv
    ec = sv * norm.pdf(z) + dh * norm.cdf(z)
    gi = norm.cdf(z)
    # B satırlarının katkısı iki uçlu (yer tutucu ≈ 0 / normal ≈ d − 900): Bernoulli varyansı paydaya ve B düzeltmesine
    vb = np.where(B, np.clip(dh, 0, None) * np.clip(np.maximum(d - 900.0, dh) - np.clip(dh, 0, None), 0, None), 0.0)
    V = np.bincount(kid, weights=(gi * sv) ** 2 + vb)
    duz = (~B | np.isin(apt, b_apt)) & ~np.isin(apt, haric)  # haric: hiç düzeltilmeyen havalimanları (LIRF: iki uçlu yankı rejimi)
    gi_d = np.where(duz, gi, 0.0)
    S = np.bincount(kid, weights=ec)
    Tk = pd.Series(Tv).groupby(kid).first().reindex(range(kid.max() + 1)).to_numpy()
    sT = A_T * np.nan_to_num(Tk) + B_T
    k = np.where(np.isfinite(Tk), (Tk - S) / (V + sT ** 2), 0.0)
    dl = np.clip(AL * (gi_d * sv ** 2 + np.where(gi_d > 0, vb, 0.0)) * k[kid], -TAVAN, TAVAN)  # satır başına tavan (CV: zararsız)
    yeni = np.maximum(60.0, p0 - dl)
    oz = pd.DataFrame({"apt": apt, "tur": np.where(B, "B", "A"), "ay": mt.dt.month.to_numpy(), "f": yeni - p0})
    print(f"satır {len(p0):,} | gün toplamı bulunan {np.isfinite(Tv).mean():.4f} | havalimanı-günü {kid.max() + 1:,} | "
          f"düzeltilen B havalimanları {b_apt or '-'}")
    print("gün sapması (T − ΣE[c]) / T: medyan", np.round(np.nanmedian((Tk - S) / np.where(Tk > 0, Tk, np.nan)), 4),
          "| |sapma| > %10 gün payı", np.round(np.nanmean(np.abs((Tk - S) / np.where(Tk > 0, Tk, np.nan)) > 0.1), 3))
    print(oz.groupby(["apt", "tur"]).f.agg(["size", "mean", lambda q: float(np.sqrt((q ** 2).mean())), lambda q: float(q.abs().max())])
          .round(1).rename(columns={"<lambda_0>": "rms", "<lambda_1>": "maks"}).to_string())
    return yeni


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    poz = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    kaynak, note = poz[0], (poz[1] if len(poz) > 1 else "")
    b_apt = [x for x in opt.get("b_apt", "").split(",") if x]
    sub = pd.read_parquet(kaynak)
    t0 = sub["TAXITIME_SEC_mvt"].to_numpy(float)
    haric = [x for x in opt.get("haric", "").split(",") if x]
    yeni = duzelt(sub["MVT_ID_mvt"].to_numpy(), t0, opt.get("a_dosya", "D_kqgpPeT2_adsb_42p9T"), b_apt, haric)
    yeni, km = G.lobt_kirp(sub["MVT_ID_mvt"].to_numpy(), yeni)
    # 11.10 kural (d): LEMD / LEBL'de taksi 2025'te hiç 3.604 sn'yi aşmadı (≈ 400 bin uçuş) → üst sınır
    ap = pd.DataFrame({"MVT_ID_mvt": sub["MVT_ID_mvt"].to_numpy()}).merge(pd.read_parquet(config.RANK_RAW, columns=["MVT_ID_mvt", "ADEP_mvt"]),
                                                                     on="MVT_ID_mvt", how="left")["ADEP_mvt"].to_numpy()
    ust = np.isin(ap, ["LEMD", "LEBL"]) & (yeni > 3604.0)
    yeni = np.where(ust, 3604.0, yeni)
    print(f"LEMD / LEBL 3.604 sn üst sınırı: {int(ust.sum())} satır")
    d = yeni - t0
    print(f"değişen satır {int((np.abs(d) > 1e-6).sum()):,} | RMS {np.sqrt(np.mean(d ** 2)):.2f} | ort {d.mean():+.2f} | maks |fark| "
          f"{np.abs(d).max():.0f} | LOBT kırpılan {int(km.sum())}")
    out = pd.DataFrame({"MVT_ID_mvt": sub["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": yeni.astype(np.float32)})
    tpl = pd.read_parquet(config.SUBMIT_TEMPLATE, columns=["MVT_ID_mvt"])
    assert len(out) == G.N_ROWS and np.array_equal(out.MVT_ID_mvt.to_numpy(), tpl.MVT_ID_mvt.to_numpy())
    assert np.isfinite(yeni).all() and (yeni >= 60).all()
    name = G.next_name()
    path = G.OUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    h = G.md5(path)
    rep = [f"# Gönderim denetimi — {name}", "", f"- Dosya: `{path}`", f"- MD5: `{h}`",
           f"- Kaynak: `{kaynak}` (MD5 {G.md5(Path(kaynak))}) + günlük kalkış öncesi gecikme kısıtı (scripts/g_gunluk.py; "
           f"düzeltilen B havalimanları: {','.join(b_apt) or 'yok'}; hariç: {','.join(haric) or 'yok'}) + LOBT penceresi",
           f"- Not: {note}", "", f"- Değişim: RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.2f} sn; LOBT kırpılan {int(km.sum())}"]
    (config.LOG_DIR / f"gonderim_{name}.md").write_text("\n".join(rep), encoding="utf-8")
    pd.DataFrame([{"tarih": time.strftime("%Y-%m-%d %H:%M"), "dosya": path.name, "md5": h, "aday": f"gunluk({Path(kaynak).name})",
                   "yeni_pay": 1.0, "lb_skor": "", "not": note}]).to_csv(G.DEFTER, mode="a", header=False, index=False, encoding="utf-8")
    print("\n".join(rep[:5]))


if __name__ == "__main__":
    main()
