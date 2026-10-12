import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src import adsb as A, config
d = A.d_dw_split("split_1")
d["taban"] = A.taksi(d.NM_TAXI_DIFF_SEC, d.ofset_taban); d["yt"] = d.NM_TAXI_DIFF_SEC - d.y
r = pd.read_parquet("data/preds/ADSB_duz_cv_opdi_satir.parquet")
d = d.merge(r, on="MVT_ID_mvt", how="left")
d["p"] = np.maximum(60, d.taban + d.rhat.fillna(0))
k = ["MVT_ID_mvt", "SCHED_DIFF_SEC", "LOBT_DIFF_SEC", "IOBT_DIFF_SEC", "EOBT_DIFF_SEC", "AOBT_DIFF_SEC"]
d = d.merge(pd.read_parquet(config.DEP_TRAIN, columns=k), on="MVT_ID_mvt", how="left")
y = d.yt
def yak(c): return (y + d[c]).abs() <= 60  # X_DIFF = X − MVT (PRC_2/src/features.py:127) → kopya: y = −X_DIFF
d["tur"] = np.select([y > 40000, yak("SCHED_DIFF_SEC"), yak("LOBT_DIFF_SEC"), yak("IOBT_DIFF_SEC"), yak("EOBT_DIFF_SEC"), y > 2400],
                     ["T24 (24 saat)", "P tarife kopyası", "LOBT kopyası", "IOBT kopyası", "EOBT kopyası", "uzun gerçek (>40 dk)"], "normal")
d["e2"] = (d.p - y) ** 2; d["e"] = d.p - y
top = d.e2.quantile(0.99)
g = d.groupby("tur").agg(n=("e2", "size"), sse_pay=("e2", "sum"), ort_hata=("e", "mean"), rmse=("e2", lambda s: np.sqrt(s.mean())),
                         ust1=("e2", lambda s: (s >= top).sum()))
g["sse_pay"] = (100 * g.sse_pay / d.e2.sum()).round(1); g["ust1_pay"] = (100 * g.ust1 / (d.e2 >= top).sum()).round(1)
print(f"Oca/Tem 2025 Model A satırı {len(d):,}, RMSE {np.sqrt(d.e2.mean()):.1f}; en kötü %1'in SSE payı %{100 * d.e2[d.e2 >= top].sum() / d.e2.sum():.1f}")
print(g.sort_values("sse_pay", ascending=False).round(1).to_string())
print("\nnormal + uzun satırlar, havalimanı (SSE payı %):")
m = d.tur.isin(["normal", "uzun gerçek (>40 dk)"])
print((100 * d[m].groupby(["ADEP_mvt", "tur"]).e2.sum().unstack() / d.e2.sum()).round(1).to_string())
print("\nkopya türleri, havalimanı (SSE payı %):")
m = d.tur.str.contains("kopya|T24")
print((100 * d[m].groupby(["ADEP_mvt", "tur"]).e2.sum().unstack() / d.e2.sum()).round(1).fillna(0).to_string())
print("\n===== LIRF uzun satırlar")
o = A.ozellikler()[["MVT_ID_mvt", "ADSB_ESLESME", "ADSB_HAREKET", "ADSB_ILK", "ADSB_PARK_U", "ADSB_ILK_GS"]]
L = d[(d.ADEP_mvt == "LIRF") & (d.tur == "uzun gerçek (>40 dk)")].merge(o, on="MVT_ID_mvt", how="left")
print(f"satır {len(L)}; y çeyrekleri {L.yt.quantile([.1,.25,.5,.75,.9]).round(0).tolist()}; tahmin medyanı {L.p.median():.0f}")
print(f"gecikme (MVT − tarife) çeyrekleri {(-L.SCHED_DIFF_SEC).quantile([.1,.25,.5,.75,.9]).round(0).tolist()}")
for c in ["SCHED_DIFF_SEC", "LOBT_DIFF_SEC", "EOBT_DIFF_SEC", "IOBT_DIFF_SEC", "AOBT_DIFF_SEC"]:
    f = L.yt + L[c]
    print(f"  y − {c[:4]}: |fark| ≤ 300: %{100 * (f.abs() <= 300).mean():.0f}, ≤ 900: %{100 * (f.abs() <= 900).mean():.0f}, medyan {f.median():.0f}")
e = L.ADSB_ESLESME.fillna(0) > 0
print(f"ADS-B eşleşmeli %{100 * e.mean():.0f}; bunlarda ADS-B hareket medyanı {L.ADSB_HAREKET[e].median():.0f} sn (gerçek medyan {L.yt[e].median():.0f}); "
      f"hareket ≥ 1.800 olan %{100 * (L.ADSB_HAREKET[e] >= 1800).mean():.0f}; park yerinde görülen (ilk GS<3) %{100 * (L.ADSB_ILK_GS[e] < 3).mean():.0f}")
L["saat"] = pd.to_datetime(L.MVT_ID_mvt, errors="coerce") if False else 0
print(L.groupby(L.MVT_MONTH).size().to_dict())
print("\n===== saat dilimi kaymalı tarife kopyası: y − (MVT − tarife) = tarife − BLOCK")
kf = y + d.SCHED_DIFF_SEC
for s in (-7200, -3600, 3600, 7200, 10800):
    m = (kf - s).abs() <= 120
    if m.sum():
        print(f"tarife − BLOCK ≈ {s:+d} (±120 sn): {int(m.sum()):,} satır, SSE payı %{100 * d.e2[m].sum() / d.e2.sum():.1f}, ort hata {d.e[m].mean():.0f} | "
              + ", ".join(f"{a} {n}" for a, n in d[m].groupby(['ADEP_mvt', 'MVT_MONTH']).size().items()))
h = np.histogram(kf[(d.ADEP_mvt == "LIRF") & (d.tur == "uzun gerçek (>40 dk)")], bins=[-1e9, 0, 3000, 3480, 3720, 6000, 7080, 7320, 1e9])
print("LIRF uzun satırlarda tarife − BLOCK dağılımı:", dict(zip([f"{a:.0f}..{b:.0f}" for a, b in zip(h[1][:-1], h[1][1:])], h[0].tolist())))
print("\n===== Uzun satırlar: model seziyor mu? (havalimanı başına)")
from sklearn.metrics import roc_auc_score
for a in ["LIRF", "LFPG", "LTFM", "EGLL"]:
    m = d.ADEP_mvt == a; u = (y[m] > 2400).astype(int)
    pa = d.p[m]
    print(f"{a}: uzun %{100 * u.mean():.1f} | AUC(tahmin) {roc_auc_score(u, pa):.3f} | uzun satırda tahmin < 1.200: %{100 * (pa[u == 1] < 1200).mean():.0f}, "
          f"tahmin medyanı {pa[u == 1].median():.0f} / gerçek {y[m][u == 1].median():.0f} | uzun satırların SSE payı %{100 * d.e2[m][u == 1].sum() / d.e2.sum():.1f}")
print("\n===== Tahmin düzeyine göre kalibrasyon (ortalama gerçek − ortalama tahmin)")
bins = [0, 600, 900, 1200, 1500, 1800, 2400, 3000, 4000, 1e9]
d["kova"] = pd.cut(d.p, bins)
t = d.groupby(["kova", "MVT_MONTH"], observed=True).apply(lambda g: pd.Series({"n": len(g), "fark": (g.yt - g.p).mean()})).unstack("MVT_MONTH")
print(t.round(0).to_string())
# ay dışı sınama: bir ayda kova başına ortalama fark öğren, öbür ayda uygula (yalnız p ≥ 1.200 kovaları)
def uygula(tr, te):
    f = d[tr].groupby("kova", observed=True).apply(lambda g: (g.yt - g.p).mean())
    adj = d.loc[te, "kova"].map(f).astype(float).fillna(0).to_numpy()
    adj = np.where(d.loc[te, "p"].to_numpy() >= 1200, adj, 0.0)
    p2 = np.maximum(60, d.loc[te, "p"].to_numpy() + adj)
    return np.sum((p2 - y[te]) ** 2) - np.sum((d.loc[te, "p"] - y[te]) ** 2)
for a, b in ((1, 7), (7, 1)):
    ds = uygula(d.MVT_MONTH == a, d.MVT_MONTH == b)
    n = (d.MVT_MONTH == b).sum(); base = np.sqrt(d.e2[d.MVT_MONTH == b].mean())
    print(f"{a} → {b}: ΔSSE {ds / 1e6:+.1f} M, RMSE {base:.2f} → {np.sqrt(base ** 2 + ds / n):.2f}")
print("\n===== Gecikmeli uçuşlarda kopya rejimleri (y > 1.800 sn)")
u = y > 1800
def yak2(c, tol): return (y + d[c]).abs() <= tol
sinif = np.select([u & yak2("SCHED_DIFF_SEC", 60), u & (yak2("LOBT_DIFF_SEC", 60) | yak2("EOBT_DIFF_SEC", 60) | yak2("IOBT_DIFF_SEC", 60)),
                   u & (yak2("SCHED_DIFF_SEC", 900) | yak2("LOBT_DIFF_SEC", 900) | yak2("EOBT_DIFF_SEC", 900)), u],
                  ["tarife tam kopya", "LOBT/EOBT/IOBT tam kopya", "planlı saate ±15 dk (yumuşak)", "uzun, planlı saatten uzak"], "kısa (≤ 30 dk)")
d["s2"] = sinif
g = d.groupby("s2").agg(n=("e2", "size"), sse=("e2", "sum"), ort_y=("yt", "mean"), ort_p=("p", "mean"))
g["sse_pay"] = (100 * g.sse / d.e2.sum()).round(1); print(g.drop(columns="sse").round(0).to_string())
print((100 * d[d.s2 != "kısa (≤ 30 dk)"].groupby(["ADEP_mvt", "s2"]).e2.sum().unstack() / d.e2.sum()).round(1).fillna(0).to_string())
print("\n===== Sabit sınır taraması (doğru işaret: BLOCK − X = −X_DIFF − y)")
k = ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "SCHED_DIFF_SEC", "LOBT_DIFF_SEC", "IOBT_DIFF_SEC", "EOBT_DIFF_SEC", "AOBT_DIFF_SEC"]
t = pd.read_parquet(config.DEP_TRAIN, columns=k); yy = t.TAXITIME_SEC_mvt.astype(float)
r = pd.read_parquet(config.DEP_RANK, columns=[c for c in k if c != "TAXITIME_SEC_mvt"]).merge(pd.read_parquet("gonderim/strong-moose_v259.parquet"), on="MVT_ID_mvt")
for c in k[2:]:
    v = (-t[c] - yy).dropna(); lo, hi = v.min(), v.max()
    tv = (-r[c] - r.TAXITIME_SEC_mvt).dropna()
    print(f"{c[:5]}: 2025 BLOCK − X ∈ [{lo:,.0f}, {hi:,.0f}] (n {len(v):,}) | v259 dışarıda {int(((tv < lo) | (tv > hi)).sum())}")
