"""LIRF NM'siz (Model B) satırlarında 24 saatlik tarih kayması için olasılıksal karışım (08.10 akşam).

Bulgu (2025, 12 ay):
- 24 saat kayması yalnız Model B'de: Model A'da 2.062.577 satırda 0 (tek ≥ 23 sa satırı yer tutucu, taksi = gecikme).
  LIRF dışı NM'siz satırlarda 3 tane (LFPG ×2, LSZH), gecikmeleri normal (0,3–0,6 sa), ayırt eden alan yok.
- LIRF'te mekanizma: BLOCK'un tarihi tarifeden, saati gerçek park ayrılışından → taksi = 86.400 + gerçek taksi (12–33 dk).
  Yalnız kalkışın yerel tarihi tarife tarihinden sonra olan satırlarda (gecikme > yerel günün dakikası · 60) olabilir.
- LIRF NM'siz, gece yarısını geçen satırlarda sınıf, gecikmeye göre:
  2–6 sa: 24 normal / 62 yer tutucu / 1 kayma; 6–14 sa: 25 yer tutucu / 0 kayma; ≥ 14 sa: 0 normal / 8 yer tutucu / 11 kayma.
  Saat, yoğunluk (DEP_COUNT_60) ve havayolu yer tutucu oranı kaymayı ayırmıyor (küçük örnek).
- Mevcut sistemde açık: b_lirf_te.fit_lirf ve model_b.fit_predict 24 saat bileşenini "gecikme > 40.000 ise gecikme"
  alıyor; yani ≥ 11,1 sa gecikmede kayma seçeneği hiç yok, tahmin = yer tutucu (taksi ≈ gecikme). Ön koşullu satırlar
  (on_kosul_24s) da v126'dan aynı değeri taşıyor.
Ne: LIRF NM'siz, gece yarısını geçen ve gecikmesi ≥ ESIK_SA olan satırlarda
    tahmin = p · (86.400 + τ) + (1 − p) · mevcut,  p = (kayma + 1) / (satır + 2) bant içinde (12 ay), τ = medyan gerçek taksi.
Sınama: ay dışarıda (LOMO) ve çeyrek dışarıda; bant sınırları seçenekleri yan yana. Kesin kural (p = 1) kullanılmaz
(v81, v114 dersleri: LIRF'te kesin kurallar LB'de +12,7 / +16,6).
Kullanım: python scripts/b_lirf_tarih_kaymasi.py cv
          python scripts/b_lirf_tarih_kaymasi.py uretim <kaynak> <ek>  → TEST_B_<kaynak><ek>.parquet (MVT_ID_mvt, taksi)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"
ESIK_SA = [14.0]          # bant sınırları (sa); ilk sınırın altı p = 0
N_TEST = 344_841


def veri(bolum):
    d = MB.load_b(bolum)
    d = d[d["ADEP_mvt"].eq("LIRF")].reset_index(drop=True)
    g = d[MB.DELAY].to_numpy(float)
    d["gec"] = g
    d["gece"] = g > d["LOCAL_MINOFDAY"].to_numpy(float) * 60
    if bolum == "train":
        y = d[MB.TARGET_B].to_numpy(float)
        d["kayma"] = (y >= 80000) & (np.abs(y - g) > 60)
    return d


def bant(g, sinir):
    return np.digitize(g / 3600, sinir)


def oran(tr, sinir):
    """Bant → (p, satır, kayma); yalnız gece yarısını geçen satırlar; Beta(1, 1) öncülü."""
    a = tr[tr["gece"]]
    b = bant(a["gec"].to_numpy(), sinir)
    k = a["kayma"].to_numpy()
    return {i: ((k[b == i].sum() + 1) / ((b == i).sum() + 2), int((b == i).sum()), int(k[b == i].sum()))
            for i in range(1, len(sinir) + 1)}


def uygula(d, mevcut, sinir, p_bant, tau):
    b = bant(d["gec"].to_numpy(), sinir)
    p = np.array([p_bant[i][0] if i > 0 else 0.0 for i in b])
    p[~d["gece"].to_numpy()] = 0.0
    return p * tau + (1 - p) * mevcut, p


def tau_hesap(tr):
    return 86400.0 + float(np.median(tr.loc[tr["kayma"], MB.TARGET_B].to_numpy(float) - 86400))


def cv():
    sys.stdout.reconfigure(encoding="utf-8")
    d = veri("train")
    y, g = d[MB.TARGET_B].to_numpy(float), d["gec"].to_numpy()
    aday = d["gece"].to_numpy() & (g >= 40000)  # mevcut sistemin yer tutucu saydığı bölge (≥ 11,1 sa)
    print(f"LIRF NM'siz {len(d):,} satır; gece yarısını geçen ve gecikmesi ≥ 11,1 sa: {aday.sum()} (kayma {int(d.kayma[aday].sum())})")
    print(f"τ (12 ay) = 86.400 + {tau_hesap(d) - 86400:.0f} sn")
    sse0 = (g - y)[aday] ** 2  # mevcut: tahmin = gecikme (yer tutucu)
    secenek = {"≥14 sa tek bant": [14.0], "≥12 sa tek bant": [12.0], "14–18 / ≥18": [14.0, 18.0], "14 / 18 / 26": [14.0, 18.0, 26.0]}
    for kat, katlar in (("ay dışarıda", [(str(m), d["MVT_MONTH"].eq(m).to_numpy()) for m in range(1, 13)]),
                        ("çeyrek dışarıda", [(q, va) for q, _, va in V.quarter_folds(d["MVT_MONTH"])])):
        print(f"\n== {kat}: aday satırlarda SSE (milyar sn²); fark − iyi; 1 milyar ≈ test benzeri {1e9 / N_TEST:,.0f} MSE")
        print(f"   mevcut (= gecikme): {sse0.sum() / 1e9:.2f}")
        for ad, sinir in secenek.items():
            pred = g.copy()
            for _, va in katlar:
                te = aday & va
                if te.any():
                    pr, _ = uygula(d[te], g[te], sinir, oran(d[~va], sinir), tau_hesap(d[~va]))
                    pred[te] = pr
            s = (pred - y)[aday] ** 2
            kat_fark = {k: ((s - sse0)[(va[aday])]).sum() / 1e9 for k, va in katlar}
            iyi = sum(v < -1e-3 for v in kat_fark.values()); kotu = sum(v > 1e-3 for v in kat_fark.values())
            print(f"   {ad:<16} {s.sum() / 1e9:5.2f} ({(s.sum() - sse0.sum()) / 1e9:+.2f}) | iyi {iyi}, kötü {kotu} | "
                  + " ".join(f"{k}:{v:+.2f}" for k, v in kat_fark.items() if abs(v) > 1e-3))
    print("\nBantlar (12 ay):")
    for ad, sinir in secenek.items():
        print(f"   {ad:<16} " + " | ".join(f"bant {i}: {n} satır, {s} kayma, p {p:.3f}" for i, (p, n, s) in oran(d, sinir).items()))


def uretim(kaynak, ek):
    sys.stdout.reconfigure(encoding="utf-8")
    tr, rk = veri("train"), veri("rank")
    p_bant, tau = oran(tr, ESIK_SA), tau_hesap(tr)
    t = pd.read_parquet(PRED / f"TEST_B_{kaynak}.parquet")
    x = rk[["MVT_ID_mvt", "gec", "gece"]].merge(t[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    assert x["taksi"].notna().all(), "LIRF test satırı kaynakta yok"
    yeni, p = uygula(x, x["taksi"].to_numpy(float), ESIK_SA, p_bant, tau)
    x["yeni"], x["p"] = yeni, p
    o = t.merge(x[["MVT_ID_mvt", "yeni"]], on="MVT_ID_mvt", how="left")
    o["taksi"] = o["yeni"].fillna(o["taksi"]).astype(float)
    o.drop(columns=["yeni"]).to_parquet(PRED / f"TEST_B_{kaynak}{ek}.parquet", index=False)
    print(f"bantlar {ESIK_SA}: " + " | ".join(f"{n} satır, {s} kayma → p {p_:.3f}" for p_, n, s in p_bant.values())
          + f"; τ = 86.400 + {tau - 86400:.0f}")
    m = x["p"] > 0
    print(f"değişen LIRF test satırı: {int(m.sum())}")
    print(x[m].assign(gec_sa=lambda z: z.gec / 3600, fark=lambda z: z.yeni - z.taksi)[["MVT_ID_mvt", "gec_sa", "p", "taksi", "yeni", "fark"]]
          .round(2).to_string(index=False))


if __name__ == "__main__":
    cv() if sys.argv[1] == "cv" else uretim(sys.argv[2], sys.argv[3])
