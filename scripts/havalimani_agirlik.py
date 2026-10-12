"""H1 — Model A harmanında havalimanı ağırlıkları (07.10 gece).

Harman 5 bileşenli (xgb_P, gcb_P4, e1h3_P6, lgbh3_P6, dw_P5), ağırlıklar tek (kutuk/g1_agirliklar_kqgpPe.json). 29.09'da eski
bileşenlerle havalimanı ağırlıkları tutarsızdı. Bugünkü bileşenlerle, düzeltici OOF'u olan 5 ayda
(Oca, Mar, Tem, Eki, Kas) ay dışarıda bırakmalı sınama (ofset uzayında, toplam 1, negatif değil; düzeltici kalıntısı sabit):
  global ağırlığı yeniden seçmek          SSE −%0,000 (mevcut ağırlıklar zaten iyi)
  tahmin büyüklüğüne göre 5 kova           SSE +%0,039 (ret)
  global çarpan p·s / yayılım çarpanı      −%0,02 / −%0,07, Ocak'ta kötü (ret); sabit 0,98 / 1,05: +%0,36 / +%0,41
  havalimanı ağırlığı, λ = 0,3 / 0,5 / 0,7 −%0,102 / −%0,129 / −%0,121 (λ = 0,5: Oca −0,15, Mar −0,30, Tem −0,08,
                                           Eki −0,26, Kas +0,16)
Seçilen: w_apt = 0,5 · w(havalimanı, 5 ay) + 0,5 · w(global, 5 ay). Teste: ofset farkı (yeni − eski) tabana eklenir,
taksi = eski taksi + (yeni taban − eski taban) (düzeltici kalıntısı ve LIRF yankı hedge'i aynen kalır).
Kullanım: python scripts/havalimani_agirlik.py <aday, ör. D_kqgpPe_adsb_40o>  → TEST_<aday>_ha.parquet,
          kutuk/g1_havalimani_kqgpPe_l05.json
Ortam: PRC_HARMAN=kqgpPe PRC_ADSB_OZ=adsb5.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import adsb as A, config  # noqa: E402

LAM = 0.5
import os  # noqa: E402
# PRC_DUZ_CV=parkb (07.10 gece): düzeltici kalıntıları park yeri özellikli düzelticinin (_42o) CV'lerinden
_CV = {"": ("ADSB_duz_cv_opdi_satir", "ADSB_duz_cv_bosluk_taban", "ADSB_duz_cv_opdi_satir_kas"),
       "parkb": ("ADSB_duz_cv_parkb", "ADSB_duz_cv_bosluk_parkb", "ADSB_duz_cv_parkb_kas"),
       # 08.10: düzeltici LightGBM + CatBoost (model=lgb+cb, parkb=1; scripts/zincir_duz_lgbcb.sh)
       "lgbcb": ("ADSB_duz_cv_lgbcb", "ADSB_duz_cv_bosluk_lgbcb", "ADSB_duz_cv_lgbcb_kas"),
       # 08.10: final hattı parkb OOF'larını _f ekiyle yeniden üretir (yeniden üretilebilirlik)
       "parkbf": ("ADSB_duz_cv_parkb_f", "ADSB_duz_cv_bosluk_parkb_f", "ADSB_duz_cv_parkb_kas_f"),
       # 09.10 akşam: parkbsh = parkb + Şubat / Haziran 2025 (düzeltici CV sinama=2,6; scripts/zincir_subhaz_cv.sh) — yalnız sınama
       "parkbsh": ("ADSB_duz_cv_parkb", "ADSB_duz_cv_bosluk_parkb", "ADSB_duz_cv_parkb_kas")}[os.environ.get("PRC_DUZ_CV", "")]
KAYNAK = (("split_1", (1, 7), _CV[0]), ("split_4", (3,), _CV[1]), ("split_5", (10,), _CV[1]), ("split_4", (11,), _CV[2]))
if os.environ.get("PRC_DUZ_CV") == "parkbsh":
    KAYNAK = KAYNAK + (("split_2", (2,), "ADSB_duz_cv_parkb_subhaz"), ("split_3", (6,), "ADSB_duz_cv_parkb_subhaz"))


def veri(ad):
    parca = []
    for sp, aylar, rf in KAYNAK:
        d = pd.read_parquet(A.PRED / f"B1_{sp}.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y"])
        d = d[d.MVT_MONTH.isin(aylar)].copy()
        for a in ad:
            f, col = A.BILESEN[a]
            d = d.merge(pd.read_parquet(A.PRED / f"{f}_{sp}.parquet", columns=["MVT_ID_mvt", col]).rename(columns={col: a}),
                        on="MVT_ID_mvt", how="left")
        d = d.merge(pd.read_parquet(A.PRED / f"{rf}.parquet"), on="MVT_ID_mvt", how="left")
        d["rhat"] = d["rhat"].fillna(0.0)
        parca.append(d)
    return pd.concat(parca, ignore_index=True)


def wfit(O, z, w0):
    f = lambda w: np.sum((O @ w - z) ** 2) / len(z)  # noqa: E731
    r = minimize(f, w0, method="SLSQP", bounds=[(0, 1)] * len(w0), constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return r.x


def agirliklar(d, ad, w0, maske):
    O, z, apt = d[ad].to_numpy(float), (d.NM_TAXI_DIFF_SEC + d.rhat - (d.NM_TAXI_DIFF_SEC - d.y)).to_numpy(float), d.ADEP_mvt.to_numpy()
    wg = wfit(O[maske], z[maske], w0)
    return {a: LAM * wfit(O[maske & (apt == a)], z[maske & (apt == a)], w0) + (1 - LAM) * wg for a in np.unique(apt)}


def uygula(d, ad, W, off_eski):
    O, apt = d[ad].to_numpy(float), d.ADEP_mvt.astype(str).to_numpy()
    off = off_eski.copy()
    for a, w in W.items():
        m = apt == a
        off[m] = O[m] @ w
    return off


def main(aday):
    w = json.loads(A.AGIRLIK.read_text(encoding="utf-8"))
    ad, W0 = w["bilesenler"], np.array(w["agirlik"])
    d = veri(ad)
    NM, r, y, ay = d.NM_TAXI_DIFF_SEC.to_numpy(float), d.rhat.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.MVT_MONTH.to_numpy()
    O = d[ad].to_numpy(float)
    p0 = np.maximum(60.0, np.maximum(60.0, NM - O @ W0) + r)
    p = np.empty_like(p0)
    for m in sorted(np.unique(ay)):
        W = agirliklar(d, ad, W0, ay != m)
        te = ay == m
        p[te] = np.maximum(60.0, np.maximum(60.0, NM[te] - uygula(d[te], ad, W, O[te] @ W0)) + r[te])
        print(f"  ay {m:2d} dışarıda: SSE %{100 * (np.sum((p[te] - y[te]) ** 2) / np.sum((p0[te] - y[te]) ** 2) - 1):+.2f}")
    print(f"  toplam SSE %{100 * (np.sum((p - y) ** 2) / np.sum((p0 - y) ** 2) - 1):+.3f}")
    W = agirliklar(d, ad, W0, np.ones(len(d), bool))
    (config.LOG_DIR / f"g1_havalimani_kqgpPe_l05{('_' + os.environ['PRC_DUZ_CV']) if os.environ.get('PRC_DUZ_CV') else ''}.json").write_text(json.dumps(
        {"bilesenler": ad, "lambda": LAM, "agirlik": {a: [round(float(x), 4) for x in v] for a, v in W.items()},
         "not": "07.10 gece: scripts/havalimani_agirlik.py (5 ay, ay dışarıda SSE −%0,129)"}, indent=1), encoding="utf-8")
    t = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{A.TEST_AD}.parquet")
    c = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{aday}.parquet")
    t = c[["MVT_ID_mvt", "ADEP_mvt"]].merge(t, on="MVT_ID_mvt", how="left", validate="one_to_one")
    O_t = t[ad].to_numpy(float)
    off_eski = O_t @ W0
    off_yeni = uygula(t, ad, W, off_eski)
    NM_t = t.NM_TAXI_DIFF_SEC.to_numpy(float)
    fark = np.maximum(60.0, NM_t - off_yeni) - np.maximum(60.0, NM_t - off_eski)
    c["taksi"] = np.maximum(60.0, c["taksi"].to_numpy(float) + fark)
    c["taksi_taban"] = c["taksi_taban"].to_numpy(float) + fark
    cik = config.DATA_DIR / "preds" / f"TEST_{aday}_ha.parquet"
    c.to_parquet(cik, index=False)
    print(f"test: taban farkı RMS {np.sqrt(np.mean(fark ** 2)):.2f} sn, ort {fark.mean():+.2f}; havalimanı ort: "
          + ", ".join(f"{a} {v:+.1f}" for a, v in pd.Series(fark).groupby(t.ADEP_mvt.to_numpy()).mean().items()))
    print(f"yazıldı: {cik}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
