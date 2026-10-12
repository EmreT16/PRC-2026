"""Model B üretimi — tüm satırlarla (12 ay, Model A + B) eğitilen model; testin LIRF dışı NM'siz satırları.

CV (b_model_cv_tum.py, 29.09): LIRF dışı, 24 saat hariç 20.980 satırda v126_B 774,2 → 699,2 (tum_mu_yt_tavan3500);
dört çeyreğin ve Ocak+Temmuz'un hepsinde, 9 havalimanının 7'sinde daha iyi (EDDF ve LEBL ≈ eşit). LIRF'te v126_B daha iyi
(24 saat ön koşulu satırları), bu yüzden LIRF satırları v126'da kalır.
Çıktı: data/preds/TEST_B_tum.parquet (MVT_ID_mvt, taksi, bileşenler; yalnızca LIRF dışı satırlar) → g_gonderim.py b=tum.
Kullanım: python scripts/b_uretim_tum.py [hepsi] [durum=kom,gs,pc,yte] [ad=<çıktı>] [tohum=N] [mu_cb=1]
mu_cb=1 (08.10): CatBoost μ da eğitilir (bilesen_mu_cb sütunu; taksi sütunu yalnız LightGBM μ ile kalır).
durum (05.10 akşam): b_model_cv_tum.py'deki durum grupları; A–B eksik oranı farkı > 0,3 olan sütunlar (PC_KUYRUK_W,
PC_AGIR_N: AOBT_3'e bağlı) CV'deki gibi atılır. ad verilmezse çıktı TEST_B_tum[_hepsi].parquet (eski davranış).
hepsi: tüm test B satırları (LIRF dahil) → TEST_B_tum_hepsi.parquet; taksi_yt_tum_tavan (LIRF dışı seçimi) ve taksi_yt_b
(LIRF seçimi: B satırlı rejim olasılıkları + tüm satırlı μ; CV'de LIRF ön koşul dışı 3.187 → 3.079) sütunları ve on_kosul.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402
import b_model_cv_tum as BT  # noqa: E402  (aynı klasördeki betik: load_all)

V126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"


def main(hepsi=False, gruplar=(), ad=None, tohum=1, mu_cb=False, cb_tohum=1):
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    feats = MB.features_tum(pq.read_schema(config.DEP_TRAIN).names)
    df = BT.load_all(feats)
    rk = MB.load_b("rank")
    rk["IS_B"] = np.float32(1.0)
    if gruplar:
        df, dcols = BT.durum_ekle(df, gruplar)
        rk, _ = BT.durum_ekle(rk, gruplar)
        isb = df["IS_B"].to_numpy() == 1
        gap = (df.loc[~isb, dcols].isna().mean() - df.loc[isb, dcols].isna().mean()).abs()
        at = gap[gap > BT.MISS_GAP].index.tolist()
        feats = feats + [c for c in dcols if c not in at]
        print(f"durum {list(gruplar)}: +{len(dcols) - len(at)} sütun (atılan {at}); test B boş oranı en çok "
              f"{rk[[c for c in dcols if c not in at]].isna().mean().max():.3f}", flush=True)
    ap = (rk if hepsi else rk[~rk["ADEP_mvt"].eq("LIRF")]).reset_index(drop=True)
    print(f"eğitim {len(df):,} satır (B {int(df['IS_B'].sum()):,}) | test B {len(rk):,}, LIRF dışı {len(ap):,} | "
          f"{time.time() - t0:.0f} sn", flush=True)
    c = MB.fit_components_tum(df, ap, feats, tohumlar=[config.SEED + i for i in range(tohum)], mu_cb=mu_cb,
                              cb_tohumlar=[config.SEED + i for i in range(cb_tohum)])
    taksi = MB.combine_tum(c, ap["ADEP_mvt"], yt="yt_tum", cap_non_lirf=3500.0)
    out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": ap["ADEP_mvt"].to_numpy(), "taksi": taksi,
                        **{f"bilesen_{k}": v for k, v in c.items()}})
    if hepsi:
        lirf = ap["ADEP_mvt"].eq("LIRF").to_numpy()
        out["taksi_yt_tum_tavan"] = taksi
        out["taksi_yt_b"] = MB.combine_tum(c, ap["ADEP_mvt"], yt="yt_b")
        out["taksi"] = np.where(lirf, out["taksi_yt_b"], taksi)
        out["on_kosul"] = MB.on_kosul_24s(ap)
    # 08.10: final testinde v126 gönderimi yok → karşılaştırma sütunu v126 B'nin final karşılığı (scripts/final_b_taban.py)
    ref = (pd.read_parquet(config.DATA_DIR / "preds" / "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "taban"])
           .rename(columns={"taban": "TAXITIME_SEC_mvt"}) if config.FINAL else pd.read_parquet(V126))
    old = out[["MVT_ID_mvt"]].merge(ref, on="MVT_ID_mvt", how="left", validate="one_to_one")
    out["v126"] = old["TAXITIME_SEC_mvt"].to_numpy(float)
    assert out["taksi"].notna().all() and out["v126"].notna().all()
    out.to_parquet(config.DATA_DIR / "preds" / f"TEST_B_{ad or 'tum' + ('_hepsi' if hepsi else '')}.parquet", index=False)
    diff = out["taksi"] - out["v126"]
    print(f"LIRF dışı test B: ort. tahmin {out['taksi'].mean():.1f} (v126 {out['v126'].mean():.1f}) | RMS fark "
          f"{np.sqrt((diff ** 2).mean()):.1f} | 3.500 tavanında {int((out['taksi'] >= 3500).sum())} "
          f"(v126 {int((out['v126'] >= 3500).sum())}) | süre {time.time() - t0:.0f} sn")
    print(out.groupby("ADEP_mvt")[["taksi", "v126"]].mean().round(1).to_string())
    V.log_experiment({"ad": f"TEST_B_tum{'_hepsi' if hepsi else ''}", "tur": "Model B", "satir": len(out), "ozellik_sayisi": len(feats),
                      "ort_tahmin": float(out["taksi"].mean()), "ort_v126": float(out["v126"].mean()),
                      "rms_fark": float(np.sqrt((diff ** 2).mean())), "sure_sn": round(time.time() - t0, 1)})


if __name__ == "__main__":
    arg = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    main("hepsi" in sys.argv[1:], [g for g in arg.get("durum", "").split(",") if g], arg.get("ad"), int(arg.get("tohum", 1)),
         arg.get("mu_cb") == "1", int(arg.get("cb_tohum", 1)))
