"""10.10: Model B'nin normal rejim bileşeni μ için TabM (ağaç dışı aile).

Gerekçe: Model A'da TabM harmana katıldığında düzeltici CV 196,14 → 194,69, LB v341 − v330 = −0,365. Model B'nin μ'sü şu an
0,5 · LightGBM + 0,25 · CatBoost + 0,25 · CatBoost (src/model_b.py fit_components_tum; v316 melezi); B satırları hatanın ~%20'si.
Eğitim: b_model_cv_tum.py ile aynı çerçeve (tüm satırlar, A + B, IS_B; durum grupları), yalnız normal rejim (regimes = 0),
hedef taksi süresi; TabM (scripts/nn_bilesen.py genel_tahmin: k = 8, fp16, Huber δ 1.000), kategorikler MB.CATS_TUM.
Özellik süzgeci CV ve üretimde aynı: A ile B satırları arasında eksik oranı farkı > 0,3 olan sayısal sütunlar atılır.
Kullanım:
  python scripts/b_mu_nn.py cv durum=park,ktah,kom,gs,pc,yte epok=12 tohum=42   → data/preds/BM_cv_mu_nn<ek>_oof.parquet
  python scripts/b_mu_nn.py uretim [hepsi] durum=… epok=12 tohum=42,7           → data/preds/TEST_B_mu_nn<ek>[_hepsi].parquet
  (PRC_FINAL=1 ile final testi; çıktı data_final/preds)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402
import b_model_cv_tum as BT  # noqa: E402
import nn_bilesen as NN  # noqa: E402


def cerceve(gruplar):
    feats = MB.features_tum(pq.read_schema(config.DEP_TRAIN).names)
    df = BT.load_all(feats)
    if gruplar:
        df, dcols = BT.durum_ekle(df, gruplar)
        feats = feats + dcols
    isb = df["IS_B"].to_numpy() == 1
    num = [f for f in feats if f not in MB.CATS_TUM and f != "IS_B"]
    gap = (df.loc[~isb, num].isna().mean() - df.loc[isb, num].isna().mean()).abs()
    at = gap[gap > BT.MISS_GAP].index.tolist()
    feats = [f for f in feats if f not in at]
    print(f"satır {len(df):,} (B {isb.sum():,}) | özellik {len(feats)} | atılan {at}", flush=True)
    return df, feats


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    kip = sys.argv[1]
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    opt.setdefault("model", "tabm"); opt.setdefault("tabm_k", "8"); opt.setdefault("amp", "1")
    gruplar = [g for g in opt.get("durum", "").split(",") if g]
    ek = (f"_d{''.join(gruplar)}" if gruplar else "") + f"_e{opt.get('epok', 6)}" + (
        f"_t{len(opt.get('tohum', '42').split(','))}") + (f"_dl{opt['delta']}" if "delta" in opt else "") + (
        f"_bd{opt['tabm_d']}" if "tabm_d" in opt else "") + (  # 11.10: varyant ekleri (yoksa eski adlar)
        f"_s{opt['tohum']}" if kip == "cv" and opt.get("tohum", "42") != "42" else "") + (  # CV'de başka tohum (kontrol)
        f"_bw{opt['b_agirlik']}" if "b_agirlik" in opt else "")
    df, feats = cerceve(gruplar)
    y = df[MB.TARGET_B].to_numpy(float)
    r = MB.regimes(df)
    # 11.10: b_agirlik=… — eğitimde B satırlarının (IS_B; eğitimin ~%1'i) Huber ağırlığı; yoksa ağırlıksız (eski hesap)
    w_b = np.where(df["IS_B"].to_numpy() == 1, float(opt["b_agirlik"]), 1.0) if "b_agirlik" in opt else None
    if kip == "cv":
        isb = df["IS_B"].to_numpy() == 1
        dfb = df[isb].reset_index(drop=True)
        oof = np.full(len(dfb), np.nan)
        for q, _, va_m in V.quarter_folds(dfb["MVT_MONTH"]):
            tq = time.time()
            m = (~df["MVT_MONTH"].isin(V.QUARTERS[q])).to_numpy() & (r == 0)
            oof[va_m] = NN.genel_tahmin(df[m], dfb[va_m], feats, y[m], opt, cats=MB.CATS_TUM,
                                        agirlik=None if w_b is None else w_b[m])
            print(f"  {q}: eğitim {m.sum():,} | doğrulama {va_m.sum():,} | {time.time() - tq:.0f} sn", flush=True)
        out = pd.DataFrame({"MVT_ID_mvt": dfb["MVT_ID_mvt"].to_numpy(), "bilesen_mu_nn": oof})
        out.to_parquet(config.DATA_DIR / "preds" / f"BM_cv_mu_nn{ek}_oof.parquet", index=False)
        print(f"OOF → BM_cv_mu_nn{ek}_oof.parquet | {time.time() - t0:.0f} sn")
        return
    hepsi = "hepsi" in sys.argv[2:]
    rk = MB.load_b("rank")
    rk["IS_B"] = np.float32(1.0)
    if gruplar:
        rk, _ = BT.durum_ekle(rk, gruplar)
    ap = (rk if hepsi else rk[~rk["ADEP_mvt"].eq("LIRF")]).reset_index(drop=True)
    m = r == 0
    p = NN.genel_tahmin(df[m], ap, feats, y[m], opt, cats=MB.CATS_TUM, agirlik=None if w_b is None else w_b[m])
    out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "ADEP_mvt": ap["ADEP_mvt"].to_numpy(), "bilesen_mu_nn": p})
    ad = f"TEST_B_mu_nn{ek}{'_hepsi' if hepsi else ''}.parquet"
    out.to_parquet(config.DATA_DIR / "preds" / ad, index=False)
    print(f"{ad}: {len(out):,} satır | ort μ_nn {p.mean():.1f} | {time.time() - t0:.0f} sn")


if __name__ == "__main__":
    main()
