"""Model B — tüm satırlarla (Model A + B) eğitilen adaylar, 4 çeyrekli CV; değerlendirme yalnızca NM'siz satırlarda.

29.09: NM'ye bağlı sütunlar çıkarılınca kalan özellikler 2,08 milyon eğitim satırının hepsinde var.
Katlar b_model_cv.py ile aynı (V.quarter_folds, NM'siz satırların ayı); her katta eğitim = diğer 9 ayın TÜM satırları.
Bileşenler src/model_b.py fit_components_tum; adaylar combine_tum ile (p_yt tüm satırlardan ya da B'den; ± LIRF dışı
3.500 tavan). Taban: b_model_cv.py'nin OOF'u (data/preds/BM_cv_oof.parquet: v126_B ve B satırlı adaylar), aynı satırlar.
Çıktı: kutuk/b_model_cv_tum.md, data/preds/BM_cv_tum_oof.parquet.
Varyantlar: python scripts/b_model_cv_tum.py [agirlik=W] [huber=D] [ek=1] → B satırı ağırlığı W, μ için Huber δ, ek fiziksel sütunlar (EK_TUM); çıktı adlarına
_w<W>_h<D> eki (ör. BM_cv_tum_w10_oof.parquet).
durum=kom,gs,pc,yte (05.10 akşam): Model A'ya LB'de kazandıran "durum" özellik grupları
(komşu kalkış aralıkları, EUROCONTROL günlük serileri, pist konfigürasyonu, yankı geçmişi) tüm satırlı modele eklenir;
dördü de B satırlarını %100 kapsıyor (komşuların tahmini taksisi Model A tahmini gerektirdiği için yok). Ek: _d<gruplar>.
tohum=N (07.10 gece): μ / rejim / yt modelleri N tohumla (42, 43, …) eğitilip ortalanır. Ek: _t<N>.
mu_cb=1 (08.10): μ için ayrıca CatBoost (bilesen_mu_cb). Ek: _cb.
yt_cb=1 (08.10): yer tutucu olasılığı için ayrıca CatBoost (bilesen_yt_tum_cb). Ek: _ytcb.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, model_b as MB, validation as V  # noqa: E402
import b_model_cv as BC  # noqa: E402  (aynı klasördeki betik: summary, md_table)

CANDIDATES = {
    "tum_mu": dict(yt="yt_b"),
    "tum_mu_yt": dict(yt="yt_tum"),
    "tum_mu_yt_tavan3500": dict(yt="yt_tum", cap_non_lirf=3500.0),
}
MISS_GAP = 0.3
DURUM = {"kom": ("komsu_oz", ["KOM_PP_2", "KOM_PP_3", "KOM_PP_5", "KOM_PP_10", "KOM_PS_2", "KOM_PS_3", "KOM_PS_5",
                              "KOM_PS_10", "KOM_AP_5", "KOM_AP_10", "KOM_AS_5", "KOM_AS_10"]),
         "gs": ("gunluk_oz", ["GS_REG_PAY", "GS_GEC_PAY", "GS_ERK_PAY", "GS_ONCE_DK", "GS_ATC_DK"]),
         "pc": ("pist_oz", ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_KUYRUK_W", "PC_AGIR_N",
                            "PC_SPAN20", "PC_CFG_TE"]),
         "yte": ("yanki_te_oz", ["YTE_UCUS", "YTE_HY", "YTE_VARIS", "YTE_PARK", "YTE_N_UCUS"]),
         # 07.10 gece (scripts/park_b_ozellik.py): park yeri dizisi kalkış anına göre ve komşu NM'li kalkışların tahmini taksisi
         "park": ("park_b_oz", ["PB_ON1", "PB_ON1_HY", "PB_ON1_TIP", "PB_ON1_GEC", "PB_ON2", "PB_AYNI", "PB_SON1", "PD_ON1",
                                "PD_SON1"]),
         "ktah": ("park_b_oz", ["PB_KT15", "PB_KT15_N", "PB_KT15_P", "PB_KT15_P_N"]),
         # 07.10 gece: kalkış anına dayalı (NM off-block'a değil) yüzey ve hava grupları; NM_YAPAY* / NMT_YAPAY B'de anlamsız
         "yuzey": ("yuzey_oz", ["NMT_C15_G", "NMT_F15_G", "NMT_PIST_C15_G", "ARR_TI_W30_MED", "ARR_TI_W30_N", "ARR_TI_W30_MAX",
                                "ARR_TI_W15_MED", "ARR_TI_F15_MED"]),
         "wx": ("wx_oz", ["WX_TSY_0", "WX_TSY_1", "WX_TS_0", "WX_TS_1", "WX_FZ_0", "WX_FZ_1", "WX_SN_0", "WX_SN_1", "WX_FG_0",
                          "WX_RAY_0", "WX_TS_S1", "WX_SN_3S", "WX_SPREAD", "WX_GORUS", "WX_RUZGAR", "WX_HAMLE"]),
         "varis": ("varis_wx_oz", ["AR_APT_15", "AR_PIST_15", "AR_PIST_PAY_60"]),
         # 09.10: aksama günü (NM'siz kalkış yoğunluğu; etiketsiz, scripts/aksama_ozellik.py)
         "ak": ("aksama_oz", ["AK_B_GUN", "AK_B_KAT", "AK_B_PAY", "AK_B_60", "AK_B_PAY_60"])}


def durum_ekle(df, gruplar):
    """df'ye seçilen durum gruplarının sütunlarını ekler (MVT_ID_mvt ile); (df, eklenen sütunlar)."""
    eklenen = []
    for g in gruplar:
        dosya, cols = DURUM[g]
        x = pd.read_parquet(config.DATA_DIR / f"{dosya}.parquet", columns=["MVT_ID_mvt"] + cols)
        df = df.merge(x, on="MVT_ID_mvt", how="left", validate="many_to_one")
        eklenen += cols
    return df, eklenen


def load_all(feats):
    base = ["MVT_ID_mvt", "MVT_MONTH", "MVT_DAY", "ADEP_mvt", MB.TARGET_B, MB.DELAY, "NM_TAXI_DIFF_SEC"]
    cols = list(dict.fromkeys(base + [f for f in feats if f != "IS_B"]))
    df = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=cols)
    df = df[df[MB.TARGET_B].notna()].reset_index(drop=True)
    df["IS_B"] = df["NM_TAXI_DIFF_SEC"].isna().astype(np.float32)
    return df


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    w_b, huber = float(opt.get("agirlik", 1)), (float(opt["huber"]) if "huber" in opt else None)
    gruplar = [g for g in opt.get("durum", "").split(",") if g]
    ek = (f"_w{opt['agirlik']}" if "agirlik" in opt else "") + (f"_h{opt['huber']}" if "huber" in opt else "") + (
        "_ek" if opt.get("ek") == "1" else "") + (f"_d{''.join(gruplar)}" if gruplar else "") + (
        f"_t{opt['tohum']}" if "tohum" in opt else "") + ("_cb" if opt.get("mu_cb") == "1" else "") + (
        "_ytcb" if opt.get("yt_cb") == "1" else "") + ("_p24cb" if opt.get("p24_cb") == "1" else "")
    tohumlar = [config.SEED + i for i in range(int(opt.get("tohum", 1)))]
    feats = MB.features_tum(pq.read_schema(config.DEP_TRAIN).names, ek=opt.get("ek") == "1")
    df = load_all(feats)
    if gruplar:
        df, dcols = durum_ekle(df, gruplar)
        feats = feats + dcols
        print(f"durum grupları {gruplar}: {len(dcols)} sütun; B satırlarında boş oranı "
              f"{df.loc[df['IS_B'] == 1, dcols].isna().mean().max():.3f} (en kötü sütun)", flush=True)
    isb = df["IS_B"].to_numpy() == 1
    num = [f for f in feats if f not in MB.CATS_TUM and f != "IS_B"]
    miss = pd.DataFrame({"A": df.loc[~isb, num].isna().mean(), "B": df.loc[isb, num].isna().mean()})
    gap = miss[(miss["A"] - miss["B"]).abs() > MISS_GAP]
    print(f"A ile B arasında eksik oranı farkı > {MISS_GAP} olan (atılan) özellikler:\n{gap.round(3).to_string()}")
    feats = [f for f in feats if f not in gap.index]
    print(f"satır {len(df):,} (B {isb.sum():,}) | özellik {len(feats)} | {time.time() - t0:.0f} sn", flush=True)

    dfb = df[isb].reset_index(drop=True)
    base = pd.read_parquet(config.DATA_DIR / "preds" / "BM_cv_oof.parquet")
    dfb = dfb.merge(base.drop(columns="y"), on="MVT_ID_mvt", how="left", validate="one_to_one")
    ref = [c for c in base.columns if c not in ("MVT_ID_mvt", "y")]
    assert dfb[ref].notna().all().all(), "taban OOF'u eksik"

    oof = {k: np.full(len(dfb), np.nan) for k in CANDIDATES}
    comps = {}
    for q, _, va_m in V.quarter_folds(dfb["MVT_MONTH"]):
        tq = time.time()
        tr = df[~df["MVT_MONTH"].isin(V.QUARTERS[q])].reset_index(drop=True)
        va = dfb[va_m].reset_index(drop=True)
        c = MB.fit_components_tum(tr, va, feats, w_b=w_b, mu_huber=huber, tohumlar=tohumlar, mu_cb=opt.get("mu_cb") == "1",
                                  yt_cb=opt.get("yt_cb") == "1", p24_cb=opt.get("p24_cb") == "1")
        for k, v in c.items():
            comps.setdefault(k, np.full(len(dfb), np.nan))[va_m] = v
        for name, kw in CANDIDATES.items():
            oof[name][va_m] = MB.combine_tum(c, va["ADEP_mvt"], **kw)
        print(f"  {q}: eğitim {len(tr):,} | doğrulama {va_m.sum():,} | {time.time() - tq:.0f} sn", flush=True)

    rows = {k: BC.summary(dfb, dfb[k].to_numpy(float)) for k in ref}
    rows.update({k: BC.summary(dfb, p) for k, p in oof.items()})
    tab = pd.DataFrame(rows).T
    print(tab.round(1).to_string())
    md = [f"# Model B — tüm satırlarla eğitim, 4 çeyrekli CV{ek}", "",
          f"Eğitim: her katta diğer 9 ayın tüm satırları ({len(df):,}); değerlendirme {len(dfb):,} NM'siz satırda. "
          f"Özellik {len(feats)}. RMSE (sn), taksi 60 sn alt sınırlı. İlk satırlar b_model_cv.py tabanları.", "",
          BC.md_table(tab.round(1))]
    (config.LOG_DIR / f"b_model_cv_tum{ek}.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame({"MVT_ID_mvt": dfb["MVT_ID_mvt"].to_numpy(), **oof, **{f"bilesen_{k}": v for k, v in comps.items()}}
                 ).to_parquet(config.DATA_DIR / "preds" / f"BM_cv_tum{ek}_oof.parquet", index=False)
    V.log_experiment({"ad": f"BM_ceyrek_cv_tum{ek}", "tur": "Model B", "ozellik_sayisi": len(feats), "w_b": w_b,
                      "huber": huber,
                      "atilan_ozellikler": gap.index.tolist(), "sonuc": tab.round(3).to_dict(orient="index"),
                      "sure_sn": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
