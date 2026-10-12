"""11.10 (scripts/zincir_lirf_mu_nn.sh yardımcısı): LIRF μ kısmi TabM adaylarının denetimi ve final 8 havalimanlı yerleştirme.
  python scripts/lirf_aday_kiyas.py normal <strong-moose_vN.parquet>   → v361'e göre fark (LIRF B satırları / diğerleri)
  python scripts/lirf_aday_kiyas.py yerlestir                           → (PRC_FINAL=1) _bN85L B dosyaları
  python scripts/lirf_aday_kiyas.py final <gonderim_final/…a<N>.parquet> → a64'e göre fark
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

SEKIZ = ["EDDM", "EGLL", "LEBL", "LEMD", "LFPG", "LTFM", "EHAM", "LSZH"]


def fark(b, a, n):
    assert len(b) == n and b.notna().all() and b.min() >= 60
    r = pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "NM_TAXI_DIFF_SEC"]).set_index("MVT_ID_mvt").loc[b.index]
    d = (b - a.reindex(b.index)).to_numpy()
    lb = (r.ADEP_mvt.eq("LIRF") & r.NM_TAXI_DIFF_SEC.isna()).to_numpy()
    print(f"  değişen {(np.abs(d) > 1e-3).sum():,}; LIRF B satırları ({lb.sum()}) RMS {np.sqrt(np.mean(d[lb] ** 2)):.1f}, ort {d[lb].mean():+.2f}, "
          f"değişen {(np.abs(d[lb]) > 1e-3).sum()}; diğer satırlar maks |fark| {np.abs(d[~lb]).max():.2f}")


def yerlestir():
    P = config.DATA_DIR / "preds"
    d = pd.read_parquet(P / "TEST_B_tum_dpdcb6c5N5.parquet")[["MVT_ID_mvt", "taksi"]].rename(columns={"taksi": "yeni"})
    for ad, kay in (("tum6M_adsb", "tum6M_adsb"), ("tum6M_adsb_lirfte", "tum6M_adsb_lirfte_mu2n_tk")):
        k = pd.read_parquet(P / f"TEST_B_{kay}.parquet")
        if "ADEP_mvt" not in k:
            k = k.merge(pd.read_parquet(P / "TEST_B_tum6M_adsb.parquet", columns=["MVT_ID_mvt", "ADEP_mvt"]), on="MVT_ID_mvt")
        k = k.merge(d, on="MVT_ID_mvt", how="left")
        m = k.ADEP_mvt.isin(SEKIZ).to_numpy()
        assert k.loc[m, "yeni"].notna().all()
        for c in ("taban", "taksi"):
            if c in k:
                k[c] = k[c].astype(float)
                k.loc[m, c] = k.loc[m, "yeni"]
        k.drop(columns=["yeni"]).to_parquet(P / f"TEST_B_{ad}_bN85L.parquet", index=False)
        print(ad, "→ _bN85L:", int(m.sum()), "satır")
    a = pd.read_parquet(P / "TEST_B_tum6M_adsb_lirfte_bN85.parquet")
    b = pd.read_parquet(P / "TEST_B_tum6M_adsb_lirfte_bN85L.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    f = b.taksi.to_numpy(float) - a.taksi.to_numpy(float)
    print(f"lirfte _bN85L − _bN85: değişen {int((np.abs(f) > 1e-6).sum())}, RMS {np.sqrt(np.mean(f ** 2)):.1f}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    kip = sys.argv[1]
    if kip == "yerlestir":
        return yerlestir()
    if kip == "normal":
        b = pd.read_parquet("gonderim/" + sys.argv[2]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        a = pd.read_parquet("gonderim/strong-moose_v361.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        print(f"{sys.argv[2]} − v361:")
        fark(b, a, 344_841)
    else:
        b = pd.read_parquet(sys.argv[2]).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        a = pd.read_parquet("gonderim_final/strong-moose_final_a64.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
        print(f"{sys.argv[2]} − a64:")
        fark(b, a, 670_790)


if __name__ == "__main__":
    main()
