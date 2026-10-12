"""Kuyruk / zaman penceresi özellikleri (30.09): her kalkış için NM off-block (AOBT_3) ile kalkış (MVT) arasındaki trafik.

Neden: Model A'nın kuyruk özellikleri (PRC_2 traffic_features.py) kalkış anında ölçülüyor: DEP_QUEUE_N, o an yolda olan
kalkışların sayısı. Uçağın kendi taksi penceresindeki trafik ise hiç yok. Pist dakikada ~1 kalkış verdiği için
"NM off-block ile kalkış arasında aynı pistten kaç uçak kalktı" taksi süresinin doğrudan ölçüsü. Tüm kalkış anları ve
NM off-block'ları testte de biliniyor (etiket kullanılmaz; BLOCK_TIME yalnızca varışlarda, taksi-in için okunmaz).

Özellikler (NM off-block O, kalkış T; havalimanı a, pist r):
  KQ_TO_PIST_PEN / KQ_TO_APT_PEN : (O, T) aralığında aynı pistten / havalimanından kalkan diğer uçak sayısı
  KQ_VARIS_PEN                   : (O, T) aralığında aynı havalimanına inen uçak sayısı
  KQ_ONDE_PIST / KQ_ONDE_APT     : O anında sistemde olan (O_j < O < T_j) diğer kalkışlar (NM zamanlarıyla önündeki kuyruk)
  KQ_ITME_20                     : O ± 10 dk içindeki diğer NM off-block sayısı (itme yoğunluğu)
  KQ_ONCEKI_PIST / KQ_SONRAKI_PIST : aynı pistte önceki / sonraki kalkışa süre (sn)
  KQ_PIST_HIZ                    : KQ_TO_PIST_PEN / (NM taksi dk)   KQ_SERBEST : NM taksi − 90 · KQ_TO_PIST_PEN
Çıktı: data/kuyruk_oz.parquet (MVT_ID_mvt + özellikler; 2025'in 12 ayı ve 2026 testi).
Kullanım: python scripts/kuyruk_ozellik.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

RAW = config.PRC2_DIR / "data"
CIKTI = config.DATA_DIR / "kuyruk_oz.parquet"
KOL = ["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "AOBT_3_flt"]


def ep(s):
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def say_aralik(sirali, alt, ust):
    """sirali dizide alt < x < ust olan eleman sayısı (alt/ust NaN ise NaN)."""
    n = np.searchsorted(sirali, ust, "left") - np.searchsorted(sirali, alt, "right")
    return np.where(np.isfinite(alt) & np.isfinite(ust), np.maximum(n, 0), np.nan)


def grup_ozellik(dep, arr_t):
    """Tek havalimanının kalkışları (dep: T, O, pist) ve varış anları için özellikler; dep sırasıyla."""
    T, O = dep["T"].to_numpy(float), dep["O"].to_numpy(float)
    out = {}
    Ts = np.sort(T)
    out["KQ_TO_APT_PEN"] = say_aralik(Ts, O, T)
    out["KQ_VARIS_PEN"] = say_aralik(np.sort(arr_t), O, T)
    nm = np.isfinite(O)
    Os, Tn = np.sort(O[nm]), np.sort(T[nm])
    # O anında sistemde olan: O_j < O ve T_j > O  ≈ #(O_j < O) − #(T_j ≤ O)  (NM'li kalkışlar arasında; kendisi hariç)
    onde = np.searchsorted(Os, O, "left") - np.searchsorted(Tn, O, "right")
    out["KQ_ONDE_APT"] = np.where(nm, np.maximum(onde, 0), np.nan)
    out["KQ_ITME_20"] = np.where(nm, np.searchsorted(Os, O + 600, "right") - np.searchsorted(Os, O - 600, "left") - 1, np.nan)
    for c in ("KQ_TO_PIST_PEN", "KQ_ONDE_PIST", "KQ_ONCEKI_PIST", "KQ_SONRAKI_PIST"):
        out[c] = np.full(len(dep), np.nan)
    for _, idx in dep.groupby("RUNWAY_mvt", dropna=False).indices.items():
        t, o = T[idx], O[idx]
        ts = np.sort(t)
        out["KQ_TO_PIST_PEN"][idx] = say_aralik(ts, o, t)
        m = np.isfinite(o)
        os_, tn = np.sort(o[m]), np.sort(t[m])
        out["KQ_ONDE_PIST"][idx] = np.where(m, np.maximum(np.searchsorted(os_, o, "left") - np.searchsorted(tn, o, "right"), 0), np.nan)
        i = np.searchsorted(ts, t, "left")  # t'nin sıralı dizideki yeri (eşitlerde ilki)
        onceki = np.where(i > 0, ts[np.clip(i - 1, 0, None)], np.nan)
        j = np.searchsorted(ts, t, "right")
        sonraki = np.where(j < len(ts), ts[np.clip(j, None, len(ts) - 1)], np.nan)
        out["KQ_ONCEKI_PIST"][idx] = t - onceki
        out["KQ_SONRAKI_PIST"][idx] = sonraki - t
    nm_dk = (T - O) / 60.0
    out["KQ_PIST_HIZ"] = np.where(nm_dk > 0, out["KQ_TO_PIST_PEN"] / np.maximum(nm_dk, 1.0), np.nan)
    out["KQ_SERBEST"] = (T - O) - 90.0 * out["KQ_TO_PIST_PEN"]
    return pd.DataFrame(out, index=dep.index)


def blok(df):
    """Bir veri bloğu (2025'in tamamı ya da 2026 testi): havalimanı başına özellik."""
    df = df.copy()
    df["T"] = ep(df["MVT_TIME_UTC_mvt"])
    df["O"] = ep(df["AOBT_3_flt"])
    dep = df[df["PHASE_mvt"].eq("DEP")]
    arr = df[df["PHASE_mvt"].eq("ARR")]
    parca = []
    for apt, d in dep.groupby("ADEP_mvt"):
        a = arr.loc[arr["ADES_mvt"].eq(apt), "T"].to_numpy(float)
        parca.append(pd.concat([d[["MVT_ID_mvt"]], grup_ozellik(d, a)], axis=1))
    return pd.concat(parca)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                           "training_2025-12-01_2026-01-01.parquet"), columns=KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=KOL)
    out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    out.to_parquet(CIKTI, index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn → {CIKTI}")
    print(out.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
