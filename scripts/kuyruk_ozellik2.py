"""Kuyruk özellikleri, ikinci grup (30.09): sollama. NM off-block O ve kalkış T ile:
  KQ2_SOLLANAN_APT / _PIST : O_j ∈ (O, T) ve T_j < T — bizden sonra itip bizden önce kalkanlar (bizi bekleten durum)
  KQ2_SOLLAYAN_APT / _PIST : O_j < O ve T_j > T — bizden önce itip bizden sonra kalkanlar (biz onları geçtik)
  KQ2_SIRA_FARK_PIST        : itme sırası − kalkış sırası (aynı pistte, gün içinde değil tüm akışta; kayan fark)
Çıktı: data/kuyruk_oz2.parquet. Kullanım: python scripts/kuyruk_ozellik2.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kuyruk_ozellik as K  # noqa: E402
from src import config  # noqa: E402

PEN = 4 * 3600.0  # sollama araması için en uzun taksi


def sollama(T, O):
    """(sollanan, sollayan) sayıları; yalnızca NM'li satırlar (O sonlu) arasında."""
    n = len(T)
    sollanan, sollayan = np.full(n, np.nan), np.full(n, np.nan)
    m = np.isfinite(O)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return sollanan, sollayan
    o, t = O[idx], T[idx]
    s = np.argsort(o, kind="stable")
    os_, ts_ = o[s], t[s]
    for k in range(len(s)):
        oi, ti = os_[k], ts_[k]
        hi = np.searchsorted(os_, ti, "left")          # O_j < T_i
        a = ts_[k + 1:hi]                              # O_j ∈ (O_i, T_i) (sıralı; eşitler dahil yaklaşık)
        sollanan[idx[s[k]]] = np.count_nonzero(a < ti)
        lo = np.searchsorted(os_, oi - PEN, "left")
        b = ts_[lo:k]                                  # O_j ∈ [O_i − PEN, O_i)
        sollayan[idx[s[k]]] = np.count_nonzero(b > ti)
    return sollanan, sollayan


def grup(dep):
    T, O = dep["T"].to_numpy(float), dep["O"].to_numpy(float)
    out = pd.DataFrame(index=dep.index)
    out["KQ2_SOLLANAN_APT"], out["KQ2_SOLLAYAN_APT"] = sollama(T, O)
    sp, sy, fark = np.full(len(dep), np.nan), np.full(len(dep), np.nan), np.full(len(dep), np.nan)
    for _, ii in dep.groupby("RUNWAY_mvt", dropna=False).indices.items():
        a, b = sollama(T[ii], O[ii])
        sp[ii], sy[ii] = a, b
        m = np.isfinite(O[ii])
        if m.any():
            j = ii[m]
            itme = pd.Series(O[j]).rank(method="first").to_numpy()
            kalkis = pd.Series(T[j]).rank(method="first").to_numpy()
            fark[j] = itme - kalkis
    out["KQ2_SOLLANAN_PIST"], out["KQ2_SOLLAYAN_PIST"], out["KQ2_SIRA_FARK_PIST"] = sp, sy, fark
    return out


def blok(df):
    df = df.copy()
    df["T"], df["O"] = K.ep(df["MVT_TIME_UTC_mvt"]), K.ep(df["AOBT_3_flt"])
    dep = df[df["PHASE_mvt"].eq("DEP")]
    return pd.concat([pd.concat([d[["MVT_ID_mvt"]], grup(d)], axis=1) for _, d in dep.groupby("ADEP_mvt")])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(K.RAW / (f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet" if m < 12 else
                                             "training_2025-12-01_2026-01-01.parquet"), columns=K.KOL) for m in range(1, 13)],
                   ignore_index=True)
    rk = pd.read_parquet(config.RANK_RAW, columns=K.KOL)
    out = pd.concat([blok(tr), blok(rk)], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique
    out.to_parquet(config.DATA_DIR / "kuyruk_oz2.parquet", index=False)
    print(f"{len(out):,} kalkış | {time.time() - t0:.0f} sn")
    print(out.describe().T.round(2).to_string())


if __name__ == "__main__":
    main()
