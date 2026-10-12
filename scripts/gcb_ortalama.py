"""Global CatBoost tohum ortalaması — aynı ayarlarla farklı tohumlarda eğitilen GCB tahminlerinin eşit ağırlıklı ortalaması.

Kullanım: python scripts/gcb_ortalama.py <çıktı_etiketi> <etiket> [<etiket> ...]
  ör. python scripts/gcb_ortalama.py it6000_ort2 it6000 it6000_t2026
Düzenek: data/preds/GCB_<etiket>_<split>.parquet → GCB_<çıktı>_<split>.parquet (g1_harman_genel.py girdisi).
Test: data/preds/TEST_gcb_<etiket>.parquet → TEST_gcb_<çıktı>.parquet (g_test_harman.py girdisi). Bir adımın dosyalarından biri
yoksa o adım atlanır ve bu yazdırılır.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"


def average(paths, keep):
    frames = [pd.read_parquet(p) for p in paths]
    ids = frames[0]["MVT_ID_mvt"].to_numpy()
    for f in frames[1:]:
        assert len(f) == len(ids) and (f["MVT_ID_mvt"].to_numpy() == ids).all(), "satır sırası farklı"
    out = frames[0][keep].copy()
    out["global_cb"] = sum(f["global_cb"].to_numpy(float) for f in frames) / len(frames)
    r = frames[0]["global_cb"].corr(frames[1]["global_cb"])
    return out, r


def main(out_tag, tags):
    for s in V.SPLITS:
        paths = [PRED / f"GCB_{t}_{s}.parquet" for t in tags]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"{s}: düzenek adımı atlandı; eksik: {missing}")
            continue
        out, r = average(paths, ["MVT_ID_mvt"])
        out.to_parquet(PRED / f"GCB_{out_tag}_{s}.parquet", index=False)
        print(f"{s}: {len(out):,} satır | ilk iki tohumun korelasyonu {r:.4f}", flush=True)
    tests = [PRED / f"TEST_gcb_{t}.parquet" for t in tags]
    missing = [p.name for p in tests if not p.exists()]
    if missing:
        print(f"test adımı atlandı; eksik: {missing}")
        return
    out, r = average(tests, ["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"])
    assert len(out) == config.TEST_A_N and out["global_cb"].notna().all()
    out.to_parquet(PRED / f"TEST_gcb_{out_tag}.parquet", index=False)
    print(f"test: {len(out):,} satır | ilk iki tohumun korelasyonu {r:.4f} | ort. ofset {out['global_cb'].mean():.1f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2:])
