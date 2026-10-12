"""Test harmanı — G1 ağırlık dosyasındaki bileşenleri 12 aylık test tahminleriyle birleştirir.

Kullanım: python scripts/g_test_harman.py <agirlik_json> <aday>
  ör. python scripts/g_test_harman.py kutuk/g1_agirliklar_alti_huber.json alti_huber
Bileşen adları (g1_harman_genel.py) → 12 aylık test dosyası ve sütunu: MAP. Ağırlığı 0 olan bileşen okunmaz.
Çıktı: data/preds/TEST_<aday>.parquet (MVT_ID_mvt, NM_TAXI_DIFF_SEC, bileşenler, harman, taksi); g_gonderim.py girdisi.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"
MAP = {
    "lgb": ("TEST_tam_lgb_xgb", "lgb"),
    "xgb": ("TEST_tam_lgb_xgb", "xgb"),
    "A2_global_cb:global_cb": ("TEST_gcb", "global_cb"),
    "global_cb": ("TEST_gcb", "global_cb"),
    "E1:karisim": ("TEST_tam_e1_lgb42", "harman"),
    "karisim": ("TEST_tam_e1_lgb42", "harman"),
    "lgb_huber": ("TEST_tam_lgb_huber1000", "harman"),
    "lgbh1": ("TEST_tam_lgb_huber1000", "harman"),
    "gcb3k": ("TEST_gcb", "global_cb"),
    "gcb6k": ("TEST_gcb_it6000", "global_cb"),
    "e1": ("TEST_tam_e1_lgb42", "harman"),
    "e1h": ("TEST_v_e1_huber1000", "harman"),
    "lgbh3": ("TEST_v_lgb_huber1000_3t", "harman"),
    "e1h3": ("TEST_v_e1_huber1000_3t", "harman"),
    "gcb6k_t2026": ("TEST_gcb_it6000_t2026", "global_cb"),
    "gcb6k_ort2": ("TEST_gcb_it6000_ort2", "global_cb"),
    "dw_lr03_ek5_l2_10": ("TEST_gcb_dw_lr03_ek5_l2_10_ort3", "global_cb"),
    # 30.09: kuyruk özellikli bileşenler (kuyruk_24.sh)
    "xgb_kq": ("TEST_v_xgb_kq", "harman"),
    "gcb6k_ort2_kq": ("TEST_gcb_it6000_ort2_kq", "global_cb"),
    "e1h3_kq": ("TEST_v_e1_huber1000_3t_kq", "harman"),
    "lgbh3_kq": ("TEST_v_lgb_huber1000_3t_kq", "harman"),
    "dw_kq": ("TEST_gcb_dw_lr03_ek5_l2_10_kq_ort3", "global_cb"),
    "mlp_kq": ("TEST_nn_mlp_kq", "nn"),
    "tabm_kq": ("TEST_nn_tabm_kq", "nn"),
    "lgb_taksi_kq": ("TEST_v_lgb_taksi_kq", "harman"),
    "xgb_kqg": ("TEST_v_xgb_kqg", "harman"),
    "gcb6k_ort2_kqg": ("TEST_gcb_it6000_ort2_kqg", "global_cb"),
    "e1h3_kqg": ("TEST_v_e1_huber1000_3t_kqg", "harman"),
    "lgbh3_kqg": ("TEST_v_lgb_huber1000_3t_kqg", "harman"),
    "dw_kqg": ("TEST_gcb_dw_lr03_ek5_l2_10_kqg_ort3", "global_cb"),
    "xgb_kqgp": ("TEST_v_xgb_kqgp", "harman"),
    "gcb6k_ort2_kqgp": ("TEST_gcb_it6000_ort2_kqgp", "global_cb"),
    "e1h3_kqgp": ("TEST_v_e1_huber1000_3t_kqgp", "harman"),
    "lgbh3_kqgp": ("TEST_v_lgb_huber1000_3t_kqgp", "harman"),
    "dw_kqgp": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgp_ort3", "global_cb"),
    "xgb_kqgp_1500": ("TEST_v_xgb_kqgp_1500", "harman"),
    "gcb12k_ort2_kqgp": ("TEST_gcb_it12000_ort2_kqgp", "global_cb"),
    "xgb_kqgpy_1500": ("TEST_v_xgb_kqgpy_1500", "harman"), "gcb12k_ort2_kqgpy": ("TEST_gcb_it12000_ort2_kqgpy", "global_cb"),
    "e1h3_kqgpy": ("TEST_v_e1_huber1000_3t_kqgpy", "harman"), "lgbh3_kqgpy": ("TEST_v_lgb_huber1000_3t_kqgpy", "harman"),
    "dw_kqgpy": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3", "global_cb"),
    # 03.10 kqgpT: sıcaklıklı (_kqgpy) tam eğitim test tahminleri; OOF tarafı kqgp (src/adsb.py BILESEN *_T)
    "xgb_T": ("TEST_v_xgb_kqgpy_1500", "harman"), "gcb_T": ("TEST_gcb_it12000_ort2_kqgpy", "global_cb"),
    "e1h3_T": ("TEST_v_e1_huber1000_3t_kqgpy", "harman"), "lgbh3_T": ("TEST_v_lgb_huber1000_3t_kqgpy", "harman"),
    "dw_T": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3", "global_cb"),
    # 05.10: çok tohum (scripts/zincir_tohum.sh): GCB 4 tohum (42, 2026, 7, 123), DW 5 tohum (42, 2026, 7, 123, 999)
    "gcb_T4": ("TEST_gcb_it12000_ort4_kqgpy", "global_cb"), "dw_T5": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort5", "global_cb"),
    "e1h3_T6": ("TEST_v_e1_huber1000_6t_kqgpy", "harman"), "lgbh3_T6": ("TEST_v_lgb_huber1000_6t_kqgpy", "harman"),
    # 05.10: pist konfigürasyonlu çok tohum (scripts/zincir_pist_tohum.sh)
    "xgb_P": ("TEST_v_xgb_kqgpy_1500_pist", "harman"), "gcb_P4": ("TEST_gcb_it12000_ort4_pist_kqgpy", "global_cb"),
    "e1h3_P6": ("TEST_v_e1_huber1000_6t_pist_kqgpy", "harman"), "lgbh3_P6": ("TEST_v_lgb_huber1000_6t_pist_kqgpy", "harman"),
    "dw_P5": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort5_pist", "global_cb"),
    # 06.10: pistli final yapısı, MVT_DAY'siz (scripts/zincir_gunsuz.sh)
    "xgb_G": ("TEST_v_xgb_kqgpy_1500_gunsuz", "harman"), "gcb_G4": ("TEST_gcb_it12000_ort4_gunsuz_kqgpy", "global_cb"),
    "e1h3_G6": ("TEST_v_e1_huber1000_6t_kqgpy_gunsuz", "harman"), "lgbh3_G6": ("TEST_v_lgb_huber1000_6t_kqgpy_gunsuz", "harman"),
    "dw_G5": ("TEST_gcb_dw_lr03_ek5_l2_10_ort5_gunsuz_kqgpy", "global_cb"),
    # 08.10 final aşaması (scripts/final_a_nesil.sh): 1. / 2. nesil (yalnız komşu tahmini özelliklerinin girdisi) ekran kartı
    # bileşenleri tek tohumla (42); dosya adlarında tohum eki yok. Ağırlıklar kqgpxg / kqgpy ile aynı (g1_agirliklar_*_F.json).
    "gcb12k_kqgp_1t": ("TEST_gcb_it12000_kqgp", "global_cb"), "dw_kqgp_1t": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgp", "global_cb"),
    "gcb12k_kqgpy_1t": ("TEST_gcb_it12000_kqgpy", "global_cb"), "dw_kqgpy_1t": ("TEST_gcb_dw_lr03_ek5_l2_10_kqgpy", "global_cb"),
    # 09.10 gece: TabM test tahmini final özellikleriyle (PRC_EKOZ_DOSYA=ekoz_secim_final.json; OOF tarafı NN_tabm_kqgp)
    "tabm_T": ("TEST_nn_tabm_kqgpy", "nn"),
    "tabm_T2": (os.environ.get("PRC_TABM_T2_TEST", "TEST_nn_tabm_kqgpy_e12"), "nn"),  # 10.10: 2 tohum × 12 epok (11.10: PRC_TABM_T2_TEST=TEST_nn_tabm_kqgpy_e12t5 → 5 tohum)
    "tabm_T3": ("TEST_nn_tabm_kqgpy_d1024", "nn"),  # 10.10 akşam: d_block 1024
}

# 09.10: tek tohumlu final bileşenleri, OOF tarafındaki (g1_agirliklar_kqgpxg / kqgpy.json) adlarıyla da yazılır — B5
# (adsb_b_duzeltici5.py) test bileşenlerini PRC_HARMAN'ın OOF ağırlık dosyasındaki adlarla okuyor.
TAKMA_AD = {"gcb12k_kqgp_1t": "gcb12k_ort2_kqgp", "dw_kqgp_1t": "dw_kqgp",
            "gcb12k_kqgpy_1t": "gcb12k_ort2_kqgpy", "dw_kqgpy_1t": "dw_kqgpy"}


def main(weights_path, aday):
    g = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    w = {c: a for c, a in zip(g["bilesenler"], g["agirlik"]) if a > 0}
    missing = [c for c in w if c not in MAP]
    assert not missing, f"12 aylık test karşılığı olmayan bileşen: {missing}"
    base = None
    for c in w:
        f, col = MAP[c]
        b = pd.read_parquet(PRED / f"{f}.parquet", columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC", col])
        b = b.rename(columns={col: c})
        base = b if base is None else base.merge(b.drop(columns="NM_TAXI_DIFF_SEC"), on="MVT_ID_mvt", how="left",
                                                 validate="one_to_one")
    assert len(base) == config.TEST_A_N and base[list(w)].notna().all().all()
    tot = sum(w.values())
    base["harman"] = sum(a * base[c] for c, a in w.items()) / tot
    base["taksi"] = np.maximum(V.TAXI_FLOOR, base["NM_TAXI_DIFF_SEC"] - base["harman"])
    for c, k in TAKMA_AD.items():  # 09.10: OOF adıyla okuyanlar (adsb_b_duzeltici5 → A.bilesenler()) için eş sütun
        if c in base:
            base[k] = base[c]
    base.to_parquet(PRED / f"TEST_{aday}.parquet", index=False)
    print(f"[{aday}] ağırlıklar { {c: round(a / tot, 4) for c, a in w.items()} } | ort. taksi {base['taksi'].mean():.1f}")
    print(base[list(w)].corr().round(3).to_string())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])
