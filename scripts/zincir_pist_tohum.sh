#!/bin/bash
# 05.10: pistli final yapısı, v215'in LB sonucundan önce hazırlık (boş ekran kartı).
# Pist konfigürasyonlu (pist_oz) ek tohumlar: GCB 7, 123 (→ 4), DW 123, 999 (→ 5), lgbh3 / E1 1, 2, 3 (→ 6). Mevcut 2–3
# tohumlu pistli test dosyaları (*_pist, zincir_y1_pist.sh) ile ortalanır → harman kqgpP (T8 ağırlıkları) → düzeltici
# (ek=_35a) → gönderim (lobt=1) → Y1 havalimanı + pist. ekoz_secim.json eğitim süresince pist_oz'lu, sonra geri döner.
# Kayıt: kutuk/zincir_pist_tohum.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
cp kutuk/ekoz_secim.json kutuk/ekoz_secim_onpist2.json
python - <<'PY'
import json
s = json.load(open("kutuk/ekoz_secim.json", encoding="utf-8"))
s["pist_oz.parquet"] = ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_KUYRUK_W", "PC_AGIR_N", "PC_SPAN20", "PC_CFG_TE"]
json.dump(s, open("kutuk/ekoz_secim.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ekoz_secim:", list(s))
PY
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "başladı $(date +%T)"
(
  until grep -q "zincir bitti" kutuk/zincir_final_hazirlik.log 2>/dev/null; do sleep 30; done
  run pt_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy_pb \
  && run pt_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy_pb
  echo "islemci: $? $(date +%T)" > kutuk/pt_cpu.done
) &
(
  run pt_gcb7 scripts/d_global_cb.py 10000 it12000_p7_kqgpy 7 \
  && run pt_gcb123 scripts/d_global_cb.py 10000 it12000_p123_kqgpy 123 \
  && run pt_dw scripts/d_global_cb.py 9000 dw_lr03_ek5_l2_10_pist_kqgpy 123,999 lr=0.03 depth=9 kat=ek5 \
     cb:grow_policy=Depthwise cb:min_data_in_leaf=50 cb:l2_leaf_reg=10
  echo "ekran: $? $(date +%T)" > kutuk/pt_gpu.done
) &
wait
cat kutuk/pt_cpu.done kutuk/pt_gpu.done
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
cp kutuk/ekoz_secim_onpist2.json kutuk/ekoz_secim.json
grep -q "islemci: 0" kutuk/pt_cpu.done && grep -q "ekran: 0" kutuk/pt_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
T=dw_lr03_ek5_l2_10
run pt_gcb_ort4 scripts/gcb_ortalama.py it12000_ort4_pist_kqgpy it12000_kqgpy_pist it12000_t2026_kqgpy_pist it12000_p7_kqgpy it12000_p123_kqgpy
run pt_dw_ort5 scripts/gcb_ortalama.py ${T}_kqgpy_ort5_pist ${T}_kqgpy_t42_pist ${T}_kqgpy_t2026_pist ${T}_kqgpy_t7_pist ${T}_pist_kqgpy_t123 ${T}_pist_kqgpy_t999
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
for a, b, o in (("TEST_v_lgb_huber1000_3t_kqgpy_pist", "TEST_v_lgb_huber1000_3t_kqgpy_pb", "TEST_v_lgb_huber1000_6t_pist_kqgpy"),
                ("TEST_v_e1_huber1000_3t_kqgpy_pist", "TEST_v_e1_huber1000_3t_kqgpy_pb", "TEST_v_e1_huber1000_6t_pist_kqgpy")):
    x, y = pd.read_parquet(P + a + ".parquet"), pd.read_parquet(P + b + ".parquet")
    assert (x.MVT_ID_mvt.to_numpy() == y.MVT_ID_mvt.to_numpy()).all()
    z = x.copy()
    for c in x.columns:
        if c not in ("MVT_ID_mvt", "NM_TAXI_DIFF_SEC") and np.issubdtype(x[c].dtype, np.floating):
            z[c] = (x[c].to_numpy(float) + y[c].to_numpy(float)) / 2
    z["taksi"] = np.maximum(60.0, z.NM_TAXI_DIFF_SEC.to_numpy(float) - z.harman.to_numpy(float))
    z.to_parquet(P + o + ".parquet", index=False)
    print(o, "korelasyon", round(float(np.corrcoef(x.harman, y.harman)[0, 1]), 5))
PY
export PRC_HARMAN=kqgpP PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpP.json D_kqgpP > kutuk/g_test_D_kqgpP.log 2>&1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_35a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpP_35a.log
python scripts/g_gonderim.py D_kqgpP_adsb_35a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "Pistli final yapısı: v215'in özellikleri (komşu + yankı + günlük + pist konfigürasyonu) + T8 ağırlıkları + çok tohum (lgbh3 / E1 6, GCB 4, DW 5; XGB 2) + LOBT." 2>&1 | grep -E "Dosya|MD5"
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
python -u scripts/g_kalibre.py "$SON" pist=1 "$(basename $SON) + Y1 havalimanı + pist katmanı + LOBT: pistli final yapısı." 2>&1 | tail -n 4
echo "zincir bitti $(date +%T)"
