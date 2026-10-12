#!/bin/bash
# 04.10 akşam: v206 (yankı geçmişi) LB −0,49 → yankı geçmişi + günlük seriler birlikte (v205 + YTE + GS). v208 − v206 = günlük
# serilerin yankılı tabandaki etkisi. zincir_gun.sh bitince başlar. Geri dönüş: ekoz_secim_gun.json, TEST_*kqgpy*_gun (v207).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -qE "zincir bitti|durdu" kutuk/zincir_gun.log 2>/dev/null; do sleep 120; done
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
for f in data/preds/TEST_*kqgpy*.parquet; do case "$f" in *_sicak.parquet|*_k1.parquet|*_k2.parquet|*_yte.parquet|*_gun.parquet) ;; *) cp "$f" "${f%.parquet}_gun.parquet";; esac; done
cp kutuk/ekoz_secim.json kutuk/ekoz_secim_gun.json
python - <<'PY'
import json
s = json.load(open("kutuk/ekoz_secim_yte.json", encoding="utf-8"))
s["gunluk_oz.parquet"] = ["GS_REG_PAY", "GS_GEC_PAY", "GS_ERK_PAY", "GS_ONCE_DK", "GS_ATC_DK"]
json.dump(s, open("kutuk/ekoz_secim.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ekoz_secim:", list(s))
PY
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "eğitim başladı $(date +%T)"
(
  run d_yg_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy \
  && run d_yg_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy
  echo "islemci: $? $(date +%T)" > kutuk/yg_cpu.done
) &
(
  T=dw_lr03_ek5_l2_10_kqgpy
  run d_yg_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
     cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
  && run yg_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
  && run d_yg_gcb12k scripts/d_global_cb.py 10000 it12000_kqgpy \
  && run d_yg_gcb12k_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 \
  && run yg_gcb12k_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
  && run d_yg_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500
  echo "ekran: $? $(date +%T)" > kutuk/yg_gpu.done
) &
wait
cat kutuk/yg_cpu.done kutuk/yg_gpu.done
grep -q "islemci: 0" kutuk/yg_cpu.done && grep -q "ekran: 0" kutuk/yg_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
export PRC_HARMAN=kqgpT5 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT.json D_kqgpT5 > kutuk/g_test_D_kqgpT5.log 2>&1
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_D_kqgpT3.parquet", columns=["MVT_ID_mvt", "taksi"])
b = pd.read_parquet(P + "TEST_D_kqgpT5.parquet", columns=["MVT_ID_mvt", "taksi"])
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
d = a.merge(b, on="MVT_ID_mvt", suffixes=("_eski", "_yeni")).merge(r, on="MVT_ID_mvt")
d["f"] = d["taksi_yeni"] - d["taksi_eski"]
print("Model A test tabanı, yankı + günlük − v206: RMS", round(float(np.sqrt((d.f ** 2).mean())), 1), "| ort", round(d.f.mean(), 1))
print(d.groupby(["ADEP_mvt", "MVT_MONTH"]).f.mean().unstack().round(1).to_string())
PY
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_28a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT5_28a.log
python scripts/g_gonderim.py D_kqgpT5_adsb_28a 1.0 b=D_kqgpxg_17a_b5 "v206 ile aynı (komşu 1. + 2. tur + yankı geçmişi, düzeltici OOF kqgpxg) + Model A'ya EUROCONTROL günlük serileri (gün × havalimanı: slotlu kalkış payı, slot dışı geç / erken pay, kalkış öncesi gecikme tümü / ATC; scripts/gunluk_seri_ozellik.py). v206 ile farkı = günlük serilerin yankılı tabandaki etkisi." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"
