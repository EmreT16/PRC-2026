#!/bin/bash
# 05.10:
#  a) Y1 kalibrasyonu (scripts/g_kalibre.py): v208 + Y1 (tek değişken) ve zincir_agirlik'in son dosyası + Y1.
#  b) Pist konfigürasyonu (data/pist_oz.parquet, PC_*; taramada nötr, ama "durum" özellikleri LB'de CV'den çok kazandırıyor):
#     v208 kurgusu (aynı tohum / ağırlık) + PC_* ile tam yeniden eğitim → harman kqgpT9 → gönderim (lobt=1).
#     Bitince TEST dosyaları ve ekoz_secim.json v208 hâline döner (PC_*'li sürümler *_pist).
# Kayıt: kutuk/zincir_y1_pist.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
until grep -qE "zincir bitti|durdu" kutuk/zincir_agirlik.log 2>/dev/null; do sleep 60; done
SON=$(ls -t gonderim/strong-moose_v*.parquet | head -1)
echo "başladı $(date +%T); son gönderim $SON"
python -u scripts/g_kalibre.py gonderim/strong-moose_v208.parquet "v208 + Y1 EUROCONTROL aylık taksi-out kalibrasyonu (scripts/g_kalibre.py; 2025 tanım farkı düşülür, 2·σ yumuşak eşik, geçersiz payı > %10 aylar atlanır) + LOBT penceresi. v208 ile farkı = Y1." > kutuk/kalibre_v208.log 2>&1
tail -n 5 kutuk/kalibre_v208.log
python -u scripts/g_kalibre.py "$SON" "$(basename $SON) + Y1 kalibrasyonu (aynı kural) + LOBT penceresi. Kaynakla farkı = Y1." > kutuk/kalibre_son.log 2>&1
tail -n 5 kutuk/kalibre_son.log
# --- pist konfigürasyonu
P=data/preds
BIL="TEST_v_lgb_huber1000_3t_kqgpy TEST_v_e1_huber1000_3t_kqgpy TEST_v_xgb_kqgpy_1500 TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_t42 TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_t2026 TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_t7 TEST_gcb_dw_lr03_ek5_l2_10_kqgpy_ort3 TEST_gcb_it12000_kqgpy TEST_gcb_it12000_t2026_kqgpy TEST_gcb_it12000_ort2_kqgpy"
for b in $BIL; do cp "$P/$b.parquet" "$P/${b}_onpist.parquet"; done
cp kutuk/ekoz_secim.json kutuk/ekoz_secim_onpist.json
python - <<'PY'
import json
s = json.load(open("kutuk/ekoz_secim.json", encoding="utf-8"))
s["pist_oz.parquet"] = ["PC_N_DEP", "PC_N_ARR", "PC_DEGISTI", "PC_PAY", "PC_BASKIN", "PC_KUYRUK_W", "PC_AGIR_N", "PC_SPAN20", "PC_CFG_TE"]
json.dump(s, open("kutuk/ekoz_secim.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ekoz_secim:", list(s))
PY
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "pist eğitimi başladı $(date +%T)"
(
  run pi_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy \
  && run pi_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy
  echo "islemci: $? $(date +%T)" > kutuk/pi_cpu.done
) &
(
  T=dw_lr03_ek5_l2_10_kqgpy
  run pi_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
     cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
  && run pi_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
  && run pi_gcb scripts/d_global_cb.py 10000 it12000_kqgpy \
  && run pi_gcb_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 \
  && run pi_gcb_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
  && run pi_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500
  echo "ekran: $? $(date +%T)" > kutuk/pi_gpu.done
) &
wait
cat kutuk/pi_cpu.done kutuk/pi_gpu.done
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
cp kutuk/ekoz_secim_onpist.json kutuk/ekoz_secim.json
if grep -q "islemci: 0" kutuk/pi_cpu.done && grep -q "ekran: 0" kutuk/pi_gpu.done; then
  export PRC_HARMAN=kqgpT9 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
  python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT.json D_kqgpT9 > kutuk/g_test_D_kqgpT9.log 2>&1
  python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
x = pd.read_parquet(P + "TEST_D_kqgpT5.parquet", columns=["MVT_ID_mvt", "taksi"]).merge(
    pd.read_parquet(P + "TEST_D_kqgpT9.parquet", columns=["MVT_ID_mvt", "taksi"]), on="MVT_ID_mvt")
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
x = x.merge(r, on="MVT_ID_mvt", how="left")
x["f"] = x.taksi_y - x.taksi_x
print(f"Model A test tabanı, pist − v208: RMS {np.sqrt((x.f ** 2).mean()):.1f} | ort {x.f.mean():+.1f}")
print(x.groupby(["ADEP_mvt", "MVT_MONTH"]).f.mean().unstack().round(1).to_string())
PY
  D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
  python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_33a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT9_33a.log
  python scripts/g_gonderim.py D_kqgpT9_adsb_33a 1.0 b=D_kqgpxg_17a_b5 lobt=1 "v208 ile aynı (komşu + yankı + günlük seriler; tohum ve ağırlıklar v208'inki) + Model A'ya pist konfigürasyonu özellikleri (scripts/pist_ozellik.py, PC_*: pist sayıları, konfigürasyon değişimi, pist payı / baskınlık, iz türbülanslı pist kuyruğu, ağır uçak, son 20 kalkış süresi, konfigürasyon TE) + LOBT penceresi. Taramada nötrdü (+0,003); durum özellikleri LB'de CV'den fazla kazandırdığı için LB sınaması. v208 ile farkı = pist (+ ~40 satırlık LOBT)."
else
  echo "pist eğitimi hatalı $(date +%T)"
fi
for b in $BIL; do cp "$P/$b.parquet" "$P/${b}_pist.parquet"; cp "$P/${b}_onpist.parquet" "$P/$b.parquet"; done
echo "TEST dosyaları ve ekoz_secim v208 hâline döndü"
ls -t gonderim | head -4
echo "zincir bitti $(date +%T)"
