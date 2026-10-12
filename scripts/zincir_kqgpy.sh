#!/bin/bash
# 02.10 akşam: tarama (kuyruk_38) bitince ekoz_karar → kabul edilen yüzey/kar
# özellikleriyle 5 bileşenin yeniden eğitimi ("_kqgpy"; işlemci ve ekran kartı paralel) → harman karşılaştırması →
# düzeltici CV ve üretim (δ=400, D1 yok; v198 düzeni) → gönderim (v199). Kayıt: kutuk/zincir_kqgpy.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
until grep -q "^bitti" kutuk/tarama_ay.log 2>/dev/null; do sleep 120; done
echo "tarama bitti $(date +%T)"
python scripts/ekoz_karar.py || { echo "KABUL YOK, zincir durdu $(date +%T)"; exit 0; }
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
(
  run kqgpy_lgb_split123 scripts/aday_dogrula.py v_lgb_huber1000_3t_kqgpy split_1 split_2 split_3 \
  && run kqgpy_e1_split123 scripts/aday_dogrula.py v_e1_huber1000_3t_kqgpy split_1 split_2 split_3 \
  && run d_kqgpy_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy \
  && run d_kqgpy_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy
  echo "islemci: $? $(date +%T)" > kutuk/kqgpy_cpu.done
) &
(
  T=dw_lr03_ek5_l2_10_kqgpy
  run kqgpy_dw_split123 scripts/gcb_dw_tarama.py $T split_1 split_2 split_3 \
  && run d_kqgpy_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
     cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
  && run kqgpy_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
  && run kqgpy_gcb12k scripts/gcb_varyant.py it12000_kqgpy 12000 0.06 9 split_1 split_2 split_3 \
  && run kqgpy_gcb12k_t2026 scripts/gcb_varyant.py it12000_t2026_kqgpy 12000 0.06 9 split_1 split_2 split_3 tohum=2026 \
  && run d_kqgpy_gcb12k scripts/d_global_cb.py 10000 it12000_kqgpy \
  && run d_kqgpy_gcb12k_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 \
  && run kqgpy_gcb12k_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
  && run kqgpy_xgb_split123 scripts/aday_dogrula.py v_xgb_kqgpy_1500 split_1 split_2 split_3 \
  && run d_kqgpy_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500
  echo "ekran: $? $(date +%T)" > kutuk/kqgpy_gpu.done
) &
wait
cat kutuk/kqgpy_cpu.done kutuk/kqgpy_gpu.done
grep -q "islemci: 0" kutuk/kqgpy_cpu.done && grep -q "ekran: 0" kutuk/kqgpy_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
echo "== harman karşılaştırması (6 ay OOF, taksi RMSE) $(date +%T)"
python scripts/harman_kiyas.py kqgpxg kqgpy
export PRC_HARMAN=kqgpy PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 delta=400"
echo "== düzeltici CV (kqgpy, δ=400; kqgpx δ=400 karşılığı kutuk/kuyruk_37.log) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre"
echo "== üretim $(date +%T)"
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpy.json D_kqgpy > kutuk/g_test_D_kqgpy.log 2>&1
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_18a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpy_18a.log
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 ad=D_kqgpy_18a_b5 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_b5_uretim_D_kqgpy_18a.log
python scripts/g_gonderim.py D_kqgpy_adsb_18a 1.0 b=D_kqgpy_18a_b5 "v198 düzeni (kqgpx tabanı, iz5, seyreltilmiş kopyalar, düzeltici δ=400, D1 yok) + Model A'nın 5 bileşeninde yeni özellikler (kutuk/ekoz_secim.json: varış taksi-in penceresi / yapay yığınsız NM taksisi / kar; kqgpy) ve global CatBoost 12.000 iterasyon." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -3
echo "zincir bitti $(date +%T)"
