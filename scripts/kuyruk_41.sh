#!/bin/bash
# 03.10 akşam: v202 LB 228,6852 (v200'e göre −0,99) → komşu özellikleri (KOM + KTAH,
# kutuk/ekoz_secim.json) kalıcı. Model A'nın 5 bileşeninin _kqgpy OOF'ları 6 bölmede; GCB'nin OOF'u tek tohum (test tarafı
# 2 tohum ortalaması kalır; tohum korelasyonu 0,9935). Sonra harman kqgpy ile düzeltici CV (12 ay) ve v203
# (test tabanı v202 ile aynı; fark yalnız düzelticinin komşulu OOF ile eğitilmesi). Kayıt: kutuk/kuyruk_41.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
S="split_1 split_2 split_3 split_4 split_5 split_6"
echo "başladı $(date +%T); seçim: $(cat kutuk/ekoz_secim.json | tr -d '\n ')"
(
  run kqgpy_lgb_s16 scripts/aday_dogrula.py v_lgb_huber1000_3t_kqgpy $S \
  && run kqgpy_e1_s16 scripts/aday_dogrula.py v_e1_huber1000_3t_kqgpy $S
  echo "islemci: $? $(date +%T)" > kutuk/k41_cpu.done
) &
(
  run kqgpy_dw_s16 scripts/gcb_dw_tarama.py dw_lr03_ek5_l2_10_kqgpy $S \
  && run kqgpy_xgb_s16 scripts/aday_dogrula.py v_xgb_kqgpy_1500 $S \
  && run kqgpy_gcb12k_s16 scripts/gcb_varyant.py it12000_kqgpy 12000 0.06 9 $S \
  && python - <<'PY'
import shutil
from pathlib import Path
P = Path("data/preds")
for k in range(1, 7):
    shutil.copyfile(P / f"GCB_it12000_kqgpy_split_{k}.parquet", P / f"GCB_it12000_ort2_kqgpy_split_{k}.parquet")
print("GCB ort2 OOF = tek tohum kopyası")
PY
  echo "ekran: $? $(date +%T)" > kutuk/k41_gpu.done
) &
wait
cat kutuk/k41_cpu.done kutuk/k41_gpu.done
grep -q "islemci: 0" kutuk/k41_cpu.done && grep -q "ekran: 0" kutuk/k41_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
echo "== OOF harman kıyası (6 bölme) $(date +%T)"
python -u scripts/harman_kiyas.py kqgpxg kqgpy 2>&1 | tail -15
export PRC_HARMAN=kqgpy PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
echo "== düzeltici CV, kqgpy 12 ay $(date +%T) (kqgpxg 12 ay: $(grep '^k=1.0' kutuk/duz_cv_s6.txt | awk '{print $NF}'))"
python -u scripts/adsb_duzeltici.py cv $D 2>&1 | grep -E "^k=0.0 |^k=1.0 |^[A-Z]{4} |süre" | tee kutuk/duz_cv_kqgpy.txt
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpy.json D_kqgpy > kutuk/g_test_D_kqgpy.log 2>&1
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_23a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpy_23a.log
python scripts/g_gonderim.py D_kqgpy_adsb_23a 1.0 b=D_kqgpxg_17a_b5 "v202 ile aynı test tabanı (Model A + komşu özellikleri KOM/KTAH) ama ADS-B düzelticisi komşulu Model A'nın 12 aylık OOF'larıyla eğitildi (harman kqgpy). Düzeltici CV Oca/Tem: kqgpxg $(grep '^k=1.0' kutuk/duz_cv_s6.txt | awk '{print $NF}') → kqgpy $(grep '^k=1.0' kutuk/duz_cv_kqgpy.txt | awk '{print $NF}')." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -2
echo "kuyruk bitti $(date +%T)"
