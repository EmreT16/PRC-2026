#!/bin/bash
# 09.10 gece (09.10 LB): v305 / v301 → A'da G2 CatBoost (+0,090) ve G2b (+0,045) 2026 Oca / Tem'de
# kötü. Final kararı 09.10 LB'sine (v324 / v328 / v330 / v326) kalıyor; final hattında her A değişkeni hazır olsun diye
# final_son.sh'nin ana adayından SONRA aynı B (B5 _bF) ile A değişkenleri kurulur. final_son.sh çalışırken düzenlenmedi
# (bash betiği satır satır okur); bu betik ayrı süreç.
#   G2 LightGBM (_ha_g2): final_son A kolu bittikten sonra (fs_a.done) aynı düzenle, PRC_G2_MODEL'siz.
#   Adaylar: _ha_g2 (↔ v324), _ha_g2lgbcb (↔ v326), _ha (↔ v328), _42o (↔ v330); her biri Oca / Tem satırlarında karşılığıyla.
# Kayıt: kutuk/final_a_varyant.log (+ kutuk/fv_*.log)
cd "$(dirname "$0")/.."  # 09.10: göreli kök (depo / tekrar kopyası)
export PYTHONIOENCODING=utf-8 PRC_FINAL=1 PRC_Y1_ESIK=1.0
AD=D_kqgpPe_adsb_42o
BAD=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bF
F="warn"
run() { local log="kutuk/fv_$1.log"; shift; python -u "$@" 2>&1 | grep -v -i "$F" > "$log"; local rc=${PIPESTATUS[0]}; echo "cikis: $rc" >> "$log"; return $rc; }
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
yol() { grep -o "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" "$1" | head -1 | tr '\\' '/'; }
echo "başladı $(date +%T)"
until [ -f kutuk/fs_a.done ] || grep -q HATA kutuk/final_son.log 2>/dev/null; do sleep 60; done
grep -q "A: 0" kutuk/fs_a.done 2>/dev/null || dur "final_son A kolu yok / hatalı"
PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPe PRC_DUZ_CV=parkb \
  run g2lgb scripts/gecikme_asama2.py ${AD}_ha ha=1 || dur "G2 LightGBM"
test -f data_final/preds/TEST_${AD}_ha_g2.parquet || dur "_ha_g2 çıktısı yok"
echo "G2 LightGBM tamam $(date +%T)"
until grep -qE "aday: ara|HATA" kutuk/final_son.log 2>/dev/null; do sleep 60; done
grep -q HATA kutuk/final_son.log && dur "final_son hatalı"
sleep 120   # final_son'un 5. adımı (denetim) yeni dosya yazmıyor; yine de g_gonderim adlandırması çakışmasın
for v in "_ha_g2 v324 G2_yalnız_LightGBM" "_ha_g2lgbcb v326 G2_LightGBM+CatBoost,_G2b_yok" "_ha v328 G2_yok" " v330 G2_ve_H1_yok"; do
  set -- $v
  if [ $# -eq 2 ]; then e=""; k=$1; a=$2; else e=$1; k=$2; a=$3; fi
  python scripts/g_gonderim.py ${AD}${e} 1.0 b=$BAD lobt=1 "FINAL (4 ay) A değişkeni: A = ${AD}${e} (${a//_/ }), B final_son ile aynı. Oca / Tem karşılığı $k. Ara dosya (Y1'siz)." > kutuk/fv_g_$k.log 2>&1 || dur "gönderim $k"
  ara=$(yol kutuk/fv_g_$k.log); [ -n "$ara" ] || dur "ara dosya yok $k"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. FINAL A değişkeni (${a//_/ }); Oca / Tem karşılığı $k." > kutuk/fv_k_$k.log 2>&1 || dur "Y1 $k"
  son=$(yol kutuk/fv_k_$k.log)
  echo "$k karşılığı: ara $ara → $son $(grep -o 'MD5: `[0-9a-f]*`' kutuk/fv_k_$k.log | head -1)"
  python - "$son" "$k" <<'PY'
import os, sys, numpy as np, pandas as pd
if not os.path.exists(f"gonderim/strong-moose_{sys.argv[2]}.parquet"):
    sys.exit(0)  # karşılaştırma yalnız normal test gönderimi varken; yoksa atlanır
f = pd.read_parquet(sys.argv[1]); e = pd.read_parquet(f"gonderim/strong-moose_{sys.argv[2]}.parquet")
r = pd.read_parquet("data_final/dep_rank.parquet", columns=["MVT_ID_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC"])
assert len(f) == 670_790 and f.TAXITIME_SEC_mvt.notna().all() and f.TAXITIME_SEC_mvt.min() >= 60
x = f.merge(r, on="MVT_ID_mvt")
y = x.merge(e, on="MVT_ID_mvt", suffixes=("", "_e")); d = y.TAXITIME_SEC_mvt - y.TAXITIME_SEC_mvt_e
print(f"   ay ortalaması {x.groupby('MVT_MONTH').TAXITIME_SEC_mvt.mean().round(1).to_dict()} | Oca/Tem {len(y):,}: − {sys.argv[2]} RMS {np.sqrt((d ** 2).mean()):.1f}, ort {d.mean():+.2f}, r {np.corrcoef(y.TAXITIME_SEC_mvt, y.TAXITIME_SEC_mvt_e)[0, 1]:.4f}")
PY
done
echo "zincir bitti $(date +%T)"
