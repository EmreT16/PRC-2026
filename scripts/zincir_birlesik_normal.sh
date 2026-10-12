#!/bin/bash
# 11.10 akşam: final_tabm.sh'nin tüm bayraklı (varsayılan) yapısının normal test
# karşılığı — A = D_kqgpPeT2t5_adsb_42p9T (taban TabM 5 tohum + düzeltici v 0,9, 5 tohum + T1), B = …_bdpdcb6c5N85L (B μ 5 tohum +
# LIRF μ kısmi TabM) → gönderim + Y1 (havalimanı + pist) + LOBT → yetim B penceresi → günlük kısıt (LIRF hariç).
# Ara toplanırlık denetimi: Y1 sonrası dosya − v370 yalnız LIRF B satırlarında (v363 − v361 ile aynı) değişmeli.
# Kayıt: kutuk/zincir_birlesik_normal.log (+ kutuk/zbn_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
yol() { grep -o "gonderim[\/]strong-moose_v[0-9]*\.parquet" "$1" | grep -v "$2" | head -1 | tr '\\' '/'; }
A=D_kqgpPeT2t5_adsb_42p9T
B=D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N85L
echo "başladı $(date +%T)"
python scripts/g_gonderim.py $A 1.0 b=$B lobt=1 "v370 + LIRF μ kısmi TabM (final_tabm.sh tüm bayraklar, normal). Ara dosya (Y1'siz)." \
  > kutuk/zbn_g.log 2>&1 || dur "gönderim"
ARA=$(yol kutuk/zbn_g.log NONE); [ -n "$ARA" ] || dur "ara dosya"
python -u scripts/g_kalibre.py "$ARA" pist=1 "$(basename $ARA) + Y1 havalimanı + pist (1σ) + LOBT. v370 + LIRF μ." > kutuk/zbn_k.log 2>&1 || dur "Y1"
Y1=$(yol kutuk/zbn_k.log "$(basename $ARA)"); [ -n "$Y1" ] || dur "Y1 çıktısı"
python scripts/b_yetim_pencere.py uretim "$Y1" "$(basename $Y1) + yetim B penceresi." > kutuk/zbn_yetim.log 2>&1 || dur "yetim"
YT=$(yol kutuk/zbn_yetim.log "$(basename $Y1)"); [ -n "$YT" ] || dur "yetim çıktısı"
python scripts/g_gunluk.py "$YT" a_dosya=$A haric=LIRF "$(basename $YT) + günlük kısıt (LIRF hariç). Birleşik aday: v370 + LIRF μ + yetim + günlük." \
  > kutuk/zbn_gunluk.log 2>&1 || dur "günlük"
SON=$(yol kutuk/zbn_gunluk.log "$(basename $YT)"); [ -n "$SON" ] || dur "günlük çıktısı"
echo "ara $ARA → Y1 $Y1 → yetim $YT → günlük $SON MD5: \`$(md5sum "$SON" | cut -c1-32)\`"
python - "$Y1" "$SON" <<'PY'
import sys, numpy as np, pandas as pd
r = lambda f: pd.read_parquet(f).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.astype(float)  # noqa: E731
y1, son = r(sys.argv[1]), r(sys.argv[2])
v361, v363, v370, v374 = (r(f"gonderim/strong-moose_v{n}.parquet").reindex(y1.index) for n in (361, 363, 370, 374))
v365, v368 = (r(f"gonderim/strong-moose_v{n}.parquet").reindex(y1.index) for n in (365, 368))
d1 = (y1 - v370) - (v363 - v361)
print(f"toplanırlık (Y1 − v370) − (v363 − v361): RMS {np.sqrt(np.mean(d1 ** 2)):.3f}, maks {np.abs(d1).max():.1f}")
d2 = (son - v374) - (y1 - v365)
print(f"toplanırlık (son − v374) − (Y1 − v365): RMS {np.sqrt(np.mean(d2 ** 2)):.3f}, maks {np.abs(d2).max():.1f}")
for ad, v in (("v374", v374), ("v368", v368), ("v370", v370)):
    d = (son - v).to_numpy()
    print(f"  − {ad}: değişen {(np.abs(d) > 1e-3).sum():,}, RMS {np.sqrt(np.mean(d ** 2)):.2f}, ort {d.mean():+.3f}, maks {np.abs(d).max():.0f}")
PY
echo "zincir bitti $(date +%T)"
