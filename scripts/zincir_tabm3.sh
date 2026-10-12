#!/bin/bash
# 10.10 akşam: split_1 pilotu (tabm_d1024_pilot.sh) KABUL derse tam koşu:
#   1) OOF: d_block 1024, 12 epok — tohum 42 (split_2–6; split_1 pilottan) + tohum 7 (split_1–6) → ortalama NN_tabm_kqgp3_*
#   2) 12 ay değerlendirme (kqgpxg harman + w · TabM, düzeltici rhat sabit): kqgp2 ↔ kqgp3 kendi en iyi w'lerinde.
#      Önceden tanımlı kural: kqgp3 toplamda ≥ 0,05 iyi ve ≥ 8/12 ay iyi → w = kqgp3'ün en iyi w'si (ızgara 0,30–0,50).
#   3) Kabulse: g1_agirliklar_kqgpPeT3.json (5 ağaç × (1 − w) / 0,6 ölçeğiyle + TabM w), test TabM (normal + final, 2 tohum),
#      harman D_kqgpPeT3, düzeltici _42p ayarlarıyla (planli + LightGBM + TabM) → adaylar: normal v353'ün, final a54'ün karşılığı.
# Kayıt: kutuk/zincir_tabm3.log (+ kutuk/t3_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PRC_Y1_ESIK=1.0 PYTHONHASHSEED=0
F='warn\|X\["tip"\]'
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
until grep -q "^bitti\|HATA" kutuk/tabm_d1024_pilot.log 2>/dev/null; do sleep 60; done
grep -q "KABUL (tam koşu)" kutuk/tabm_d1024_pilot.log || { echo "pilot RET — tam koşu yapılmadı $(date +%T)"; exit 0; }
echo "başladı $(date +%T)"
T="model=tabm tabm_k=8 tabm_d=1024 amp=1 epok=12"
nn() { local ad=$1; shift; PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 python -u scripts/nn_bilesen.py $ad "$@" $T > kutuk/t3_$ad.log 2>&1 || dur "$ad $*"; grep "^\[" kutuk/t3_$ad.log | tail -1; }
# ---------------------------------------------------------------- 1. OOF
for s in split_2 split_3 split_4 split_5 split_6; do test -f data/preds/NN_tabm_kqgp_d1024_$s.parquet || nn tabm_kqgp_d1024 $s tohum=42; done
for s in split_1 split_2 split_3 split_4 split_5 split_6; do test -f data/preds/NN_tabm_kqgp_d1024t7_$s.parquet || nn tabm_kqgp_d1024t7 $s tohum=7; done
python - <<'PY' || dur "ortalama"
import sys, pandas as pd
sys.path.insert(0, ".")
from src import config
P = config.DATA_DIR / "preds"
for k in range(1, 7):
    a = pd.read_parquet(P / f"NN_tabm_kqgp_d1024_split_{k}.parquet"); b = pd.read_parquet(P / f"NN_tabm_kqgp_d1024t7_split_{k}.parquet")
    assert (a.MVT_ID_mvt.to_numpy() == b.MVT_ID_mvt.to_numpy()).all()
    a["nn"] = 0.5 * (a["nn"].to_numpy() + b["nn"].to_numpy())
    a.to_parquet(P / f"NN_tabm_kqgp3_split_{k}.parquet", index=False)
print("NN_tabm_kqgp3_split_1…6 yazıldı")
PY
echo "OOF tamam $(date +%T)"
# ---------------------------------------------------------------- 2. 12 ay değerlendirme
W=$(python - <<'PY' 2>kutuk/t3_degerlendir.log
import os, sys
os.environ["PRC_HARMAN"] = "kqgpxg"
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src import adsb as A, config
P = config.DATA_DIR / "preds"
rh = pd.read_parquet(P / "ADSB_duz_cv_parkb.parquet")
parca = []
for k in range(1, 7):
    d = A.d_dw_split(f"split_{k}")[["MVT_ID_mvt", "MVT_MONTH", "NM_TAXI_DIFF_SEC", "y", "ofset_taban"]]
    for ad in ("kqgp2", "kqgp3"):
        d = d.merge(pd.read_parquet(P / f"NN_tabm_{ad}_split_{k}.parquet").rename(columns={"nn": ad}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    parca.append(d.merge(rh, on="MVT_ID_mvt", how="left"))
d = pd.concat(parca, ignore_index=True)
d["rhat"] = d["rhat"].fillna(0.0)
nm, yt, h, r, ay = d.NM_TAXI_DIFF_SEC.to_numpy(float), (d.NM_TAXI_DIFF_SEC - d.y).to_numpy(float), d.ofset_taban.to_numpy(float), d.rhat.to_numpy(float), d.MVT_MONTH.to_numpy()
WS = [0.3, 0.35, 0.4, 0.45, 0.5]
en = {}
for ad in ("kqgp2", "kqgp3"):
    n = d[ad].to_numpy(float)
    tab = {}
    for w in WS:
        e = (np.maximum(60.0, nm - ((1 - w) * h + w * n) + r) - yt) ** 2
        tab[w] = {"toplam": np.sqrt(e.mean()), **{m: np.sqrt(e[ay == m].mean()) for m in range(1, 13)}}
    t = pd.DataFrame(tab).T
    print(ad, "\n", t.round(3).to_string(), file=sys.stderr)
    en[ad] = (t["toplam"].idxmin(), t.loc[t["toplam"].idxmin()])
(w2, s2), (w3, s3) = en["kqgp2"], en["kqgp3"]
f = s3 - s2
iyi = int(sum(f[m] < 0 for m in range(1, 13)))
kabul = f["toplam"] <= -0.05 and iyi >= 8
print(f"kqgp2 (w {w2}) {s2['toplam']:.3f} ↔ kqgp3 (w {w3}) {s3['toplam']:.3f}: {f['toplam']:+.3f}, iyi ay {iyi}/12 → {'KABUL' if kabul else 'RET'}", file=sys.stderr)
print(w3 if kabul else "RET")
PY
)
cat kutuk/t3_degerlendir.log | tail -3
[ "$W" = "RET" ] && { echo "12 ay RET — üretim yok $(date +%T)"; exit 0; }
[ -n "$W" ] || dur "değerlendirme"
python - "$W" <<'PY' || dur "ağırlık dosyası"
import json, sys
w = float(sys.argv[1])
j = json.load(open("kutuk/g1_agirliklar_kqgpPeT2.json", encoding="utf-8"))
ag = [round(a / 0.6 * (1 - w), 5) for a in j["agirlik"][:5]] + [w]
json.dump({"bilesenler": j["bilesenler"][:5] + ["tabm_T3"], "agirlik": ag,
           "not": f"10.10 akşam: kqgpPeT2'nin 5 ağaç bileşeni (göreli ağırlıklar aynı) × (1 − w) + TabM d1024 (2 tohum × 12 epok) × w = {w}; kutuk/t3_degerlendir.log"},
          open("kutuk/g1_agirliklar_kqgpPeT3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("g1_agirliklar_kqgpPeT3.json:", ag)
PY
cp data/preds/NN_tabm_kqgp3_split_*.parquet data_final/preds/
# ---------------------------------------------------------------- 3. test TabM, harman, düzeltici, adaylar
E="PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json"
test -f data/preds/TEST_nn_tabm_kqgpy_d1024.parquet || env $E python -u scripts/nn_bilesen.py tabm_kqgpy_d1024 tam $T tohum=42,7 > kutuk/t3_test_normal.log 2>&1 || dur "test TabM normal"
test -f data_final/preds/TEST_nn_tabm_kqgpy_d1024.parquet || env $E PRC_FINAL=1 python -u scripts/nn_bilesen.py tabm_kqgpy_d1024 tam $T tohum=42,7 > kutuk/t3_test_final.log 2>&1 || dur "test TabM final"
echo "test TabM tamam $(date +%T)"
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10 opdi=1 opdi_satir=1"
aday() {  # $1 = kip (normal / final), $2 = preds klasörü, $3 = B adı, $4 = dosya deseni, $5 = karşılık dosya
  python scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPeT3.json D_kqgpPeT3 > kutuk/t3_harman_$1.log 2>&1 || dur "harman $1"
  PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6 PRC_HARMAN=kqgpPeT3 PRC_EPOCH_TAM=1 \
    python -u scripts/adsb_duzeltici.py uretim $D parkb=1 k=1 yanki_beta=0.25 yanki_apt=LIRF planli=1 model=lgb+nn nn_w=0.7 nn_epok=8 ek=_42p \
    2>&1 | grep -v -i "$F" > kutuk/t3_duz_$1.log
  [ ${PIPESTATUS[0]} -eq 0 ] || dur "düzeltici $1"
  python scripts/g_gonderim.py D_kqgpPeT3_adsb_42p 1.0 b=$3 lobt=1 "A = D_kqgpPeT3_adsb_42p (TabM d1024 harman + düzeltici LightGBM + TabM + planli) + 8 havalimanlı B, B μ'ye TabM. $(basename $5) ile tek fark TabM'in genişliği. Ara dosya (Y1'siz)." > kutuk/t3_g_$1.log 2>&1 || dur "gönderim $1"
  local ara=$(grep -o "$4" kutuk/t3_g_$1.log | head -1 | tr '\\' '/'); [ -n "$ara" ] || dur "ara $1"
  python -u scripts/g_kalibre.py "$ara" pist=1 "$(basename $ara) + Y1 havalimanı + pist (1σ) + LOBT. TabM d1024." > kutuk/t3_k_$1.log 2>&1 || dur "Y1 $1"
  local son=$(grep -o "$4" kutuk/t3_k_$1.log | head -1 | tr '\\' '/')
  echo "$1: ara $ara → $son $(grep -o 'MD5: `[0-9a-f]*`' kutuk/t3_k_$1.log | head -1)"
  python - "$son" "$5" <<'PY'
import sys, numpy as np, pandas as pd
b, a = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
x = a.merge(b, on="MVT_ID_mvt", suffixes=("", "_y"))
assert len(x) == len(a) == len(b) and b.TAXITIME_SEC_mvt.notna().all() and b.TAXITIME_SEC_mvt.min() >= 60
f = x.TAXITIME_SEC_mvt_y - x.TAXITIME_SEC_mvt
print(f"  − {sys.argv[2]}: RMS {np.sqrt((f ** 2).mean()):.2f}, ort {f.mean():+.3f}")
PY
}
aday normal data/preds D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bdpdcb6c5N8 "gonderim[\/]strong-moose_v[0-9]*\.parquet" gonderim/strong-moose_v353.parquet
export PRC_FINAL=1
aday final data_final/preds D_kqgpxg_17a_b5_iz5_nmsiz_d5_12ay_k125_opdi_bN8 "gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet" gonderim_final/strong-moose_final_a54.parquet
echo "zincir bitti $(date +%T)"
