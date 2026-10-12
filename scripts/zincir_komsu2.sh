#!/bin/bash
# 03.10 akşam: komşu özellikleri 2. tur (scripts/komsu2_oz.py: K2, K2OFS, K2V, K2Q).
# 1) kuyruk_41'in kqgpy LGB OOF'u bitince tarama (TARAMA_K2=1, taban lgbh3_kqgpy, kontrol + KOM + KTAH)
# 2) kuyruk_41 bitince karar: grup ort ≤ −0,10 ve ≥ 5/6 ay iyi (K2HEP en iyi tekten ≥ 0,05 iyiyse hepsi)
# 3) komsu2_oz.parquet (eğitim: kqgpy OOF 12 ay; test: TEST_D_kqgpy) → ekoz_secim'e ekle → 5 bileşen tam eğitim (_kqgpy;
#    1. tur test dosyaları *_k1 yedeği) → harman kqgpy2 (OOF kqgpy) → v204 (ek=_24a). v204 − v203 = 2. turun etkisi.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
until grep -q "cikis:" kutuk/kqgpy_lgb_s16.log 2>/dev/null; do sleep 300; done
grep -q "cikis: 0" kutuk/kqgpy_lgb_s16.log || { echo "kqgpy LGB OOF hatalı; durdu $(date +%T)"; exit 1; }
echo "== tarama $(date +%T)"
TARAMA_K2=1 TARAMA_BIL=ADAY_v_lgb_huber1000_3t_kqgpy python -u scripts/tarama_olay.py ay > kutuk/tarama_k2.log 2>&1
cat kutuk/tarama_k2.log
until grep -q "kuyruk bitti\|durdu" kutuk/kuyruk_41.log 2>/dev/null; do sleep 300; done
grep -q "kuyruk bitti" kutuk/kuyruk_41.log || { echo "kuyruk_41 hatalı; durdu $(date +%T)"; exit 1; }
SEC=$(python - <<'PY'
import re
r = {}
for l in open("kutuk/tarama_k2.log", encoding="utf-8"):
    m = re.match(r"(\w+): .*\| ort ([+-][\d.]+) \| iyi (\d)/6", l)
    if m: r[m.group(1)] = (float(m.group(2)), int(m.group(3)))
gec = {g: v for g, v in r.items() if g not in ("NOISE", "K2HEP") and v[0] <= -0.10 and v[1] >= 5}
hep = r.get("K2HEP", (0, 0))
if hep[0] <= -0.10 and hep[1] >= 5 and (not gec or hep[0] <= min(v[0] for v in gec.values()) - 0.05):
    print("K2,K2OFS,K2V,K2Q")
else:
    print(",".join(gec))
PY
)
echo "seçilen gruplar: '${SEC}' $(date +%T)"
[ -n "$SEC" ] || { echo "RET: 2. tur grubu yok; durdu $(date +%T)"; exit 0; }
python -u scripts/komsu2_oz.py $SEC > kutuk/komsu2_oz.log 2>&1 || { echo "komsu2_oz hatalı"; exit 1; }
tail -15 kutuk/komsu2_oz.log
cp kutuk/ekoz_secim.json kutuk/ekoz_secim_k1.json
python - <<'PY'
import json, pandas as pd
p = "kutuk/ekoz_secim.json"
s = json.load(open(p, encoding="utf-8"))
s["komsu2_oz.parquet"] = [c for c in pd.read_parquet("data/komsu2_oz.parquet").columns if c != "MVT_ID_mvt"]
json.dump(s, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ekoz_secim:", s)
PY
for f in data/preds/TEST_*kqgpy*.parquet; do case "$f" in *_sicak.parquet|*_k1.parquet) ;; *) cp "$f" "${f%.parquet}_k1.parquet";; esac; done
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
echo "eğitim başladı $(date +%T)"
(
  run d_k2_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy \
  && run d_k2_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy
  echo "islemci: $? $(date +%T)" > kutuk/k2_cpu.done
) &
(
  T=dw_lr03_ek5_l2_10_kqgpy
  run d_k2_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
     cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
  && run k2_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
  && run d_k2_gcb12k scripts/d_global_cb.py 10000 it12000_kqgpy \
  && run d_k2_gcb12k_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 \
  && run k2_gcb12k_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
  && run d_k2_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500
  echo "ekran: $? $(date +%T)" > kutuk/k2_gpu.done
) &
wait
cat kutuk/k2_cpu.done kutuk/k2_gpu.done
grep -q "islemci: 0" kutuk/k2_cpu.done && grep -q "ekran: 0" kutuk/k2_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
export PRC_HARMAN=kqgpy2 PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpy.json D_kqgpy2 > kutuk/g_test_D_kqgpy2.log 2>&1
python - <<'PY'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_D_kqgpy.parquet", columns=["MVT_ID_mvt", "taksi"])
b = pd.read_parquet(P + "TEST_D_kqgpy2.parquet", columns=["MVT_ID_mvt", "taksi"])
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
d = a.merge(b, on="MVT_ID_mvt", suffixes=("_eski", "_yeni")).merge(r, on="MVT_ID_mvt")
d["f"] = d["taksi_yeni"] - d["taksi_eski"]
print("Model A test tabanı, 2. tur − 1. tur: RMS", round(float(np.sqrt((d.f ** 2).mean())), 1), "| ort", round(d.f.mean(), 1))
print(d.groupby(["ADEP_mvt", "MVT_MONTH"]).f.mean().unstack().round(1).to_string())
PY
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_24a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpy2_24a.log
python scripts/g_gonderim.py D_kqgpy2_adsb_24a 1.0 b=D_kqgpxg_17a_b5 "v203 ile aynı (düzeltici komşulu OOF, harman kqgpy) ama Model A'nın 5 bileşeninin TEST tahminleri komşu özelliklerinin 2. turuyla ($SEC; komşu tahminleri komşulu tabandan, scripts/komsu2_oz.py) yeniden eğitildi. v203 ile farkı = 2. turun LB etkisi." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"
