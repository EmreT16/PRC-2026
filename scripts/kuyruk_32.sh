#!/bin/bash
# 01.10 gece: global CatBoost 12.000 iterasyon (bileşen düzeyinde 6.000'e göre Oca −0,24, Tem +0,03, Şub −0,67, Ağu −0,50).
# Eksik bölmeler + iki tohumun tam eğitimi (sabit 10.000 iterasyon; bölmelerde iç erken durdurma 8.399 / 9.866'da durdu)
# + ortalama → harman kqgpxg → düzeltici CV (iz5) → v195 (iz5 + seyrek + NM'siz LTFM/LFPG D1).
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
echo "başladı $(date +%T)"
run kqgp_gcb12k_s3 scripts/gcb_varyant.py it12000_kqgp 12000 0.06 9 split_3 \
 && run kqgp_gcb12k_t2026 scripts/gcb_varyant.py it12000_t2026_kqgp 12000 0.06 9 split_1 split_2 split_3 tohum=2026 \
 && run d_kqgp_gcb12k scripts/d_global_cb.py 10000 it12000_kqgp \
 && run d_kqgp_gcb12k_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgp 2026 \
 && run kqgp_gcb12k_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgp it12000_kqgp it12000_t2026_kqgp \
 || { echo "kuyruk hatalı $(date +%T)"; exit 1; }
echo "eğitim bitti $(date +%T)"
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST
export PRC_HARMAN=kqgpxg PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1
echo "== düzeltici CV (kqgpxg, iz5) $(date +%T)"
python -u scripts/adsb_duzeltici.py cv havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 2>&1 | grep -E "^k=0.0 |^k=1.0 |süre"
echo "== v195 $(date +%T)"
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpxg.json D_kqgpxg > kutuk/g_test_D_kqgpxg.log 2>&1
python -u scripts/adsb_duzeltici.py uretim havalimani=0 kayip=huber k=1 oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 yanki_beta=0.25 yanki_apt=LIRF kuyruk=1 iz5=1 seyrek=s30,s10 ek=_17a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpxg_17a.log
python -u scripts/adsb_b_duzeltici5.py uretim apt=EDDF,EDDM,EHAM,LEBL,LSZH,EGLL,LEMD,LFPG k=1 ad=D_kqgpxg_17a_b5 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_b5_uretim_D_kqgpxg_17a.log
python scripts/b_d1_uygula.py D_kqgpxg_17a_b5
python scripts/g_gonderim.py D_kqgpxg_adsb_17a 1.0 b=D_kqgpxg_17a_b5_d1 "v194 ile aynı (iz5, seyreltilmiş kopyalar, NM'siz LTFM/LFPG D1), Model A'da global CatBoost 12.000 iterasyon (kqgpxg)." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -3
echo "kuyruk bitti $(date +%T)"
