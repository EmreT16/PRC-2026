#!/bin/bash
# 11.10 gece: LB'de B μ TabM CV'nin ~4 katı taşındığı için (v347 −0,858)
# μ TabM'in ayarları: (1) Huber δ 1.000 → 2.000 (B normal rejiminde uzun kuyruk; RMSE'ye yakın kayıp), (2) epok 12 → 20 (kayıp
# 12. epokta hâlâ düşüyor), (3) kontrol: aynı ayarlar tohum 7 (gürültü düzeyi → eşik). Hepsi tek tohum, çeyrek dışı CV.
# Karşılaştırma scripts/b_mu_nn_kiyas.py (kural betiğin başında). B5 12 ay zinciri bittikten sonra (RAM / ekran kartı).
# Kayıt: kutuk/b_mu_varyant_cv.log (+ kutuk/bmv_*.log)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONHASHSEED=0
D="durum=park,ktah,kom,gs,pc,yte"
cv() { local ad=$1; shift; python -u scripts/b_mu_nn.py cv $D "$@" > kutuk/bmv_$ad.log 2>&1 || echo "HATA: $ad $(date +%T)"; echo "$ad bitti $(date +%T)"; }
until grep -q "^bitti" kutuk/b5_12ay.log 2>/dev/null; do sleep 60; done
echo "başladı $(date +%T)"
cv dl2000 epok=12 tohum=42 delta=2000
cv e20 epok=20 tohum=42
cv s7 epok=12 tohum=7
cv bw5 epok=12 tohum=42 b_agirlik=5
python scripts/b_mu_nn_kiyas.py kontrol=_dparkktahkomgspcyte_e12_t1_s7 _dparkktahkomgspcyte_e12_t1_dl2000 _dparkktahkomgspcyte_e20_t1 _dparkktahkomgspcyte_e12_t1_bw5 2>&1 | grep -v -i warn
echo "zincir bitti $(date +%T)"
