#!/bin/bash
# Final aşaması tek giriş noktası (09.10): final_ranking.parquet'ten bütün final aday
# dosyalarını sırayla üretir. Önkoşullar (README, How to run): ../data/final_ranking.parquet + final_submitting.parquet;
# eğitim tarafı ürünleri data/ altında (OOF tahminleri, 2025 ADS-B gün özellikleri, park merkezleri;
# docs/training_side_manifest.csv); ADS-B noktaları ve OPDI olayları PRC_ADSB_KOK altında (adsb_hat.py, opdi_indir.py);
# METAR / ansperf / AIU dosyaları ../extra_data/ altında.
# Adımlar: taban tablolar → eğitim tarafı kopyaları → v126 B tabanı → 12 özellik betiği → Model A üç nesil → final_son
# (ADS-B gün özellikleri, park dizisi, düzeltici → H1 → G2 / G2b, Model B, LIRF, 24 sa, B5, gönderim + Y1) → A değişkenleri
# → 8 havalimanlı B → TabM aşaması (final_tabm.sh: TabM'li A harmanı + düzeltici, B μ'ye TabM, B5, gönderim + Y1)
# → gonderim_final/strong-moose_final.parquet.
# Kayıt: bash scripts/final_tum.sh > kutuk/final_tum.log 2>&1 (+ alt betiklerin kendi kayıtları)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_FINAL=1
adim() { echo "[$(date +%T)] $*"; }
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
adim "başladı ($(pwd))"
rm -f kutuk/fs_a.done kutuk/fs_b.done kutuk/fa_*.done
python -u scripts/final_taban.py > kutuk/ft_taban.log 2>&1 || dur "final_taban"
adim "taban (dep_train / dep_rank / V3) tamam"
python -u scripts/final_hazirlik.py > kutuk/ft_hazirlik.log 2>&1 || dur "final_hazirlik"
python -u -c "from src import features_v126 as F; F.base_table('train'); F.base_table('rank'); print('base tamam')" > kutuk/ft_base.log 2>&1 || dur "base_table"
python -u scripts/final_b_taban.py > kutuk/ft_b_taban.log 2>&1 || dur "final_b_taban"
adim "hazırlık tamam"
bash scripts/final_ozellik.sh > kutuk/final_ozellik.log 2>&1; grep -q "^bitti" kutuk/final_ozellik.log || dur "final_ozellik"
adim "özellikler tamam"
bash scripts/final_a_nesil.sh > kutuk/final_a_nesil.log 2>&1; grep -q "zincir bitti" kutuk/final_a_nesil.log || dur "final_a_nesil"
adim "Model A nesilleri tamam"
bash scripts/final_son.sh > kutuk/final_son.log 2>&1; grep -q "aday: ara" kutuk/final_son.log || dur "final_son"
adim "ana final adayı tamam: $(grep -o 'gonderim_final[\/]strong-moose_final_a[0-9]*\.parquet → [^ ]*' kutuk/final_son.log | head -1)"
bash scripts/final_a_varyant.sh > kutuk/final_a_varyant.log 2>&1; grep -q "zincir bitti" kutuk/final_a_varyant.log || dur "final_a_varyant"
bash scripts/final_b8.sh > kutuk/final_b8.log 2>&1; grep -q "zincir bitti" kutuk/final_b8.log || dur "final_b8"
# 10.10: k = 0,883 dosyası kaldırıldı (LB v339 +0,51 → k = 1). TabM aşaması (A harmanı + düzeltici, B μ) ve son dosya:
bash scripts/final_tabm.sh > kutuk/final_tabm.log 2>&1; grep -q "zincir bitti" kutuk/final_tabm.log || dur "final_tabm"
SON=$(grep -o "final aday: ara [^ ]* → [^ ]*" kutuk/final_tabm.log | awk '{print $NF}')
cp "$SON" gonderim_final/strong-moose_final.parquet || dur "son dosya kopyası"
adim "son dosya: $SON → gonderim_final/strong-moose_final.parquet ($(md5sum gonderim_final/strong-moose_final.parquet | cut -c1-32))"
adim "bitti"
