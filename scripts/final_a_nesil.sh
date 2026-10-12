#!/bin/bash
# Final aşaması (08.10): Model A'nın üç nesli, PRC_FINAL=1 (test = final_ranking, data_final/).
# Final yapılandırması (kqgpPe) iki komşu tahmini özellik grubu kullanıyor; ikisi de önceki neslin TEST harmanından türer:
#   1. nesil kqgpxg (kqgp özellikleri)                → TEST_D_kqgpxg → komsu_tah_oz  (scripts/komsu_tah_ozellik.py)
#   2. nesil kqgpy (+ komsu_oz, komsu_tah_oz; ekoz_secim_k1.json) → TEST_D_kqgpy → komsu2_oz (scripts/komsu2_oz.py)
#   3. nesil = final (ekoz_secim_final.json: + komsu2, yankı, günlük, pist; çok tohum) → TEST_D_kqgpPe
# Bileşen ayarları ve tohumlar eski zincirlerle aynı (kuyruk_32.sh, zincir_komsu.sh, zincir_y1_pist.sh, zincir_pist_tohum.sh).
# Tek fark: 1. ve 2. nesilde ekran kartı bileşenleri (GCB 12k, DW) tek tohum (42) — bu nesillerin tahmini yalnız 10 komşunun
# ortalaması olarak özellik oluyor; ağırlıklar g1_agirliklar_{kqgpxg,kqgpy}_F.json (eskileriyle aynı). LightGBM / E1 (3 tohum)
# ve XGB aynı. Nesiller aynı dosya adlarını kullandığı için 2. neslin test dosyaları *_g2 olarak saklanır; 3. neslin pistli
# dosyaları eski zincirdeki gibi *_pist adlarına kopyalanır.
# Kayıt: kutuk/final_a_nesil.log (+ adım başına kutuk/fa_<adım>.log)
cd "$(dirname "$0")/.."  # 09.10: göreli kök (depo / tekrar kopyası)
export PYTHONIOENCODING=utf-8 PRC_FINAL=1
P=data_final/preds
run() { local log="kutuk/fa_$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
dur() { echo "HATA: $* $(date +%T)"; exit 1; }
CBDW="lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise cb:min_data_in_leaf=50 cb:l2_leaf_reg=10"
T=dw_lr03_ek5_l2_10
until grep -qE "^bitti|HATA" kutuk/final_ozellik.log 2>/dev/null; do sleep 60; done
grep -q HATA kutuk/final_ozellik.log && dur "özellik zinciri hatalı"
echo "başladı $(date +%T)"

nesil() {  # $1 = ad, $2 = ad eki (kqgp / kqgpy); CPU ve GPU kolları paralel
  local N=$1 E=$2
  ( run ${N}_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_$E && run ${N}_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_$E
    echo "islemci: $? $(date +%T)" > kutuk/fa_${N}_cpu.done ) &
  ( run ${N}_dw scripts/d_global_cb.py 9000 ${T}_$E 42 $CBDW && run ${N}_gcb scripts/d_global_cb.py 10000 it12000_$E \
      && run ${N}_xgb scripts/d_tam_egitim.py v_xgb_${E}_1500
    echo "ekran: $? $(date +%T)" > kutuk/fa_${N}_gpu.done ) &
  wait
  grep -q "islemci: 0" kutuk/fa_${N}_cpu.done && grep -q "ekran: 0" kutuk/fa_${N}_gpu.done || dur "$N eğitimi"
}

# ---------------------------------------------------------------- 1. nesil (kqgp)
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1
nesil n1 kqgp
run n1_harman scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpxg_F.json D_kqgpxg || dur "1. nesil harmanı"
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST
run n1_komsu_tah scripts/komsu_tah_ozellik.py || dur "komsu_tah_oz"
echo "1. nesil + komsu_tah_oz tamam $(date +%T)"

# ---------------------------------------------------------------- 2. nesil (kqgpy, komşu 1. tur)
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_k1.json
nesil n2 kqgpy
run n2_harman scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpy_F.json D_kqgpy || dur "2. nesil harmanı"
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ PRC_EKOZ_DOSYA
run n2_komsu2 scripts/komsu2_oz.py || dur "komsu2_oz"
for b in v_lgb_huber1000_3t_kqgpy v_e1_huber1000_3t_kqgpy v_xgb_kqgpy_1500 gcb_${T}_kqgpy gcb_it12000_kqgpy; do
  cp "$P/TEST_$b.parquet" "$P/TEST_${b}_g2.parquet"; done
echo "2. nesil + komsu2_oz tamam $(date +%T)"

# ---------------------------------------------------------------- 3. nesil (final: kqgpPe, çok tohum)
export PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1 PRC_EKOZ_DOSYA=ekoz_secim_final.json
( run n3_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy && cp $P/TEST_v_lgb_huber1000_3t_kqgpy.parquet $P/TEST_v_lgb_huber1000_3t_kqgpy_pist.parquet \
  && run n3_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy && cp $P/TEST_v_e1_huber1000_3t_kqgpy.parquet $P/TEST_v_e1_huber1000_3t_kqgpy_pist.parquet \
  && run n3_lgb_pb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy_pb && run n3_e1_pb scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy_pb
  echo "islemci: $? $(date +%T)" > kutuk/fa_n3_cpu.done ) &
( run n3_dw scripts/d_global_cb.py 9000 ${T}_kqgpy 42,2026,7 $CBDW \
  && for s in 42 2026 7; do cp $P/TEST_gcb_${T}_kqgpy_t$s.parquet $P/TEST_gcb_${T}_kqgpy_t${s}_pist.parquet; done \
  && run n3_dw2 scripts/d_global_cb.py 9000 ${T}_pist_kqgpy 123,999 $CBDW \
  && run n3_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500 && cp $P/TEST_v_xgb_kqgpy_1500.parquet $P/TEST_v_xgb_kqgpy_1500_pist.parquet \
  && run n3_gcb scripts/d_global_cb.py 10000 it12000_kqgpy && cp $P/TEST_gcb_it12000_kqgpy.parquet $P/TEST_gcb_it12000_kqgpy_pist.parquet \
  && run n3_gcb_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 && cp $P/TEST_gcb_it12000_t2026_kqgpy.parquet $P/TEST_gcb_it12000_t2026_kqgpy_pist.parquet \
  && run n3_gcb_p7 scripts/d_global_cb.py 10000 it12000_p7_kqgpy 7 \
  && run n3_gcb_p123 scripts/d_global_cb.py 10000 it12000_p123_kqgpy 123
  echo "ekran: $? $(date +%T)" > kutuk/fa_n3_gpu.done ) &
wait
grep -q "islemci: 0" kutuk/fa_n3_cpu.done && grep -q "ekran: 0" kutuk/fa_n3_gpu.done || dur "3. nesil eğitimi"
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ PRC_EKOZ_DOSYA
run n3_gcb_ort4 scripts/gcb_ortalama.py it12000_ort4_pist_kqgpy it12000_kqgpy_pist it12000_t2026_kqgpy_pist it12000_p7_kqgpy it12000_p123_kqgpy || dur "GCB ortalaması"
run n3_dw_ort5 scripts/gcb_ortalama.py ${T}_kqgpy_ort5_pist ${T}_kqgpy_t42_pist ${T}_kqgpy_t2026_pist ${T}_kqgpy_t7_pist ${T}_pist_kqgpy_t123 ${T}_pist_kqgpy_t999 || dur "DW ortalaması"
python - <<'PY' || dur "LightGBM / E1 6 tohum ortalaması"
import pandas as pd, numpy as np
P = "data_final/preds/"
for a, b, o in (("TEST_v_lgb_huber1000_3t_kqgpy_pist", "TEST_v_lgb_huber1000_3t_kqgpy_pb", "TEST_v_lgb_huber1000_6t_pist_kqgpy"),
                ("TEST_v_e1_huber1000_3t_kqgpy_pist", "TEST_v_e1_huber1000_3t_kqgpy_pb", "TEST_v_e1_huber1000_6t_pist_kqgpy")):
    x, y = pd.read_parquet(P + a + ".parquet"), pd.read_parquet(P + b + ".parquet")
    assert (x.MVT_ID_mvt.to_numpy() == y.MVT_ID_mvt.to_numpy()).all()
    z = x.copy()
    for c in x.columns:
        if c not in ("MVT_ID_mvt", "NM_TAXI_DIFF_SEC") and np.issubdtype(x[c].dtype, np.floating):
            z[c] = (x[c].to_numpy(float) + y[c].to_numpy(float)) / 2
    z["taksi"] = np.maximum(60.0, z.NM_TAXI_DIFF_SEC.to_numpy(float) - z.harman.to_numpy(float))
    z.to_parquet(P + o + ".parquet", index=False)
    print(o, "korelasyon", round(float(np.corrcoef(x.harman, y.harman)[0, 1]), 5))
PY
run n3_harman scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpPe.json D_kqgpPe || dur "final harmanı"
echo "3. nesil tamam $(date +%T)"

# ---------------------------------------------------------------- denetim: Oca / Tem satırlarında yeni ↔ eski (normal mod) harmanlar
python - <<'PY'
import os, pandas as pd, numpy as np
r = pd.read_parquet("data_final/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
for ad in ("D_kqgpxg", "D_kqgpy", "D_kqgpPe"):
    y = pd.read_parquet(f"data_final/preds/TEST_{ad}.parquet", columns=["MVT_ID_mvt", "taksi"])
    if not os.path.exists(f"data/preds/TEST_{ad}.parquet"):
        continue  # karşılaştırma yalnız normal test tahmini varken; yoksa atlanır
    e = pd.read_parquet(f"data/preds/TEST_{ad}.parquet", columns=["MVT_ID_mvt", "taksi"])
    x = y.merge(e, on="MVT_ID_mvt", suffixes=("_y", "_e")).merge(r, on="MVT_ID_mvt")
    f = x.taksi_y - x.taksi_e
    print(f"{ad}: Oca/Tem ortak {len(x):,} | yeni − eski RMS {np.sqrt((f ** 2).mean()):.1f}, ort {f.mean():+.2f}, korelasyon {np.corrcoef(x.taksi_y, x.taksi_e)[0, 1]:.4f}")
    a = y.merge(r, on="MVT_ID_mvt")
    print("   ay başına ortalama taksi (yeni):", a.groupby("MVT_MONTH").taksi.mean().round(1).to_dict())
PY
echo "zincir bitti $(date +%T)"
