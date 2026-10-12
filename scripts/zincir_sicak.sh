#!/bin/bash
# 03.10: Model A'ya sıcaklık / çiy noktası (kar_oz: SIC_0, SIC_MIN12, SPREAD_0; sıcaklık / çiy farkı ikili kar
# bayrağından daha bilgili olabilir; Ocak 2026 2025'ten ayazlı). Ay dışı CV bu
# özelliği haksız cezalandırdığı için yalnız LB sınaması: 5 bileşenin tam eğitimi (_kqgpy, kutuk/ekoz_secim.json), düzeltici
# v200'ünkiyle aynı (12 ay, OOF tarafı kqgpxg). v200 ile farkı = sıcaklığın etkisi. Kayıt: kutuk/zincir_sicak.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
echo "başladı $(date +%T); seçim: $(cat kutuk/ekoz_secim.json | tr -d '\n ')"
(
  run d_kqgpy_lgb scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy \
  && run d_kqgpy_e1 scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy
  echo "islemci: $? $(date +%T)" > kutuk/sicak_cpu.done
) &
(
  T=dw_lr03_ek5_l2_10_kqgpy
  run d_kqgpy_dw scripts/d_global_cb.py 9000 $T 42,2026,7 lr=0.03 depth=9 kat=ek5 cb:grow_policy=Depthwise \
     cb:min_data_in_leaf=50 cb:l2_leaf_reg=10 \
  && run kqgpy_dw_ort3 scripts/gcb_ortalama.py ${T}_ort3 ${T}_t42 ${T}_t2026 ${T}_t7 \
  && run d_kqgpy_gcb12k scripts/d_global_cb.py 10000 it12000_kqgpy \
  && run d_kqgpy_gcb12k_t2026 scripts/d_global_cb.py 10000 it12000_t2026_kqgpy 2026 \
  && run kqgpy_gcb12k_ort2 scripts/gcb_ortalama.py it12000_ort2_kqgpy it12000_kqgpy it12000_t2026_kqgpy \
  && run d_kqgpy_xgb scripts/d_tam_egitim.py v_xgb_kqgpy_1500
  echo "ekran: $? $(date +%T)" > kutuk/sicak_gpu.done
) &
wait
cat kutuk/sicak_cpu.done kutuk/sicak_gpu.done
grep -q "islemci: 0" kutuk/sicak_cpu.done && grep -q "ekran: 0" kutuk/sicak_gpu.done || { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
unset PRC_KUYRUK PRC_GECIKME PRC_GECPIST PRC_EKOZ
export PRC_HARMAN=kqgpT PRC_ADSB_OZ=adsb5 PRC_ADSB_GEVSEK=1 PRC_DUZ_EK=split_4,split_5,split_6
python -u scripts/g_test_harman.py kutuk/g1_agirliklar_kqgpT.json D_kqgpT > kutuk/g_test_D_kqgpT.log 2>&1
python - <<'EOF'
import pandas as pd, numpy as np
P = "data/preds/"
a = pd.read_parquet(P + "TEST_D_kqgpxg.parquet", columns=["MVT_ID_mvt", "taksi"])
b = pd.read_parquet(P + "TEST_D_kqgpT.parquet", columns=["MVT_ID_mvt", "taksi"])
r = pd.read_parquet("data/base_v126_rank_v1.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH"])
d = a.merge(b, on="MVT_ID_mvt", suffixes=("_eski", "_yeni")).merge(r, on="MVT_ID_mvt")
d["f"] = d["taksi_yeni"] - d["taksi_eski"]
print("Model A test tabanı, sıcaklıklı − eski: RMS", round(float(np.sqrt((d.f ** 2).mean())), 1), "| ort", round(d.f.mean(), 1))
print(d.groupby(["ADEP_mvt", "MVT_MONTH"]).f.mean().unstack().round(1).to_string())
EOF
D="havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 p:num_leaves=255 tur=800 kuyruk=1 iz5=1 seyrek=s30,s10"
python -u scripts/adsb_duzeltici.py uretim $D k=1 yanki_beta=0.25 yanki_apt=LIRF ek=_21a 2>&1 | grep -v -i "warn\|X\[\"tip\"\]" > kutuk/adsb_uretim_D_kqgpT_21a.log
python scripts/g_gonderim.py D_kqgpT_adsb_21a 1.0 b=D_kqgpxg_17a_b5 "v200 ile aynı (12 aylık düzeltici, OOF tarafı kqgpxg) ama Model A'nın 5 bileşeninin TEST tahminleri sıcaklık / çiy noktası özellikleriyle (SIC_0, SIC_MIN12, SPREAD_0; IEM METAR) yeniden eğitildi. v200 ile farkı = sıcaklığın LB etkisi (ay dışı CV bu özelliği ölçemiyor)." 2>&1 | grep -E "MD5|Dosya|✗"
ls -t gonderim | head -2
echo "zincir bitti $(date +%T)"
