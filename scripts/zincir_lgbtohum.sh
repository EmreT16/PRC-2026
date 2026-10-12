#!/bin/bash
# 05.10 gece: lgbh3 ve E1 test modellerine 3 ek tohum (1, 2, 3) → 6 tohum ortalaması.
# Başarılı olursa ve zincir_tohum.sh henüz harmana geçmediyse kqgpT7 / kqgpT8 JSON'larında e1h3_T → e1h3_T6, lgbh3_T → lgbh3_T6.
# Kayıt: kutuk/zincir_lgbtohum.log
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 PRC_KUYRUK=1 PRC_GECIKME=1 PRC_GECPIST=1 PRC_EKOZ=1
run() { local log="kutuk/$1.log"; shift; python -u "$@" > "$log" 2>&1; local rc=$?; echo "cikis: $rc" >> "$log"; return $rc; }
echo "başladı $(date +%T)"
run lt_lgb_b scripts/d_tam_egitim.py v_lgb_huber1000_3t_kqgpy_b \
&& run lt_e1_b scripts/d_tam_egitim.py v_e1_huber1000_3t_kqgpy_b \
|| { echo "eğitim hatalı; durdu $(date +%T)"; exit 1; }
python - <<'PY'
import json, pandas as pd, numpy as np
P = "data/preds/"
for a, b, o in (("TEST_v_lgb_huber1000_3t_kqgpy", "TEST_v_lgb_huber1000_3t_kqgpy_b", "TEST_v_lgb_huber1000_6t_kqgpy"),
                ("TEST_v_e1_huber1000_3t_kqgpy", "TEST_v_e1_huber1000_3t_kqgpy_b", "TEST_v_e1_huber1000_6t_kqgpy")):
    x, y = pd.read_parquet(P + a + ".parquet"), pd.read_parquet(P + b + ".parquet")
    assert (x.MVT_ID_mvt.to_numpy() == y.MVT_ID_mvt.to_numpy()).all()
    z = x.copy()
    for c in x.columns:
        if c not in ("MVT_ID_mvt", "NM_TAXI_DIFF_SEC") and np.issubdtype(x[c].dtype, np.floating):
            z[c] = (x[c].to_numpy(float) + y[c].to_numpy(float)) / 2
    z["taksi"] = np.maximum(60.0, z.NM_TAXI_DIFF_SEC.to_numpy(float) - z.harman.to_numpy(float))
    z.to_parquet(P + o + ".parquet", index=False)
    print(o, "tohum grupları korelasyonu", round(float(np.corrcoef(x.harman, y.harman)[0, 1]), 5),
          "| fark RMS", round(float(np.sqrt(np.mean((x.harman - y.harman) ** 2))), 2))
PY
if grep -q "eğitim bitti" kutuk/zincir_tohum.log; then echo "zincir_tohum harmana geçmiş; JSON değiştirilmedi $(date +%T)"; exit 0; fi
python - <<'PY'
import json
for f in ("kutuk/g1_agirliklar_kqgpT7.json", "kutuk/g1_agirliklar_kqgpT8.json"):
    j = json.load(open(f, encoding="utf-8"))
    j["bilesenler"] = [{"e1h3_T": "e1h3_T6", "lgbh3_T": "lgbh3_T6"}.get(b, b) for b in j["bilesenler"]]
    j["not"] += " + lgbh3 / E1 6 tohum (zincir_lgbtohum.sh)"
    json.dump(j, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f, j["bilesenler"])
PY
echo "zincir bitti $(date +%T)"
