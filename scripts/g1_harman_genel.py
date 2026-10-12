"""G1 genel — bileşen listesiyle çapraz bölme harmanı (eğitim yok; G1).

Kullanım: python scripts/g1_harman_genel.py <etiket> <bileşen> [<bileşen> ...]
bileşen: B1 sütunu (cb, lgb, xgb, lgb_s42) ya da <dosya>:<sütun>[=ad] (dosya data/preds/<dosya>_<split>.parquet;
ör. A2_global_cb:global_cb, E1:karisim, D2_huber_1000:pred=lgb_huber).
Ağırlıklar bir bölmede SLSQP ile seçilir (negatif olmayan, toplamı 1; ay ağırlıklı MSE), diğerinde B1 harmanına karşı
4 kabul kuralıyla sınanır. Önerilen ağırlık: iki seçimin ortalaması → kutuk/g1_agirliklar_<etiket>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"


def parse(specs):
    out = []
    for s in specs:
        if ":" in s:
            f, rest = s.split(":", 1)
            col, name = (rest.split("=", 1) + [None])[:2] if "=" in rest else (rest, None)
            out.append((f, col, name or f"{f}:{col}"))
        else:
            out.append((None, s, s))
    return out


def load(split, comps):
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    for f, col, name in comps:
        if f is None:
            if name != col:
                d[name] = d[col]
            continue
        b = pd.read_parquet(PRED_DIR / f"{f}_{split}.parquet")[["MVT_ID_mvt", col]].rename(columns={col: name})
        d = d.merge(b, on="MVT_ID_mvt", how="left", validate="one_to_one")
    names = [n for _, _, n in comps]
    assert d[names].notna().all().all()
    d["days"] = V.day_of_year(d["MVT_MONTH"], d["MVT_DAY"])
    return d, names


def choose(d, names, split):
    w = V.month_weights(V.SPLITS[split])
    P, y = d[names].to_numpy(float), d["y"].to_numpy(float)
    rw = d["MVT_MONTH"].map({m: wm / (d["MVT_MONTH"] == m).sum() for m, wm in w.items()}).to_numpy(float)
    res = minimize(lambda a: float(np.sum(rw * (P @ a - y) ** 2)), np.full(len(names), 1 / len(names)),
                   method="SLSQP", bounds=[(0, 1)] * len(names),
                   constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
    return np.round(res.x, 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tag, comps = sys.argv[1], parse(sys.argv[2:])
    data = {s: load(s, comps) for s in V.SPLITS}
    names = data["split_1"][1]
    picks, results = {}, {}
    for pick, test in (("split_1", "split_2"), ("split_2", "split_1")):
        a = choose(data[pick][0], names, pick)
        picks[pick] = a
        dt = data[test][0]
        res = V.compare(dt["y"], dt["blend"], dt[names].to_numpy(float) @ a, dt["MVT_MONTH"], dt["days"],
                        V.SPLITS[test], nm_diff=dt["NM_TAXI_DIFF_SEC"])
        results[test] = res
        print(f"[{tag}] {pick}'de seçilen {dict(zip(names, a.tolist()))}")
        print(V.format_comparison(res, f"{test}'de sınama ↔ B1 harmanı"), flush=True)
        V.log_experiment({"ad": f"G1_{tag}_{pick}_secim", "tur": "G1", "agirliklar": dict(zip(names, a.tolist())),
                          "sinama": test,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}})
    avg = (picks["split_1"] + picks["split_2"]) / 2
    avg = np.round(avg / avg.sum(), 4)
    print(f"[{tag}] önerilen ağırlıklar: {dict(zip(names, avg.tolist()))} | çapraz Δ: split_2 "
          f"{results['split_2']['delta_weighted_rmse']:+.3f}, split_1 {results['split_1']['delta_weighted_rmse']:+.3f}")
    (config.LOG_DIR / f"g1_agirliklar_{tag}.json").write_text(
        json.dumps({"bilesenler": names, "agirlik": avg.tolist(), "secimler": {k: v.tolist() for k, v in picks.items()},
                    "capraz_delta": {k: r["delta_weighted_rmse"] for k, r in results.items()}},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
