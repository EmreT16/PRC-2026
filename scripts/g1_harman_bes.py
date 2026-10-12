"""G1 — beş bileşenli harman, çapraz bölme (eğitim yok; G1).

Bileşenler: B1'in CB, LGB (3 tohum), XGB'si; A2'nin global derin CatBoost'u (GCB); E1 karışımı (LGB vekili).
Ağırlıklar bir bölmede SLSQP ile seçilir (negatif olmayan, toplamı 1; ay ağırlıklı MSE), diğer bölmede B1 harmanına
karşı 4 kabul kuralıyla sınanır. Son ağırlık önerisi: iki seçimin ortalaması.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_harman_ek import load  # noqa: E402

COMPS = ["cb", "lgb", "xgb", "global_cb", "karisim"]


def choose(d, split):
    w = V.month_weights(V.SPLITS[split])
    P = d[COMPS].to_numpy(float)
    y = d["y"].to_numpy(float)
    rw = d["MVT_MONTH"].map({m: wm / (d["MVT_MONTH"] == m).sum() for m, wm in w.items()}).to_numpy(float)

    def f(a):
        r = P @ a - y
        return float(np.sum(rw * r * r))

    res = minimize(f, np.full(len(COMPS), 1 / len(COMPS)), method="SLSQP", bounds=[(0, 1)] * len(COMPS),
                   constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}])
    return np.round(res.x, 3)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    data = {s: load(s) for s in V.SPLITS}
    picks = {}
    for pick, test in (("split_1", "split_2"), ("split_2", "split_1")):
        a = choose(data[pick], pick)
        picks[pick] = a
        dt = data[test]
        res = V.compare(dt["y"], dt["blend"], dt[COMPS].to_numpy(float) @ a, dt["MVT_MONTH"], dt["days"],
                        V.SPLITS[test], nm_diff=dt["NM_TAXI_DIFF_SEC"])
        print(f"{pick}'de seçilen ağırlıklar {dict(zip(COMPS, a.tolist()))}", flush=True)
        print(V.format_comparison(res, f"{test}'de sınama ↔ B1 harmanı"), flush=True)
        V.log_experiment({"ad": f"G1_bes_bilesen_{pick}_secim", "tur": "G1", "agirliklar": dict(zip(COMPS, a.tolist())),
                          "sinama": test,
                          "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                    "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                    "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                    "kabul": res["accepted"]}})
    avg = np.round((picks["split_1"] + picks["split_2"]) / 2, 3)
    avg = avg / avg.sum()
    print(f"önerilen (iki seçimin ortalaması): {dict(zip(COMPS, np.round(avg, 3).tolist()))}")
    for s in V.SPLITS:
        d = data[s]
        res = V.compare(d["y"], d["blend"], d[COMPS].to_numpy(float) @ avg, d["MVT_MONTH"], d["days"], V.SPLITS[s],
                        nm_diff=d["NM_TAXI_DIFF_SEC"])
        print(V.format_comparison(res, f"{s}: ortalama ağırlıklar ↔ B1 harmanı (kısmen örneklem içi)"), flush=True)
    (config.LOG_DIR / "g1_agirliklar.json").write_text(
        json.dumps({"bilesenler": COMPS, "agirlik": np.round(avg, 4).tolist(),
                    "secimler": {k: v.tolist() for k, v in picks.items()}}, ensure_ascii=False, indent=1),
        encoding="utf-8")


if __name__ == "__main__":
    main()
