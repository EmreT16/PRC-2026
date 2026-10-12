"""G1 havalimanı bazlı ağırlıklar — her havalimanı için ayrı SLSQP ağırlığı, çapraz bölmede sınanır (eğitim yok).

Kullanım: python scripts/g1_havalimani.py <etiket> <bileşen> [<bileşen> ...]   (bileşen sözdizimi: g1_harman_genel.py)
Taban: aynı bileşenlerin tek (genel) ağırlıklı harmanı; ikisi de diğer bölmede seçilen ağırlıklarla.
Aday: havalimanı ağırlığı ile genel ağırlığın λ karışımı (λ = 1: yalnızca havalimanı; λ = 0,5: yarı yarıya büzülme).
Önerilen ağırlıklar (iki seçimin ortalaması) → kutuk/g1_havalimani_<etiket>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402
import g1_harman_genel as GH  # noqa: E402  (aynı klasördeki betik)

LAMBDAS = (1.0, 0.5)


def pick(d, names, split):
    glob = GH.choose(d, names, split)
    return glob, {a: GH.choose(d[d["ADEP_mvt"] == a], names, split) for a in sorted(d["ADEP_mvt"].unique())}


def blend(d, names, glob, per, lam):
    P, out = d[names].to_numpy(float), d[names].to_numpy(float) @ glob
    for a, w in per.items():
        m = (d["ADEP_mvt"] == a).to_numpy()
        out[m] = P[m] @ (lam * w + (1 - lam) * glob)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tag, comps = sys.argv[1], GH.parse(sys.argv[2:])
    data = {s: GH.load(s, comps) for s in V.SPLITS}
    names = data["split_1"][1]
    picks = {s: pick(data[s][0], names, s) for s in V.SPLITS}
    for pick_s, test in (("split_1", "split_2"), ("split_2", "split_1")):
        dt = data[test][0]
        glob, per = picks[pick_s]
        base = dt[names].to_numpy(float) @ glob
        for lam in LAMBDAS:
            res = V.compare(dt["y"], base, blend(dt, names, glob, per, lam), dt["MVT_MONTH"], dt["days"],
                            V.SPLITS[test], nm_diff=dt["NM_TAXI_DIFF_SEC"])
            print(V.format_comparison(res, f"{test}'de sınama: havalimanı ağırlıkları (λ={lam}) ↔ genel ağırlıklar"),
                  flush=True)
            V.log_experiment({"ad": f"G1H_{tag}_{pick_s}_secim_l{lam}", "tur": "G1", "sinama": test,
                              "sonuc": {"taban": res["base"]["weighted_rmse"], "aday": res["new"]["weighted_rmse"],
                                        "delta": res["delta_weighted_rmse"], "delta_ci": res["bootstrap"]["drmse_ci"],
                                        "gunler": f"{res['days_won']}/{res['n_days']}", "kurallar": res["rules"],
                                        "kabul": res["accepted"]}})
    avg = {a: ((picks["split_1"][1][a] + picks["split_2"][1][a]) / 2).round(4).tolist() for a in picks["split_1"][1]}
    for a, w in avg.items():
        print(f"  {a}: {dict(zip(names, w))}")
    (config.LOG_DIR / f"g1_havalimani_{tag}.json").write_text(
        json.dumps({"bilesenler": names, "havalimani": avg,
                    "genel": ((picks["split_1"][0] + picks["split_2"][0]) / 2).round(4).tolist()},
                   ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
