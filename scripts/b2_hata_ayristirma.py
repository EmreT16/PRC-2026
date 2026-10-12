"""B2 — Taban hatasının ayrıştırması. B1 tahminlerini kullanır; model eğitmez.

Hata e = B1 harman ofset tahmini (60 sn alt sınırlı, validation.floor_offset) − gerçek ofset; taksi hatasının
işaret değiştirmiş hâli, karesi aynı.
Paylar ay ağırlıklı MSE'ye göre: satır i'nin katkısı w_ay · e_i² / n_ay.
Dilimler: havalimanı, ay, |hata| (en kötü %0,1 / %1 / %5 / %10), |gerçek ofset|, MVT − AOBT_3 (NM_TAXI_DIFF),
AOBT_3 tam dakika bayrağı (ham aylık dosyalardan).
Sıra kararı: en kötü %5'lik hata dilimi ağırlıklı MSE'nin yarısından fazlasını taşıyorsa E fazı D'den önce.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, validation as V  # noqa: E402

PRED_DIR = config.DATA_DIR / "preds"
RAW = str(config.PRC2_DIR / "data" / "training_2025-{m:02d}-01_*.parquet")


def load(split):
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    d["e"] = V.floor_offset(d["blend"], d["NM_TAXI_DIFF_SEC"]) - d["y"]
    w = V.month_weights(V.SPLITS[split])
    n = d.groupby("MVT_MONTH")["e"].transform("size")
    d["katki"] = d["MVT_MONTH"].map(w) * d["e"] ** 2 / n
    raw = pd.concat([pd.read_parquet(glob.glob(RAW.format(m=m))[0], columns=["MVT_ID_mvt", "AOBT_3_flt"])
                     for m in V.SPLITS[split]])
    raw["AOBT3_TAM_DAKIKA"] = pd.to_datetime(raw["AOBT_3_flt"], utc=True).dt.second.eq(0)
    d = d.merge(raw[["MVT_ID_mvt", "AOBT3_TAM_DAKIKA"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    return d


def table(d, by):
    total = d["katki"].sum()
    g = d.groupby(by, observed=True)
    t = pd.DataFrame({"satir_payi": g.size() / len(d), "mse_payi": g["katki"].sum() / total,
                      "rmse": g["e"].apply(lambda s: float(np.sqrt(np.mean(s ** 2)))),
                      "ort_hata": g["e"].mean()})
    return t


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    md = ["# B2 — Taban (B1) hatasının ayrıştırması", "",
          "Hata = B1 harman ofset tahmini (60 sn alt sınırlı) − gerçek ofset. Paylar ay ağırlıklı MSE'ye göre.", ""]
    decision = {}
    for split in V.SPLITS:
        d = load(split)
        total = d["katki"].sum()
        md += [f"## {split}: ağırlıklı RMSE {np.sqrt(total):.3f} sn ({len(d):,} satır)", ""]

        a = d["e"].abs().to_numpy()
        order = np.argsort(-a)
        cum = np.cumsum(d["katki"].to_numpy()[order]) / total
        tops = {f"en kötü %{q * 100:g}": float(cum[max(0, int(np.ceil(q * len(d))) - 1)])
                for q in (0.001, 0.01, 0.05, 0.10)}
        decision[split] = tops["en kötü %5"]
        md += ["| Hata dilimi | Ağırlıklı MSE payı |", "|---|---:|"] + [f"| {k} satır | {v:.1%} |" for k, v in tops.items()]
        md.append("")

        slices = {
            "Havalimanı": table(d, "ADEP_mvt"),
            "Ay": table(d, "MVT_MONTH"),
            "Gerçek ofset (sn)": table(d, pd.cut(d["y"].abs(), [-1, 60, 300, 600, 1800, 3600, np.inf],
                                                 labels=["≤60", "60–300", "300–600", "600–1.800", "1.800–3.600",
                                                         ">3.600"])),
            "MVT − AOBT_3 (sn)": table(d, pd.cut(d["NM_TAXI_DIFF_SEC"], [-np.inf, 0, 300, 600, 1200, 1800, 3600,
                                                                          np.inf],
                                                 labels=["<0", "0–300", "300–600", "600–1.200", "1.200–1.800",
                                                         "1.800–3.600", ">3.600"])),
            "AOBT_3 tam dakika": table(d, "AOBT3_TAM_DAKIKA"),
        }
        for name, t in slices.items():
            md += [f"### {name}", "", "| Dilim | Satır payı | MSE payı | RMSE | Ort. hata |", "|---|---:|---:|---:|---:|"]
            md += [f"| {i} | {r.satir_payi:.2%} | {r.mse_payi:.1%} | {r.rmse:.1f} | {r.ort_hata:+.1f} |"
                   for i, r in t.iterrows()]
            md.append("")
        big = d[d["NM_TAXI_DIFF_SEC"] > 40_000]
        md += [f"NM_TAXI_DIFF > 40.000 sn olan satır: {len(big)}; MSE payı {big['katki'].sum() / total:.1%}.", ""]
        print(f"[{split}] ağırlıklı RMSE {np.sqrt(total):.3f} | " + " | ".join(f"{k}: {v:.1%}" for k, v in tops.items()))
        print(slices["Havalimanı"].round(3).to_string())
        V.log_experiment({"ad": f"B2_hata_ayristirma_{split}", "tur": "B2", "split": split,
                          "en_kotu_dilim_payi": tops,
                          "havalimani": slices["Havalimanı"].round(4).to_dict(orient="index"),
                          "ofset_dilimi": {str(k): v for k, v in slices["Gerçek ofset (sn)"].round(4)
                                           .to_dict(orient="index").items()}})

    first = all(v > 0.5 for v in decision.values())
    verdict = ("E fazı (uç satırlar) D'den önce: en kötü %5'lik dilim her iki bölmede MSE'nin yarısından fazlası."
               if first else "E fazı D'den sonra: en kötü %5'lik dilim en az bir bölmede MSE'nin yarısını geçmiyor.")
    md += ["## Sıra kararı", "", verdict, ""]
    (config.LOG_DIR / "b2_hata_ayristirma.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + verdict)


if __name__ == "__main__":
    main()
