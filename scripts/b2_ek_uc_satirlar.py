"""B2 eki — Uç satırların yapısı (E fazı D'den önce geldiği için). Model eğitmez.

B1 harman tahminleri (60 sn alt sınırlı) + ham aylık dosyalardan zaman damgaları (yalnızca doğrulama ayları, 8 sütun).
1. BLOCK bir plan zamanına "kilitli" mi (|BLOCK − plan| ≤ 60 sn): SCHED, EOBT_1, LOBT, IOBT; satır ve MSE payı.
2. Kilitli ve etkili satırlar: BLOCK ≈ SCHED ve |SCHED − AOBT_3| > 600 sn (kilit, NM'nin AOBT_3'ünden belirgin uzakta).
   Kâhin: bu satırlarda tahmin SCHED − AOBT_3 olsaydı ağırlıklı RMSE ne olurdu (E1'in üst sınırı; gerçekleşemez).
   Taban oranı: |SCHED − AOBT_3| > 600 olan satırların kaçı kilitli (sınıflandırıcının işi bu ayrımı yapmak).
3. En kötü %5 hata diliminin bileşimi.
4. AOBT_3 tam dakika bayrağı havalimanı bazında (F1'in önceliği için).
5. En büyük 15 hata (satır düzeyinde, yalnızca zaman farkları).
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
TIMES = {"MVT": "MVT_TIME_UTC_mvt", "BLOCK": "BLOCK_TIME_UTC_mvt", "SCHED": "SCHED_TIME_UTC_mvt",
         "AOBT3": "AOBT_3_flt", "EOBT1": "EOBT_1_flt", "LOBT": "LOBT_flt", "IOBT": "IOBT_flt"}
TOL = 60


def load(split):
    d = pd.read_parquet(PRED_DIR / f"B1_{split}.parquet")
    raw = pd.concat([pd.read_parquet(glob.glob(RAW.format(m=m))[0], columns=["MVT_ID_mvt", *TIMES.values()])
                     for m in V.SPLITS[split]])
    d = d.merge(raw, on="MVT_ID_mvt", how="left", validate="one_to_one")
    sec = lambda a, b: (d[TIMES[a]] - d[TIMES[b]]).dt.total_seconds()  # noqa: E731
    chk = np.abs(sec("BLOCK", "AOBT3").to_numpy() - d["y"].to_numpy())
    assert np.nanmax(chk) < 1e-3, f"y ≠ BLOCK − AOBT_3 (en büyük fark {np.nanmax(chk)})"
    for p in ("SCHED", "EOBT1", "LOBT", "IOBT"):
        d[f"BLK_{p}"] = sec("BLOCK", p)
    d["SCHED_AOBT3"] = sec("SCHED", "AOBT3")
    d["pred"] = V.floor_offset(d["blend"], d["NM_TAXI_DIFF_SEC"])
    d["e"] = d["pred"] - d["y"]
    w = V.month_weights(V.SPLITS[split])
    d["katki"] = d["MVT_MONTH"].map(w) * d["e"] ** 2 / d.groupby("MVT_MONTH")["e"].transform("size")
    d["AOBT3_TAM_DAKIKA"] = d[TIMES["AOBT3"]].dt.second.eq(0)
    return d, w


def share(d, mask, total):
    e = d.loc[mask, "e"]
    return {"satir": mask.mean(), "mse": d.loc[mask, "katki"].sum() / total,
            "rmse": float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan,
            "ort_hata": float(e.mean()) if len(e) else np.nan, "n": int(mask.sum())}


def row(name, s):
    return f"| {name} | {s['n']:,} | {s['satir']:.2%} | {s['mse']:.1%} | {s['rmse']:.0f} | {s['ort_hata']:+.0f} |"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    md = ["# B2 eki — Uç satırların yapısı", "",
          f"Hata = B1 harman ofset tahmini (60 sn alt sınırlı) − gerçek ofset. \"Kilitli\": \\|BLOCK − plan zamanı\\| ≤ {TOL} sn. "
          "MSE payı ay ağırlıklı.", ""]
    head = ["| Dilim | Satır | Satır payı | MSE payı | RMSE | Ort. hata |", "|---|---:|---:|---:|---:|---:|"]
    log = {}
    for split in V.SPLITS:
        d, w = load(split)
        total = d["katki"].sum()
        md += [f"## {split} ({len(d):,} satır, ağırlıklı RMSE {np.sqrt(total):.3f})", "", "### 1. BLOCK kilidi", ""]
        md += head
        for p in ("SCHED", "EOBT1", "LOBT", "IOBT"):
            md.append(row(f"BLOCK ≈ {p}", share(d, d[f"BLK_{p}"].abs() <= TOL, total)))
        md.append(row("BLOCK ≈ AOBT_3 (\\|y\\| ≤ 60)", share(d, d["y"].abs() <= TOL, total)))
        md.append("")

        far = d["SCHED_AOBT3"].abs() > 600
        locked = (d["BLK_SCHED"].abs() <= TOL) & far
        s_far, s_lock = share(d, far, total), share(d, locked, total)
        oracle = V.floor_offset(np.where(locked, d["SCHED_AOBT3"], d["pred"]), d["NM_TAXI_DIFF_SEC"])
        o = V.weighted_rmse(d["y"], oracle, d["MVT_MONTH"], w)["weighted_rmse"]
        md += ["### 2. Kilitli ve etkili satırlar (BLOCK ≈ SCHED, \\|SCHED − AOBT_3\\| > 600)", ""] + head
        md += [row("\\|SCHED − AOBT_3\\| > 600", s_far), row("… bunların kilitli olanları", s_lock),
               row("… kilitli olmayanları", share(d, far & ~locked, total)), "",
               f"- Taban oranı: \\|SCHED − AOBT_3\\| > 600 olan satırların %{100 * locked.sum() / max(far.sum(), 1):.1f}'i "
               "kilitli.",
               f"- **Kâhin üst sınırı:** kilitli satırlar kusursuz bilinip tahmin SCHED − AOBT_3 yapılsaydı ağırlıklı RMSE "
               f"{np.sqrt(total):.3f} → {o:.3f} (Δ {o - np.sqrt(total):+.3f}). Gerçekleşemez; sınıflandırıcı bunun bir "
               "kısmını alabilir.", ""]
        by_apt = pd.DataFrame({"satir": d.groupby("ADEP_mvt").size(),
                               "uzak": far.groupby(d["ADEP_mvt"]).sum(),
                               "kilitli": locked.groupby(d["ADEP_mvt"]).sum(),
                               "kilitli_mse_payi": d["katki"].where(locked, 0).groupby(d["ADEP_mvt"]).sum() / total})
        by_apt["taban_orani"] = by_apt["kilitli"] / by_apt["uzak"].clip(lower=1)
        md += ["| Havalimanı | Satır | \\|SCHED − AOBT_3\\| > 600 | Kilitli | Taban oranı | Kilitlilerin MSE payı |",
               "|---|---:|---:|---:|---:|---:|"]
        md += [f"| {r.Index} | {r.satir:,} | {r.uzak:,} | {r.kilitli:,} | {r.taban_orani:.1%} | {r.kilitli_mse_payi:.1%} |"
               for r in by_apt.itertuples()]
        md.append("")

        order = np.argsort(-d["e"].abs().to_numpy())
        top = np.zeros(len(d), bool)
        top[order[:int(np.ceil(0.05 * len(d)))]] = True
        top = pd.Series(top, index=d.index)
        md += ["### 3. En kötü %5'lik hata diliminin bileşimi", "", "| Özellik | Dilim içindeki satır payı | Tüm satırlarda |",
               "|---|---:|---:|"]
        comp = {"BLOCK ≈ SCHED": d["BLK_SCHED"].abs() <= TOL, "Kilitli ve etkili": locked,
                "\\|SCHED − AOBT_3\\| > 600": far, "Gerçek ofset < 0": d["y"] < 0,
                "\\|gerçek ofset\\| > 600": d["y"].abs() > 600, "AOBT_3 tam dakika değil": ~d["AOBT3_TAM_DAKIKA"]}
        md += [f"| {k} | {m[top].mean():.1%} | {m.mean():.1%} |" for k, m in comp.items()]
        md.append("")

        g = d.groupby(["ADEP_mvt", "AOBT3_TAM_DAKIKA"])
        fl = pd.DataFrame({"n": g.size(), "ort_y": g["y"].mean(), "ort_hata": g["e"].mean(),
                           "rmse": g["e"].apply(lambda s: float(np.sqrt(np.mean(s ** 2))))}).unstack()
        md += ["### 4. AOBT_3 tam dakika bayrağı (havalimanı bazında)", "",
               "| Havalimanı | n (tam değil) | Ort. ofset tam değil / tam | Ort. hata tam değil / tam | RMSE tam değil / tam |",
               "|---|---:|---:|---:|---:|"]
        for a, r in fl.iterrows():
            nn = r[("n", False)]
            if pd.isna(nn) or nn == 0:
                continue
            md.append(f"| {a} | {int(nn):,} | {r[('ort_y', False)]:+.0f} / {r[('ort_y', True)]:+.0f} | "
                      f"{r[('ort_hata', False)]:+.0f} / {r[('ort_hata', True)]:+.0f} | "
                      f"{r[('rmse', False)]:.0f} / {r[('rmse', True)]:.0f} |")
        # F1 üst sınırı: tam dakika olmayan satırların kare hatası, aynı havalimanı ve aydaki tam dakika satırlarının
        # ortalama kare hatasına insin (bayrak bu satırları tipik satırlardan iyi tahmin ettiremez varsayımı).
        e2 = d["e"] ** 2
        typical = e2.where(d["AOBT3_TAM_DAKIKA"]).groupby([d["ADEP_mvt"], d["MVT_MONTH"]]).transform("mean")
        e2_f1 = e2.where(d["AOBT3_TAM_DAKIKA"], np.minimum(e2, typical))
        f1 = float(np.sqrt(sum(wm * e2_f1[d["MVT_MONTH"] == mo].mean() for mo, wm in w.items())))
        md += ["", f"- **F1 üst sınırı:** tam dakika olmayan satırlar, aynı havalimanı ve aydaki tam dakika satırları "
               f"kadar iyi tahmin edilseydi ağırlıklı RMSE {np.sqrt(total):.3f} → {f1:.3f} "
               f"(Δ {f1 - np.sqrt(total):+.3f}).", ""]

        t15 = d.iloc[order[:15]]
        md += ["### 5. En büyük 15 hata (sn)", "",
               "| Havalimanı | Ay | Gerçek ofset | Tahmin (alt sınırlı) | MVT − AOBT_3 | BLOCK − SCHED | SCHED − AOBT_3 | BLOCK − EOBT_1 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        md += [f"| {r.ADEP_mvt} | {r.MVT_MONTH} | {r.y:+.0f} | {r.pred:+.0f} | {r.NM_TAXI_DIFF_SEC:.0f} | "
               f"{r.BLK_SCHED:+.0f} | {r.SCHED_AOBT3:+.0f} | {r.BLK_EOBT1:+.0f} |" for r in t15.itertuples()]
        md.append("")

        log[split] = {"kilitli_etkili": s_lock, "uzak": s_far, "kahin_rmse": o, "taban_rmse": float(np.sqrt(total)),
                      "top5_kilitli_payi": float(locked[top].mean()), "f1_ust_sinir_rmse": f1}
        print(f"[{split}] kilitli+etkili: satır {s_lock['satir']:.2%}, MSE {s_lock['mse']:.1%}, RMSE {s_lock['rmse']:.0f} | "
              f"taban oranı {locked.sum() / max(far.sum(), 1):.1%} | kâhin {np.sqrt(total):.3f} → {o:.3f}", flush=True)
    V.log_experiment({"ad": "B2_ek_uc_satirlar", "tur": "B2", "sonuc": log})
    (config.LOG_DIR / "b2_ek_uc_satirlar.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
