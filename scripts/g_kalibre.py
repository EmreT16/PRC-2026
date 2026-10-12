"""Y1 — EUROCONTROL aylık taksi-out kalibrasyonu (05.10).

Kaynak: ansperformance.eu "Taxi-out additional time" aylık havalimanı verisi (extra_data/ansperf/txout.xlsx →
txout_10apt.parquet; sürüm 15.09.2026).
Sütunlar: TF (toplam kalkış), VALID_FL (geçerli uçuş; taksi > 120 dk, AOBT sonrası buz çözme ve
eksik veri hariç), yay_ort = (TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) · 60 / TOTAL_REF_NB_FL (sn; scripts/ansperf_txout.py).

Kural (önceden tanımlı; LB'ye bakılmadan):
  1. Tanım farkı: 2025'in her ayında fk = yay_ort − (gerçek taksinin en uzun K = TF − VALID_FL satırı atılmış ortalaması);
     havalimanı başına fk_ort ve fk_std (12 ay).
  2. 2026 hedefi = yay_ort − fk_ort. Gönderimin havalimanı × ay tahminlerinden en büyük K' = K · n / TF tanesi atılır,
     kalanların ortalaması − hedef = sapma.
  3. Uygulanan kayma = −işaret(sapma) · max(0, |sapma| − e · fk_std) (yumuşak eşik; e = PRC_Y1_ESIK: varsayılan 2, final hattı 1); geçersiz payı (K / TF) %10'u aşan
     aylar atlanır (buz çözme ayları: kırpılmış ortalama güvenilmez). Kayma o havalimanı × ayın tüm satırlarına (A ve B)
     eklenir, sonra 60 sn alt sınır ve LOBT penceresi (g_gonderim.lobt_kirp).
  4. pist=1 (05.10, ayrı aday): havalimanı kaymasından sonra pist katmanı. Kaynak AIU havalimanı panosu "Monthly Taxi-Out
     Time per Runway" (https://aiu-airport-dashboard.netlify.app/<icao>; extra_data/aiu/aiu_pist_aylik.parquet;
     referans + ek süre). Pist tanım farkı 2025 ayları üzerinden; koşullar: test satırı ≥ 1.000, ≥ 9 ay, fk_std ≤ 25,
     |fk_ort| ≤ 50, eşik 3·fk_std (2025 Oca/Tem ay dışı doğrulamada nötr: 299,475 → 299,477); o ayın herhangi bir pistinde
     |sapma| > 500 sn ise (pist adı tutarsızlığı, ör. LTFM Ocak 2026) o ay pist katmanı atlanır.
Kullanım: python scripts/g_kalibre.py <kaynak_gönderim.parquet> [pist=1] [not]   (yalnız dökümü görmek için: ... kuru)
Çıktı: gonderim/strong-moose_v<N>.parquet, kutuk/gonderim_<ad>.md, skor defterine satır."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # noqa: E402
import g_gonderim as G  # noqa: E402

YAY = config.EK_VERI / "ansperf" / "txout_10apt.parquet"
import os  # noqa: E402
ESIK_SIGMA = float(os.environ.get("PRC_Y1_ESIK", 2.0))  # 08.10: 2025 ay dışarıda benzetimi 1σ −%0,174 / 2σ −%0,147
GECERSIZ_UST = 0.10
AIU = config.EK_VERI / "aiu" / "aiu_pist_aylik.parquet"
PIST_SIGMA, PIST_MIN_N, PIST_MIN_AY, PIST_MAX_STD, PIST_MAX_FK, PIST_TUTARSIZ = 3.0, 1000, 9, 25.0, 50.0, 500.0


def tanim_farki():
    pub = pd.read_parquet(YAY)
    tr = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=["ADEP_mvt", "MVT_MONTH", "TAXITIME_SEC_mvt"])
    tr = tr[tr.TAXITIME_SEC_mvt.notna()]
    sat = []
    for q in pub[pub.YEAR == 2025].itertuples():
        v = np.sort(tr[(tr.ADEP_mvt == q.APT_ICAO) & (tr.MVT_MONTH == q.MONTH_NUM)].TAXITIME_SEC_mvt.to_numpy(float))
        K = int(round(max(q.TF - q.VALID_FL, 0) * len(v) / q.TF))
        sat.append((q.APT_ICAO, q.MONTH_NUM, q.yay_ort - v[:len(v) - K].mean()))
    fk = pd.DataFrame(sat, columns=["apt", "ay", "fk"])
    return fk.groupby("apt").fk.agg(["mean", "std"]).rename(columns={"mean": "fk_ort", "std": "fk_std"}), pub


def _kirp(v, g):
    v = np.sort(v)
    K = int(round(g * len(v)))
    return v[:len(v) - K].mean() if len(v) - K > 0 else np.nan


def pist_tanim_farki(pub):
    w = pd.read_parquet(AIU)
    tr = pd.read_parquet(config.DATA_DIR / "base_v126_train_v1.parquet", columns=["ADEP_mvt", "MVT_MONTH", "RUNWAY_mvt", "TAXITIME_SEC_mvt"])
    tr = tr[tr.TAXITIME_SEC_mvt.notna()]
    gp = {(q.APT_ICAO, q.MONTH_NUM): max(q.TF - q.VALID_FL, 0) / q.TF for q in pub[pub.YEAR == 2025].itertuples()}
    w25 = w[w.yil == 2025].set_index(["apt", "ay", "pist"]).taksi_sn
    sat = []
    for (a, m, p), g in tr.groupby(["ADEP_mvt", "MVT_MONTH", tr.RUNWAY_mvt.astype(str)]):
        if len(g) >= 100 and (a, m, p) in w25.index:
            sat.append((a, p, w25[(a, m, p)] - _kirp(g.TAXITIME_SEC_mvt.to_numpy(float), gp[(a, m)])))
    f = pd.DataFrame(sat, columns=["apt", "pist", "fk"]).groupby(["apt", "pist"]).fk.agg(["mean", "std", "count"])
    return f, w[w.yil == 2026].set_index(["apt", "ay", "pist"]).taksi_sn


def kalibre(mvt_id, taksi, pist=False):
    """(yeni taksi, döküm tablosu)."""
    fk, pub = tanim_farki()
    r = pd.read_parquet(G.HAM_RANK, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"])
    r = pd.DataFrame({"MVT_ID_mvt": mvt_id}).merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    ay = pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    apt = r.ADEP_mvt.to_numpy()
    pst = r.RUNWAY_mvt.astype(str).to_numpy()
    if pist:
        pf, w26 = pist_tanim_farki(pub)
    yeni = taksi.copy()
    sat = []
    for a in sorted(set(apt)):
        for m in config.TEST_AYLAR:  # 08.10: finalde 1, 2, 6, 7
            ix = np.where((apt == a) & (ay == m))[0]
            q = pub[(pub.YEAR == 2026) & (pub.APT_ICAO == a) & (pub.MONTH_NUM == m)]
            if not len(ix) or q.empty or not np.isfinite(q.yay_ort.iloc[0]):
                continue
            q = q.iloc[0]
            gp = max(q.TF - q.VALID_FL, 0) / q.TF
            K = int(round(gp * len(ix)))
            p = np.sort(taksi[ix])
            sapma = p[:len(p) - K].mean() - (q.yay_ort - fk.loc[a, "fk_ort"])
            esik = ESIK_SIGMA * fk.loc[a, "fk_std"]
            kay = 0.0 if gp > GECERSIZ_UST else -np.sign(sapma) * max(0.0, abs(sapma) - esik)
            yeni[ix] = np.maximum(60.0, taksi[ix] + kay)
            sat.append((a, m, "hepsi", len(ix), int(q.TF), round(gp * 100, 1), round(q.yay_ort, 1), round(fk.loc[a, "fk_ort"], 1),
                        round(fk.loc[a, "fk_std"], 1), round(sapma, 1), round(kay, 1)))
            if not pist or gp > GECERSIZ_UST:
                continue
            ps = {}
            for p in sorted(set(pst[ix])):
                ixr = ix[pst[ix] == p]
                if (a, m, p) not in w26.index or (a, p) not in pf.index:
                    continue
                hedef_r = w26[(a, m, p)] - pf.loc[(a, p), "mean"]
                ps[p] = (ixr, _kirp(yeni[ixr], gp) - hedef_r)
            if any(abs(v[1]) > PIST_TUTARSIZ for v in ps.values()):
                sat.append((a, m, "pist atlandı (tutarsız)", 0, 0, 0, 0, 0, 0, 0, 0))
                continue
            for p, (ixr, sp) in ps.items():
                f = pf.loc[(a, p)]
                if len(ixr) < PIST_MIN_N or f["count"] < PIST_MIN_AY or f["std"] > PIST_MAX_STD or abs(f["mean"]) > PIST_MAX_FK:
                    continue
                kp = -np.sign(sp) * max(0.0, abs(sp) - PIST_SIGMA * f["std"])
                yeni[ixr] = np.maximum(60.0, yeni[ixr] + kp)
                sat.append((a, m, f"pist {p}", len(ixr), 0, 0, round(w26[(a, m, p)], 1), round(f["mean"], 1), round(f["std"], 1),
                            round(sp, 1), round(kp, 1)))
    t = pd.DataFrame(sat, columns=["apt", "ay", "kapsam", "n", "TF", "geçersiz_%", "yayımlanan", "fk_ort", "fk_std", "sapma", "kayma"])
    return yeni, t


def main(kaynak, note, kuru=False, pist=False):
    sub = pd.read_parquet(kaynak)
    t0 = sub["TAXITIME_SEC_mvt"].to_numpy(float)
    yeni, t = kalibre(sub["MVT_ID_mvt"].to_numpy(), t0, pist)
    yeni, km = G.lobt_kirp(sub["MVT_ID_mvt"].to_numpy(), yeni)
    print(t.to_string(index=False))
    d = yeni - t0
    print(f"değişen satır {int((np.abs(d) > 1e-6).sum()):,} | RMS {np.sqrt(np.mean(d ** 2)):.2f} | ort {d.mean():+.2f} | LOBT kırpılan {int(km.sum())}")
    if kuru:
        return
    out = pd.DataFrame({"MVT_ID_mvt": sub["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": yeni.astype(np.float32)})
    tpl = pd.read_parquet(config.SUBMIT_TEMPLATE, columns=["MVT_ID_mvt"])
    assert len(out) == G.N_ROWS and np.array_equal(out.MVT_ID_mvt.to_numpy(), tpl.MVT_ID_mvt.to_numpy())
    assert np.isfinite(yeni).all() and (yeni >= 60).all()
    name = G.next_name()
    path = G.OUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    h = G.md5(path)
    rep = [f"# Gönderim denetimi — {name}", "", f"- Dosya: `{path}`", f"- MD5: `{h}`",
           f"- Kaynak: `{kaynak}` (MD5 {G.md5(Path(kaynak))}) + Y1 kalibrasyonu{' (havalimanı + pist)' if pist else ''} (scripts/g_kalibre.py) + LOBT penceresi",
           f"- Not: {note}", "", "```", t.to_string(index=False), "```", "",
           f"- Değişim: RMS {np.sqrt(np.mean(d ** 2)):.2f} sn, ort {d.mean():+.2f} sn; LOBT kırpılan {int(km.sum())}"]
    (config.LOG_DIR / f"gonderim_{name}.md").write_text("\n".join(rep), encoding="utf-8")
    pd.DataFrame([{"tarih": time.strftime("%Y-%m-%d %H:%M"), "dosya": path.name, "md5": h, "aday": f"kalibre({Path(kaynak).name})",
                   "yeni_pay": 1.0, "lb_skor": "", "not": note}]).to_csv(G.DEFTER, mode="a",
                                                                       header=False, index=False, encoding="utf-8")
    print("\n".join(rep[:5]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    kuru = "kuru" in sys.argv[2:]
    pist = "pist=1" in sys.argv[2:]
    main(sys.argv[1], " ".join(a for a in sys.argv[2:] if a not in ("kuru", "pist=1")), kuru, pist)
