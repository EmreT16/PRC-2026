"""11.10 akşam: günlük kalkış öncesi gecikme kısıtı LIRF'te iki noktalı modelle.
LIRF'te gecikmesi uzun (d > 1 sa) uçuşlar ya tarifeye kilitli kaydedilir (y ≈ d,
günlük toplama katkı ≈ 0) ya da normal bekletilmiş (y ≈ τ, katkı ≈ d − τ). g_gunluk.py LIRF'i bu yüzden dışarıda bırakıyor.
Model (havalimanı-günü başına):
  uzun satırlar (d > 3.600 ve ŷ ≤ d): π_i = clip((ŷ_i − τ) / (d_i − τ), 0, 1) (önsel ortalama ŷ'ye eşit); 2^k bileşim (k ≤ 14; fazlası
    belirsizliğe göre ilk 14, kalanlar beklenen değerde) → R = T − ΣE[c_sıradan] − Σ_{kilitsiz} (d_i − τ),
    Var = V_sıradan + n_kilitsiz · s_τ² + σ_T²; sonsal P(kilit_i | T) → ŷ_i ← P_i · d_i + (1 − P_i) · τ.
  sıradan satırlar (sira=1): g_gunluk.py formülü, kalıntı = T − E_sonsal[uzun katkı] − ΣE[c_sıradan].
τ, s_τ: 2025 LIRF, d > 1 sa ve |y − d| > 60 satırlarının medyanı / std'si. σ_T = a T + b.
CV (2025 Oca / Şub / Haz / Tem; A = harman + düzeltici CV, B = LIRF üretim düzeni OOF'u): önceden tanımlı kabul — iç içe ay
seçimiyle LIRF SSE iyi ve en iyi ayarda ≥ 3/4 ay iyi.
Kullanım: python scripts/lirf_gunluk.py cv
          python scripts/lirf_gunluk.py uretim <kaynak gönderim> [sira=0|1] [a=… b=…] "<not>"   (PRC_FINAL=1 → final)
"""
import glob
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, validation as V  # noqa: E402

KMAX = 14
UZUN = 3600.0
IZGARA = [(s, a, b) for s in (0, 1) for a, b in ((0.003, 60.0), (0.01, 300.0), (0.03, 600.0))]
IZGARA2 = [(s, a, b) for s in (0, 1) for a, b in ((0.01, 300.0), (0.03, 600.0), (0.1, 1800.0))]  # ikinci tur (yalniz_b=1)
PARAM = config.ROOT / "kutuk" / "lirf_gunluk_param.json"


def gun_duzelt(p0, d, sv, gi_ok, T, tau, s_tau, a, b, sira, uz_ok=None):
    """Tek havalimanı-günü. p0, d, sv: satır dizileri; gi_ok: sıradan satırlarda düzeltilebilir mi; T (sn). Yeni tahmin döner."""
    uz = (d > UZUN) & (p0 <= d) & (d > tau + 60)
    if uz_ok is not None:  # 11.10 ikinci tur: iki noktalı model yalnız B satırlarında (A satırları NM'li; CV'de kötüleşti)
        uz &= uz_ok
    dh = d - p0
    z = dh / sv
    ec = sv * norm.pdf(z) + dh * norm.cdf(z)
    gi = norm.cdf(z)
    S_ord = ec[~uz].sum()
    V_ord = ((gi * sv) ** 2)[~uz].sum()
    sT2 = (a * T + b) ** 2
    yeni = p0.copy()
    iu = np.where(uz)[0]
    if len(iu):
        pi = np.clip((p0[iu] - tau) / (d[iu] - tau), 0.0, 1.0)
        katki = d[iu] - tau
        bel = pi * (1 - pi) * katki ** 2
        sec = iu[np.argsort(-bel)[:KMAX]] if len(iu) > KMAX else iu
        sabit = np.setdiff1d(iu, sec)
        ps = np.clip((p0[sec] - tau) / (d[sec] - tau), 0.0, 1.0)
        ks = d[sec] - tau
        pf = np.clip((p0[sabit] - tau) / (d[sabit] - tau), 0.0, 1.0)
        A_sabit = ((1 - pf) * (d[sabit] - tau)).sum()
        V_sabit = (pf * (1 - pf) * (d[sabit] - tau) ** 2).sum()
        k = len(sec)
        C = np.array(list(itertools.product((0, 1), repeat=k)), dtype=float)  # 1 = kilitli
        A = (1 - C) @ ks + A_sabit
        nk = (1 - C).sum(axis=1) + (1 - pf).sum()
        R = T - S_ord - A
        Var = V_ord + V_sabit + nk * s_tau ** 2 + sT2
        lp = (C * np.log(np.maximum(ps, 1e-12)) + (1 - C) * np.log(np.maximum(1 - ps, 1e-12))).sum(axis=1)
        ll = lp - 0.5 * R ** 2 / Var - 0.5 * np.log(Var)
        w = np.exp(ll - ll.max())
        w /= w.sum()
        P = w @ C
        yeni[sec] = P * d[sec] + (1 - P) * tau
        EA = w @ A
        VA = w @ (A - EA) ** 2
    else:
        EA, VA = 0.0, 0.0
    if sira:
        r = T - S_ord - EA
        g2 = np.where(~uz & gi_ok, gi, 0.0)
        yeni = np.where(~uz & gi_ok, np.maximum(60.0, p0 - np.clip(g2 * sv ** 2 * r / (V_ord + VA + sT2), -300, 300)), yeni)
    return np.maximum(60.0, yeni)


def uygula(df, T_seri, tau, s_tau, a, b, sira, yalniz_b=False):
    """df: MVT_ID_mvt, gun, p, d, sv, gi_ok (LIRF satırları). Gün gün düzeltir."""
    out = df.p.to_numpy(float).copy()
    for g, ix in df.groupby("gun").indices.items():
        T = T_seri.get(g, np.nan)
        if not np.isfinite(T):
            continue
        q = df.iloc[ix]
        out[ix] = gun_duzelt(q.p.to_numpy(float), q.d.to_numpy(float), q.sv.to_numpy(float), q.gi_ok.to_numpy(bool), T, tau, s_tau,
                             a, b, sira, (~q.gi_ok.to_numpy(bool)) if yalniz_b else None)
    return out


def lirf_b_oof():
    """LIRF B üretim düzeni OOF'u (b_alt_sinir_analiz.py ile aynı): fit_lirf çeyrek dışı + μ melez + ön koşullu v126 + 24 sa."""
    import pyarrow.parquet as pq
    import b_lirf_te as L
    import b_lirf_tarih_kaymasi as K
    from src import model_b as MB
    ref = L.lirf_ham()
    df = L.b_satirlari("train")
    feats = MB.features_v126b(pq.read_schema(config.DEP_TRAIN).names)
    y, gg = df[MB.TARGET_B].to_numpy(float), df[MB.DELAY].to_numpy(float)
    pre = np.asarray(MB.on_kosul_24s(df), bool)
    P = np.full((len(df), 3), np.nan)
    part1, part2 = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr, va = df[~va_m].reset_index(drop=True), df[va_m].reset_index(drop=True)
        ref_q = ref[~ref["ay"].isin(V.QUARTERS[q])]
        _, p, parts = L.fit_lirf(tr, va, feats, L.te_ozellik(tr, ref_q, oof=True), L.te_ozellik(va, ref_q, oof=False))
        P[va_m], part1[va_m], part2[va_m] = p, parts[1], parts[2]
    o = lambda f: df[["MVT_ID_mvt"]].merge(pd.read_parquet(config.DATA_DIR / "preds" / f), on="MVT_ID_mvt", how="left")  # noqa: E731
    M = o("BM_cv_tum_dparkktahkomgspcyte_cb_oof.parquet")
    mu = 0.5 * M["bilesen_mu"].to_numpy(float) + 0.5 * M["bilesen_mu_cb"].to_numpy(float)
    v126 = o("BM_cv_oof.parquet")["v126_B"].to_numpy(float)
    pred = np.where(pre, v126, np.clip(P[:, 0] * mu + P[:, 1] * part1 + P[:, 2] * part2, 60.0, None))
    df["gec"], df["gece"] = gg, gg > df["LOCAL_MINOFDAY"].to_numpy(float) * 60
    df["kayma"] = (y >= 80000) & (np.abs(y - gg) > 60)
    for q, _, va_m in V.quarter_folds(df["MVT_MONTH"]):
        tr = df[~va_m]
        pred[va_m], _ = K.uygula(df[va_m], pred[va_m], K.ESIK_SA, K.oran(tr, K.ESIK_SA), K.tau_hesap(tr))
    return pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt.to_numpy(), "pB": pred})


def cv():
    os.environ.setdefault("PRC_HARMAN", "kqgpPeT2")
    import gunluk_kisit_analiz as GK
    s_tab = json.loads((config.ROOT / "kutuk" / "gunluk_kisit_sigma.json").read_text(encoding="utf-8"))
    r = GK.veri()
    r = r[r.ADEP_mvt.eq("LIRF")].reset_index(drop=True)
    r = r.merge(lirf_b_oof(), on="MVT_ID_mvt", how="left")
    r["p"] = np.where(r.tur.eq("B") & r.pB.notna(), r.pB, r.p)
    y = r.TAXITIME_SEC_mvt.to_numpy(float)
    d = r.d.to_numpy(float)
    # τ, s_τ: 2025 LIRF tüm aylar, d > 1 sa ve kilitsiz
    fs = sorted(glob.glob(str(config.PRC2_DIR / "data" / "training_2025-*.parquet")))
    t = pd.concat([pd.read_parquet(f, columns=["ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])
                   for f in fs])
    t = t[t.ADEP_mvt.eq("LIRF") & t.PHASE_mvt.eq("DEP")]
    dd = (t.MVT_TIME_UTC_mvt - t.SCHED_TIME_UTC_mvt).dt.total_seconds()
    h = t.TAXITIME_SEC_mvt[(dd > UZUN) & ((t.TAXITIME_SEC_mvt - dd).abs() > 60) & (t.TAXITIME_SEC_mvt < 7200)]
    tau, s_tau = float(h.median()), float(h.std())
    kil = ((t.TAXITIME_SEC_mvt - dd).abs() <= 60)[dd > UZUN].mean()
    print(f"LIRF 2025: d > 1 sa {int((dd > UZUN).sum())} satır, kilitli payı {kil:.3f}; τ = {tau:.0f}, s_τ = {s_tau:.0f}")
    PARAM.write_text(json.dumps({"tau": tau, "s_tau": s_tau}, indent=1), encoding="utf-8")
    grp = np.where(r.tur.eq("B"), "LIRF|B", np.where(r.iz, "LIRF|Ai", "LIRF|Az"))
    r["sv"] = [s_tab[g] for g in grp]
    r["gi_ok"] = r.tur.ne("B").to_numpy()
    mt = r.MVT_TIME_UTC_mvt.dt.tz_localize(None) if r.MVT_TIME_UTC_mvt.dt.tz is not None else r.MVT_TIME_UTC_mvt
    r["gun"] = (mt - pd.to_timedelta(np.where(d > UZUN, tau, r.p.to_numpy(float)), unit="s")).dt.normalize()
    g = pd.read_csv(config.EK_VERI / "ansperf" / "all_pre_departure_delays_2025.csv")
    g = g[g.APT_ICAO.eq("LIRF")]
    T_seri = dict(zip(pd.to_datetime(g.FLT_DATE), g.DLY_ALL_PRE_2 * 60.0))
    ay, tur = r.ay.to_numpy(), r.tur.to_numpy()
    p0 = r.p.to_numpy(float)

    def olc(p):
        e = (p - y) ** 2
        return {**{f"ay{m}": np.sqrt(e[ay == m].mean()) for m in (1, 2, 6, 7)}, "LIRF": np.sqrt(e.mean()),
                "A": np.sqrt(e[tur == "A"].mean()), "B": np.sqrt(e[tur == "B"].mean())}, e
    sat, E = {}, {}
    sat["taban"], E["taban"] = olc(p0)
    yb = "yalniz_b=1" in sys.argv
    for s, a, b in (IZGARA2 if yb else IZGARA):
        k = f"sira{s} σT={a}T+{b:.0f}"
        sat[k], E[k] = olc(uygula(r, T_seri, tau, s_tau, a, b, s, yb))
    tb = pd.DataFrame(sat).T
    print(tb.round(2).to_string())
    ad = [k for k in E if k != "taban"]
    sse = 0.0
    for m in (1, 2, 6, 7):
        en = min(ad, key=lambda k: E[k][ay != m].sum())
        sse += E[en][ay == m].sum()
        print(f"  ay {m}: diğer aylarla seçilen {en}")
    ic = sse - E["taban"].sum()
    en = tb.loc[ad, "LIRF"].idxmin()
    iyi = int(sum(tb.loc[en, f"ay{m}"] < tb.loc["taban", f"ay{m}"] for m in (1, 2, 6, 7)))
    print(f"iç içe ay seçimi LIRF SSE {ic / 1e6:+.1f} M (test ölçeğinde ≈ {ic / 2 / 344841:+.1f} sn² MSE) | en iyi {en}, iyi ay {iyi}/4 → "
          f"{'KABUL' if ic < 0 and iyi >= 3 else 'RET'}")


def uretim(kaynak, note, opt):
    import g_gonderim as G
    s_tab = json.loads((config.ROOT / "kutuk" / "gunluk_kisit_sigma.json").read_text(encoding="utf-8"))
    pr = json.loads(PARAM.read_text(encoding="utf-8"))
    a, b, sira = float(opt.get("a", 0.003)), float(opt.get("b", 60.0)), int(opt.get("sira", 0))
    sub = pd.read_parquet(kaynak)
    r = pd.read_parquet(config.RANK_RAW, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    x = sub.merge(r, on="MVT_ID_mvt", how="left")
    x = x.merge(pd.read_parquet(config.DEP_RANK, columns=["MVT_ID_mvt", "NM_TAXI_DIFF_SEC"]), on="MVT_ID_mvt", how="left")
    x = x.merge(pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{opt.get('a_dosya', 'D_kqgpPeT2_adsb_42p9T')}.parquet",
                                columns=["MVT_ID_mvt", "rhat"]), on="MVT_ID_mvt", how="left")
    L = x.ADEP_mvt.eq("LIRF").to_numpy()
    q = x[L].copy()
    B = q.NM_TAXI_DIFF_SEC.isna().to_numpy()
    iz = (q.rhat.fillna(0).abs() > 0).to_numpy() & ~B
    q["sv"] = [s_tab[g] for g in np.where(B, "LIRF|B", np.where(iz, "LIRF|Ai", "LIRF|Az"))]
    q["gi_ok"] = ~B
    mt = pd.to_datetime(q.MVT_TIME_UTC_mvt, utc=True).dt.tz_localize(None)
    q["d"] = (mt - pd.to_datetime(q.SCHED_TIME_UTC_mvt, utc=True).dt.tz_localize(None)).dt.total_seconds().to_numpy(float)
    q["p"] = q.TAXITIME_SEC_mvt.to_numpy(float)
    q["gun"] = (mt - pd.to_timedelta(np.where(q.d > UZUN, pr["tau"], q.p), unit="s")).dt.normalize()
    g = pd.read_csv(config.EK_VERI / "ansperf" / "all_pre_departure_delays_2026.csv")
    g = g[g.APT_ICAO.eq("LIRF")]
    T_seri = dict(zip(pd.to_datetime(g.FLT_DATE), g.DLY_ALL_PRE_2 * 60.0))
    yeni_l = uygula(q.reset_index(drop=True), T_seri, pr["tau"], pr["s_tau"], a, b, sira, opt.get("yalniz_b") == "1")
    t0 = sub.TAXITIME_SEC_mvt.to_numpy(float)
    yeni = t0.copy()
    yeni[L] = yeni_l
    yeni, km = G.lobt_kirp(sub.MVT_ID_mvt.to_numpy(), yeni)
    dlt = yeni - t0
    print(f"LIRF {L.sum():,} satır (B {B.sum()}), uzun (d > 1 sa) {int((q.d > UZUN).sum())}; değişen {(np.abs(dlt) > 1e-6).sum():,}, "
          f"RMS (LIRF) {np.sqrt(np.mean(dlt[L] ** 2)):.1f}, maks {np.abs(dlt).max():.0f}; LOBT kırpılan {int(km.sum())}")
    out = pd.DataFrame({"MVT_ID_mvt": sub.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": yeni.astype(np.float32)})
    tpl = pd.read_parquet(config.SUBMIT_TEMPLATE, columns=["MVT_ID_mvt"])
    assert len(out) == G.N_ROWS and np.array_equal(out.MVT_ID_mvt.to_numpy(), tpl.MVT_ID_mvt.to_numpy())
    assert np.isfinite(yeni).all() and (yeni >= 60).all()
    name = G.next_name()
    path = G.OUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    h = G.md5(path)
    rep = [f"# Gönderim denetimi — {name}", "", f"- Dosya: `{path}`", f"- MD5: `{h}`",
           f"- Kaynak: `{kaynak}` (MD5 {G.md5(Path(kaynak))}) + LIRF günlük kısıt (scripts/lirf_gunluk.py; sira={sira}, σT={a}T+{b:.0f})",
           f"- Not: {note}", "", f"- LIRF değişim RMS {np.sqrt(np.mean(dlt[L] ** 2)):.1f} sn, maks {np.abs(dlt).max():.0f} sn"]
    (config.LOG_DIR / f"gonderim_{name}.md").write_text("\n".join(rep), encoding="utf-8")
    pd.DataFrame([{"tarih": time.strftime("%Y-%m-%d %H:%M"), "dosya": path.name, "md5": h, "aday": f"lirf_gunluk({Path(kaynak).name})",
                   "yeni_pay": 1.0, "lb_skor": "", "not": note}]).to_csv(G.DEFTER, mode="a", header=False, index=False, encoding="utf-8")
    print("\n".join(rep[:5]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    poz = [a for a in sys.argv[1:] if "=" not in a]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    cv() if poz[0] == "cv" else uretim(poz[1], poz[2] if len(poz) > 2 else "", opt)
