"""11.10: NM'siz (Model B) kalkışların NM uçuşu, varış havalimanı da 10
havalimanından biriyse oradaki varış kaydından bulunabiliyor (uçuş no + kalkış / varış havalimanı + varış zamanı ≤ 8 sa). Organizatör
NM uçuşunu yalnız |off-block − LOBT| ≤ 3.606 sn ise eşleştiriyor; bu satırlar eşleşmediği için gerçek taksi pencerenin DIŞINDA
(2025'te eşleşen 908 satırın yalnız %2,5'i içeride). Kural: tahmin (kalkış − LOBT ± 3.606) aralığındaysa en yakın sınıra taşınır.
Havalimanları önceden LIRF, LTFM (NM saat düzeni farklı).
Ölçüm (2025, 12 ay): LIRF = üretim düzeni OOF'u (fit_lirf çeyrek dışı + μ melez + ön koşullu v126 + 24 sa karışımı;
b_alt_sinir_analiz.py ile aynı kurulum), LTFM = BM_cv_tum_guncel_oof. Önceden tanımlı kabul: toplam SSE iyi ve ≥ 3/4 çeyrek iyi.
Kullanım: python scripts/b_yetim_pencere.py cv
          python scripts/b_yetim_pencere.py uretim <kaynak gönderim> "<not>"   (PRC_FINAL=1 → final; yeni gönderim dosyası yazar)
"""
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config, validation as V  # noqa: E402

APT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
KURAL_APT = ["LIRF", "LTFM"]
PEN = 3606.0
C = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"]


def eslestir(d):
    """d: ham satırlar (DEP + ARR). Dönüş: NM'siz, varışı 10 havalimanında olan kalkışlar + varış kaydından LOBT (LOBT_v)."""
    dep = d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.isin(APT) & d.LOBT_flt.isna() & d.ADES_mvt.isin(APT)].copy()
    arr = d[d.PHASE_mvt.eq("ARR") & d.LOBT_flt.notna()][["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"]].rename(
        columns={"MVT_TIME_UTC_mvt": "ARR_T", "LOBT_flt": "LOBT_v"})
    m = dep.merge(arr, on=["FLIGHT_mvt", "ADEP_mvt", "ADES_mvt"], how="inner")
    dt = (m.ARR_T - m.MVT_TIME_UTC_mvt).dt.total_seconds()
    m = m[(dt > 0) & (dt < 8 * 3600)].sort_values("ARR_T").drop_duplicates("MVT_ID_mvt")
    m["g"] = (m.MVT_TIME_UTC_mvt - m.LOBT_v).dt.total_seconds()
    return m


def uygula(p, g):
    alt, ust = g - PEN, g + PEN
    ic = (p > alt) & (p < ust)
    return np.maximum(60.0, np.where(ic, np.where(np.abs(p - alt) < np.abs(p - ust), alt, ust), p)), ic


def cv():
    import b_lirf_te as L
    import b_lirf_tarih_kaymasi as K
    import pyarrow.parquet as pq
    from src import model_b as MB
    fs = sorted(glob.glob(str(config.PRC2_DIR / "data" / "training_2025-*.parquet")))
    d = pd.concat([pd.read_parquet(f, columns=C + ["TAXITIME_SEC_mvt"]) for f in fs])
    m = eslestir(d)
    m = m[m.ADEP_mvt.isin(KURAL_APT)]
    # LIRF üretim OOF'u (b_alt_sinir_analiz.py ile aynı)
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
    lirf = pd.DataFrame({"MVT_ID_mvt": df.MVT_ID_mvt.to_numpy(), "p_lirf": pred})
    ltfm = pd.read_parquet(config.DATA_DIR / "preds" / "BM_cv_tum_guncel_oof.parquet").rename(columns={"tum_mu_yt_tavan3500": "p_g"})
    m = m.merge(lirf, on="MVT_ID_mvt", how="left").merge(ltfm, on="MVT_ID_mvt", how="left")
    m["p"] = np.where(m.ADEP_mvt.eq("LIRF"), m.p_lirf, m.p_g)
    assert m.p.notna().all()
    p, yy = m.p.to_numpy(float), m.TAXITIME_SEC_mvt.to_numpy(float)
    yeni, ic = uygula(p, m.g.to_numpy(float))
    e0, e1 = (p - yy) ** 2, (yeni - yy) ** 2
    m["f"], m["ic"], m["ay"] = (e1 - e0) / 1e6, ic, m.MVT_TIME_UTC_mvt.dt.month
    print(f"eşleşen (LIRF / LTFM) {len(m)}; gerçek pencerede {np.mean((yy > m.g - PEN) & (yy < m.g + PEN)):.3f}; tahmin yasak aralıkta {ic.sum()}")
    print("havalimanı:", m.groupby("ADEP_mvt").agg(n=("ic", "size"), ic=("ic", "sum"), sse=("f", "sum")).round(1).to_dict("index"))
    qs = {q: m.f[m.ay.isin(ms)].sum() for q, ms in V.QUARTERS.items()}
    print("çeyrek SSE farkı (M):", {k: round(v, 1) for k, v in qs.items()})
    iyi = sum(v < 0 for v in qs.values())
    top = m.f.sum()
    print(f"toplam {top:+.1f} M, iyileşen {(e1 < e0).sum()}, kötüleşen {(e1 > e0).sum()}, iyi çeyrek {iyi}/4 → {'KABUL' if top < 0 and iyi >= 3 else 'RET'}")
    print(f"test ölçeğinde (12 ay → 2 ay) ≈ {top * 1e6 / 6 / 344841:+.1f} sn² MSE")


def uretim(kaynak, note):
    import g_gonderim as G
    r = pd.read_parquet(config.RANK_RAW, columns=C)
    m = eslestir(r)
    m = m[m.ADEP_mvt.isin(KURAL_APT)]
    sub = pd.read_parquet(kaynak)
    x = sub.merge(m[["MVT_ID_mvt", "g"]], on="MVT_ID_mvt", how="left")
    t0 = x.TAXITIME_SEC_mvt.to_numpy(float)
    has = x.g.notna().to_numpy()
    yeni = t0.copy()
    yeni[has], ic = uygula(t0[has], x.g.to_numpy(float)[has])
    d = yeni - t0
    print(f"eşleşen yetim B (LIRF / LTFM) {has.sum()} | tahmin yasak aralıkta {ic.sum()} | değişim RMS (değişenler) "
          f"{np.sqrt(np.mean(d[np.abs(d) > 0] ** 2)) if (np.abs(d) > 0).any() else 0:.0f}, maks {np.abs(d).max():.0f}")
    out = pd.DataFrame({"MVT_ID_mvt": sub.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": yeni.astype(np.float32)})
    tpl = pd.read_parquet(config.SUBMIT_TEMPLATE, columns=["MVT_ID_mvt"])
    assert len(out) == G.N_ROWS and np.array_equal(out.MVT_ID_mvt.to_numpy(), tpl.MVT_ID_mvt.to_numpy())
    assert np.isfinite(yeni).all() and (yeni >= 60).all()
    name = G.next_name()
    path = G.OUT_DIR / f"{name}.parquet"
    out.to_parquet(path, index=False)
    h = G.md5(path)
    rep = [f"# Gönderim denetimi — {name}", "", f"- Dosya: `{path}`", f"- MD5: `{h}`",
           f"- Kaynak: `{kaynak}` (MD5 {G.md5(Path(kaynak))}) + yetim B penceresi (scripts/b_yetim_pencere.py; {','.join(KURAL_APT)})",
           f"- Not: {note}", "", f"- Değişen satır {int((np.abs(d) > 0).sum())}, maks |fark| {np.abs(d).max():.0f} sn"]
    (config.LOG_DIR / f"gonderim_{name}.md").write_text("\n".join(rep), encoding="utf-8")
    pd.DataFrame([{"tarih": time.strftime("%Y-%m-%d %H:%M"), "dosya": path.name, "md5": h, "aday": f"yetim({Path(kaynak).name})",
                   "yeni_pay": 1.0, "lb_skor": "", "not": note}]).to_csv(G.DEFTER, mode="a", header=False, index=False, encoding="utf-8")
    print("\n".join(rep[:5]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cv() if sys.argv[1] == "cv" else uretim(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
