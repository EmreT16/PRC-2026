"""G — Gönderim dosyası ve G3 denetimi.

Kullanım: python scripts/g_gonderim.py <aday> [yeni_pay=1.0] [b=<ad>] [not]   ör.  python scripts/g_gonderim.py bes_bilesen
Girdi: data/preds/TEST_<aday>.parquet ve v126 gönderimi (PRC_4, salt okunur).
Kural (29.09): Model A yalnızca bu deponun modellerinden (yeni_pay = 1,0, v126 ile harman yok);
Model B v126 ile birebir aynı; b=<ad> verilirse yalnızca data/preds/TEST_B_<ad>.parquet'teki satırlar o dosyanın
taksisiyle değişir (29.09 akşam, Model B çalışması; CV'de kanıtlı değişiklik). yeni_pay < 1 yalnızca
elle istenirse.
Model A: taksi = (1 − w)·v126 + w·yeni, sonra 60 sn alt sınır.
lobt=1 (05.10): taksi [MVT − LOBT − 3606, MVT − LOBT + 3606] aralığına kırpılır
(2025'te LOBT'li 2.062.577 kalkışın tamamında |BLOCK − LOBT| ≤ 3606 sn; doğru değer aralıkta olduğundan kırpma hatayı artıramaz).
Çıktı: gonderim/strong-moose_v<N>.parquet (N otomatik artar; v145'ten başlar), kutuk/gonderim_<dosya>.md (denetim), kutuk/skor_defteri.csv'ye satır (LB skoru boş).
LB'ye yükleme elle yapılır.
"""
import hashlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

V126 = config.PRC4_DIR / "submissions" / "strong-moose_v126.parquet"
V126_MD5 = "486e03b2ef36eb9cc9fe559fa12ff0ff"
OUT_DIR = config.ROOT / ("gonderim_final" if config.FINAL else "gonderim")  # 08.10: final adayları ayrı klasörde
ROW_24H = 213147416  # LIRF test satırı; girdide MVT − AOBT_3 = 86.335 sn (24 sa kayık NM kaydı). Tahmini yalnız raporlanır
                    # (üst kırpma olmadığının denetimi); satır hiçbir yerde değiştirilmez.
N_ROWS = config.TEST_N
DEFTER = config.LOG_DIR / ("skor_defteri_final.csv" if config.FINAL else "skor_defteri.csv")
SON_ESKI_SURUM = 144   # önceki kod tabanlarındaki son gönderim numarası (29.09); bu depo v145'ten başlar


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def next_name():
    """Ekibin isim düzeni: strong-moose_v<N>; N, gonderim/ klasöründeki en büyük numaradan (en az 144) bir fazla.
    Final modunda (PRC_FINAL=1) adaylar gonderim_final/strong-moose_final_a<N>.parquet; yüklenecek tek dosya sonunda
    strong-moose_final.parquet adıyla kopyalanır."""
    if config.FINAL:
        n = [int(m.group(1)) for f in OUT_DIR.glob("strong-moose_final_a*.parquet")
             if (m := re.fullmatch(r"strong-moose_final_a(\d+)\.parquet", f.name))]
        return f"strong-moose_final_a{max([0] + n) + 1}"
    nums = [int(m.group(1)) for f in OUT_DIR.glob("strong-moose_v*.parquet")
            if (m := re.fullmatch(r"strong-moose_v(\d+)\.parquet", f.name))]
    return f"strong-moose_v{max([SON_ESKI_SURUM] + nums) + 1}"


HAM_RANK = config.RANK_RAW
LOBT_PENCERE = 3606.0


def lobt_kirp(mvt_id, taksi):
    """Taksiyi LOBT penceresine kırpar; (yeni taksi, değişen satır maskesi)."""
    r = pd.read_parquet(HAM_RANK, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"])
    r = pd.DataFrame({"MVT_ID_mvt": mvt_id}).merge(r, on="MVT_ID_mvt", how="left", validate="one_to_one")
    e0 = pd.Timestamp("1970-01-01", tz="UTC")
    g = ((pd.to_datetime(r.MVT_TIME_UTC_mvt, utc=True) - e0).dt.total_seconds()
         - (pd.to_datetime(r.LOBT_flt, utc=True) - e0).dt.total_seconds()).to_numpy(float)
    ok = np.isfinite(g)
    yeni = taksi.copy()
    yeni[ok] = np.clip(taksi[ok], np.maximum(60.0, g[ok] - LOBT_PENCERE), np.maximum(60.0, g[ok] + LOBT_PENCERE))
    return yeni, np.abs(yeni - taksi) > 1e-6


def build(aday, w, note="", b=None, lobt=False):
    assert 0 < w <= 1
    tpl = pd.read_parquet(config.SUBMIT_TEMPLATE, columns=["MVT_ID_mvt"])
    if config.FINAL:  # 08.10: final testinde v126 yok; her satır Model A ya da Model B dosyasından gelmeli
        assert w == 1.0 and b is not None, "final modunda yeni_pay = 1 ve b= zorunlu"
        v126 = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": np.nan})
    else:
        assert md5(V126) == V126_MD5, "v126 dosyası değişmiş"
        v126 = pd.read_parquet(V126)
    assert np.array_equal(v126["MVT_ID_mvt"].to_numpy(), tpl["MVT_ID_mvt"].to_numpy())
    new = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_{aday}.parquet")
    d = tpl.merge(new[["MVT_ID_mvt", "taksi"]], on="MVT_ID_mvt", how="left", validate="one_to_one")
    a = d["taksi"].notna().to_numpy()
    assert a.sum() == config.TEST_A_N, a.sum()
    old = v126["TAXITIME_SEC_mvt"].to_numpy(float)
    out = old.copy()
    out[a] = np.maximum(60.0, d.loc[a, "taksi"].to_numpy(float) if w == 1.0 else (1 - w) * old[a] + w * d.loc[a, "taksi"].to_numpy(float))
    bm = np.zeros(len(tpl), bool)
    if b is not None:
        nb = pd.read_parquet(config.DATA_DIR / "preds" / f"TEST_B_{b}.parquet", columns=["MVT_ID_mvt", "taksi"])
        db = tpl.merge(nb, on="MVT_ID_mvt", how="left", validate="one_to_one")
        bm = db["taksi"].notna().to_numpy()
        assert bm.sum() == len(nb) and not (bm & a).any(), "Model B dosyası eşleşmiyor ya da Model A satırı içeriyor"
        out[bm] = np.maximum(60.0, db.loc[bm, "taksi"].to_numpy(float))
    km = np.zeros(len(tpl), bool)
    if lobt:
        out, km = lobt_kirp(tpl["MVT_ID_mvt"].to_numpy(), out)
    sub = pd.DataFrame({"MVT_ID_mvt": tpl["MVT_ID_mvt"].to_numpy(), "TAXITIME_SEC_mvt": out.astype(np.float32)})

    # G3 denetimi
    t = sub["TAXITIME_SEC_mvt"].to_numpy(float)
    checks = {
        "satir_sayisi": len(sub) == N_ROWS,
        "sutunlar": list(sub.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"],
        "kimlik_sirasi": np.array_equal(sub["MVT_ID_mvt"].to_numpy(), tpl["MVT_ID_mvt"].to_numpy()),
        "nan_inf_yok": bool(np.isfinite(t).all()),
        "alt_sinir_60": bool((t >= 60).all()),
    }
    if config.FINAL:
        checks["tum_satirlar_a_ya_da_b"] = bool((a | bm).all())
    else:
        checks["model_b_ayni" if b is None else "model_b_kalan_ayni"] = bool(
            np.array_equal(t[~a & ~bm & ~km], old[~a & ~bm & ~km].astype(np.float32).astype(float)))
    if b is not None:
        checks["model_b_yeni_uygulandi"] = bool(np.allclose(t[bm], out[bm].astype(np.float32).astype(float)))
    row = sub["MVT_ID_mvt"].to_numpy() == ROW_24H
    info = {
        "model_a_rms_fark": float(np.sqrt(np.mean((t[a] - old[a]) ** 2))),
        "model_a_ort_fark": float(np.mean(t[a] - old[a])),
        "yeni_modelin_v126dan_rms_farki": float(np.sqrt(np.mean((d.loc[a, "taksi"].to_numpy(float) - old[a]) ** 2))),
        "40000_ustu": int((t > 40_000).sum()), "40000_ustu_v126": int((old > 40_000).sum()),
        "satir_24s": float(t[row][0]), "satir_24s_v126": float(old[row][0]),
        "satir_24s_yeni": float(d.loc[row, "taksi"].iloc[0]),
        "lobt_kirpilan_satir": int(km.sum()), "lobt_kirpilan_model_b_satir": int((km & ~a).sum()),
    }
    if b is not None:
        info.update({"model_b_degisen_satir": int(bm.sum()), "model_b_rms_fark": float(np.sqrt(np.mean((t[bm] - old[bm]) ** 2))),
                     "model_b_ort_fark": float(np.mean(t[bm] - old[bm])), "model_b_ort_v126": float(np.mean(old[bm])),
                     "model_b_ort_yeni": float(np.mean(t[bm]))})
    assert all(checks.values()), checks
    OUT_DIR.mkdir(exist_ok=True)
    name = next_name()
    path = OUT_DIR / f"{name}.parquet"
    sub.to_parquet(path, index=False)
    h = md5(path)
    rep = [f"# Gönderim denetimi — {name}", "", f"- Dosya: `{path}`", f"- MD5: `{h}`",
           (f"- Aday: `{aday}`; final testi (final_ranking.parquet, {N_ROWS:,} satır), taban yok" if config.FINAL else
            f"- Aday: `{aday}`, yeni modelin payı: {w:.2f}; taban v126 (MD5 {V126_MD5})"),
           (f"- Model B: TEST_B_{b} ({int(bm.sum())} satır)" if config.FINAL else
            f"- Model B: {'v126 ile aynı' if b is None else f'TEST_B_{b} satırlarında yeni ({int(bm.sum())} satır), diğerleri v126'}"),
           f"- Not: {note}", "",
           "| Denetim | Sonuç |", "|---|---|"] + [f"| {k} | {'✓' if v else '✗'} |" for k, v in checks.items()]
    rep += ["", "| Bilgi | Değer |", "|---|---:|"] + [f"| {k} | {v:,.2f} |" for k, v in info.items()]
    (config.LOG_DIR / f"gonderim_{name}.md").write_text("\n".join(rep), encoding="utf-8")
    line = pd.DataFrame([{"tarih": time.strftime("%Y-%m-%d %H:%M"), "dosya": path.name, "md5": h,
                          "aday": aday if b is None else f"{aday}+B_{b}",
                          "yeni_pay": w, "lb_skor": "", "not": note}])
    line.to_csv(DEFTER, mode="a", header=not DEFTER.exists(), index=False, encoding="utf-8")
    print("\n".join(rep))
    return path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = [x for x in sys.argv[1:] if not x.startswith(("b=", "lobt="))]
    bname = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("b=")), None)
    lobt = any(x == "lobt=1" for x in sys.argv[1:])
    build(args[0], float(args[1]) if len(args) > 1 else 1.0, " ".join(args[2:]), bname, lobt)
