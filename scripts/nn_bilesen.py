"""Sinir ağı bileşeni (30.09): gömmeli kategorik + sayısal girdili MLP; Model A harmanına ikinci model ailesi.

Neden: harmanda ağaç modelleri birbirine çok
benziyor (korelasyon ~0,99).
Girdi: F.prepare_split / prepare_full (PRC_KUYRUK=1 ise kuyruk özellikleri dahil); çerçeve öteki bileşenlerle aynı.
  sayısal: feature_list − kategorikler; eksik → eğitim ortancası + eksiklik bayrağı; QuantileTransformer (normal).
  kategorik: gcb.cats("ek5") (ADEP + v126'nın 5'i + 5 ek); eğitimde ≥ 20 kez geçenler, gerisi 0 (bilinmeyen).
Hedef: ofset (BLOCK − AOBT_3) ±7.200'de kırpık, 600'e bölünmüş; kayıp Huber (δ = 1.000 sn). Sabit epok (erken durdurma yok).
Kullanım: python scripts/nn_bilesen.py <ad> split_1 [split_2 split_3]   → data/preds/NN_<ad>_<split>.parquet (sütun: nn)
          python scripts/nn_bilesen.py <ad> tam                         → data/preds/TEST_nn_<ad>.parquet
Seçenekler: epok=6 tohum=42,7 gizli=512,256,128 model=mlp|tabm (tabm: yandex-research TabM, ICLR 2025, Apache-2.0;
            parçalı doğrusal sayısal gömme, k=32 paylaşımlı üye; kategorikler one-hot, ≥ 50 kez geçenler)
            tabm_k=32 tabm_d=512 amp=0 bs=2048 (09.10: 4 GB'lık kartta tabm_k=8 amp=1)
            hedef=taksi (10.10): doğrudan taksi hedefli eğitim, çıktı yine ofset (nn = NM − taksi tahmini); ad '_tx' içermeli
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, features_v126 as F, gcb as G, validation as V  # noqa: E402

PRED = config.DATA_DIR / "preds"
OLCEK, DELTA, KIRP = 600.0, 1000.0, 7200.0
MIN_SAYI = 20


def hazirla(tr, ap, feats, min_sayi=MIN_SAYI, cats=None):
    """(Xn_tr, Xc_tr, Xn_ap, Xc_ap, kardinaliteler). cats (10.10): kategorik sütunlar dışarıdan (Model B); None → Model A'nınki."""
    cats = G.cats("ek5") if cats is None else [c for c in cats if c in feats]
    num = [c for c in feats if c not in cats]
    A = tr[num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    B = ap[num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    A[~np.isfinite(A)], B[~np.isfinite(B)] = np.nan, np.nan
    eksik = np.isnan(A).mean(axis=0) > 0.01
    med = np.nanmedian(A, axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    ma, mb = np.isnan(A[:, eksik]), np.isnan(B[:, eksik])
    A, B = np.where(np.isnan(A), med, A), np.where(np.isnan(B), med, B)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=300_000, random_state=0)
    A = qt.fit_transform(A).astype(np.float32)
    B = qt.transform(B).astype(np.float32)
    A = np.hstack([A, ma.astype(np.float32)])
    B = np.hstack([B, mb.astype(np.float32)])
    Ca, Cb, kard = [], [], []
    for c in cats:
        sa = tr[c].fillna("nan").astype(str)
        vc = sa.value_counts()
        sozluk = {v: i + 1 for i, v in enumerate(vc.index[vc >= min_sayi])}
        Ca.append(sa.map(sozluk).fillna(0).to_numpy(np.int64))
        Cb.append(ap[c].fillna("nan").astype(str).map(sozluk).fillna(0).to_numpy(np.int64))
        kard.append(len(sozluk) + 1)
    return A, np.column_stack(Ca), B, np.column_stack(Cb), kard


class Ag(nn.Module):
    def __init__(self, n_num, kard, gizli):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(k, min(24, (k + 1) // 2 + 1)) for k in kard])
        d = n_num + sum(e.embedding_dim for e in self.emb)
        katman = []
        for h in gizli:
            katman += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(0.1)]
            d = h
        katman.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*katman)

    def forward(self, xn, xc):
        e = [m(xc[:, i]) for i, m in enumerate(self.emb)]
        return self.mlp(torch.cat([xn] + e, dim=1)).squeeze(1)


def tabm_modeli(A, kard, k=32, d_block=512):
    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
    from tabm import TabM
    i = np.random.default_rng(0).choice(len(A), size=min(len(A), 200_000), replace=False)
    bins = compute_bins(torch.from_numpy(A[i]), n_bins=32)
    return TabM.make(n_num_features=A.shape[1], cat_cardinalities=kard, d_out=1, k=k, d_block=d_block,
                     num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=12, activation=False, version="B"))


def egit_tahmin(A, Ca, y, B, Cb, kard, epok, tohum, gizli, cihaz, tur="mlp", tabm_k=32, tabm_d=512, amp=False, bs=2048,
                delta=DELTA, agirlik=None):
    # 09.10 akşam: tabm_k / tabm_d / amp — varsayılan TabM (k = 32, 512) 4 GB'lık kartta belleği doldurup paylaşılan
    # belleğe taşıyor (adım başına 1,9 sn); k = 8 + fp16 karışık duyarlık kartta kalıyor. Varsayılanlar eski davranış.
    # 10.10: B None → eğitilmiş model döner (TabMReg); delta: Huber δ (sn; düzelticide 600).
    # 11.10: agirlik — satır ağırlıklı Huber (B μ'de B satırları; None → eski hesap, birebir).
    torch.manual_seed(tohum)
    np.random.seed(tohum)
    model = (tabm_modeli(A, kard, tabm_k, tabm_d) if tur == "tabm" else Ag(A.shape[1], kard, gizli)).to(cihaz)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    adim = int(np.ceil(len(A) / bs)) * epok
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=adim, pct_start=0.2)
    kayip = nn.HuberLoss(delta=delta / OLCEK)
    amp = amp and cihaz == "cuda"
    olcek = torch.amp.GradScaler("cuda", enabled=amp)
    At, Ct = torch.from_numpy(A), torch.from_numpy(Ca)
    yt = torch.from_numpy((np.clip(y, -KIRP, KIRP) / OLCEK).astype(np.float32))
    if agirlik is not None:
        Wt, kayip_n = torch.from_numpy(np.asarray(agirlik, np.float32)), nn.HuberLoss(delta=delta / OLCEK, reduction="none")
    for ep in range(epok):
        model.train()
        sira = torch.randperm(len(A))
        top, t0 = 0.0, time.time()
        for k in range(0, len(A), bs):
            i = sira[k:k + bs]
            xb, cb, yb = At[i].to(cihaz, non_blocking=True), Ct[i].to(cihaz, non_blocking=True), yt[i].to(cihaz)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                cikti = model(xb, cb)
            cikti = cikti.float()
            if agirlik is None:
                l = kayip(cikti.squeeze(-1), yb.unsqueeze(1).expand(-1, cikti.shape[1])) if tur == "tabm" else kayip(cikti, yb)
            else:
                wb = Wt[i].to(cihaz)
                le = kayip_n(cikti.squeeze(-1), yb.unsqueeze(1).expand(-1, cikti.shape[1])) if tur == "tabm" else kayip_n(cikti, yb)
                l = (le * (wb[:, None] if le.dim() == 2 else wb)).sum() / (wb.sum() * (le.shape[1] if le.dim() == 2 else 1))
            olcek.scale(l).backward()
            olcek.step(opt)
            olcek.update()
            sch.step()
            top += float(l) * len(i)
        print(f"    epok {ep + 1}/{epok}: kayıp {top / len(A):.4f} | {time.time() - t0:.0f} sn", flush=True)
    model.eval()
    if B is None:
        return model
    return model_tahmin(model, B, Cb, tur, amp, cihaz)


def model_tahmin(model, B, Cb, tur, amp, cihaz):
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp and cihaz == "cuda"):
        for k in range(0, len(B), 16384):
            o = model(torch.from_numpy(B[k:k + 16384]).to(cihaz), torch.from_numpy(Cb[k:k + 16384]).to(cihaz)).float()
            out.append((o.mean(dim=1).squeeze(-1) if tur == "tabm" else o).cpu().numpy())
    return np.concatenate(out) * OLCEK


class TabMReg:
    """10.10 (Model A düzelticisine ikinci aile; scripts/adsb_duzeltici.py model=lgb+nn / nn): sonradan tahmin edilebilen
    TabM. Girdi DataFrame; kategorik (category dtype) sütunlar gömme, ötekiler hazirla() ile aynı ön işleme (eksik → ortanca +
    eksiklik bayrağı, QuantileTransformer). Ön işleme durumu saklanır; predict yeni çerçeveye aynısını uygular."""
    tabm = True

    def __init__(self, epok=8, tohumlar=(42,), delta=DELTA, tabm_k=8, amp=True, bs=2048, min_sayi=50, tabm_d=512):
        self.epok, self.tohumlar, self.delta = epok, list(tohumlar), delta
        self.tabm_k, self.amp, self.bs, self.min_sayi, self.tabm_d = tabm_k, amp, bs, min_sayi, tabm_d
        self.cihaz = "cuda" if torch.cuda.is_available() else "cpu"

    def _donustur(self, X):
        A = np.array(X[self.num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32), copy=True)
        A[~np.isfinite(A)] = np.nan
        ma = np.isnan(A[:, self.eksik])
        A = np.where(np.isnan(A), self.med, A)
        A = np.hstack([self.qt.transform(A).astype(np.float32), ma.astype(np.float32)])[:, self.tut]
        C = (np.column_stack([X[c].astype(object).fillna("nan").astype(str).map(s).fillna(0).to_numpy(np.int64)
                              for c, s in zip(self.cats, self.sozluk)]) if self.cats else np.zeros((len(A), 1), np.int64))
        return np.ascontiguousarray(A), C

    def fit(self, X, y):
        t0 = time.time()
        self.cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)] or []
        self.num = [c for c in X.columns if c not in self.cats]
        A = np.array(X[self.num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32), copy=True)
        A[~np.isfinite(A)] = np.nan
        self.eksik = np.isnan(A).mean(axis=0) > 0.01
        med = np.nanmedian(A, axis=0)
        self.med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=300_000, random_state=0)
        self.qt.fit(np.where(np.isnan(A), self.med, A))
        self.tut = np.ones(len(self.num) + int(self.eksik.sum()), bool)
        self.sozluk = []
        for c in self.cats:
            vc = X[c].astype(object).fillna("nan").astype(str).value_counts()
            self.sozluk.append({v: i + 1 for i, v in enumerate(vc.index[vc >= self.min_sayi])})
        del A
        A, C = self._donustur(X)
        i = np.random.default_rng(0).choice(len(A), size=min(len(A), 200_000), replace=False)
        self.tut = np.array([len(np.unique(A[i, j])) > 1 for j in range(A.shape[1])])
        A = np.ascontiguousarray(A[:, self.tut])
        if not self.cats:  # TabM kategorisiz de çalışsın: tek sabit kategori
            C = np.zeros((len(A), 1), np.int64)
        self.kard = [len(s) + 1 for s in self.sozluk] or [1]
        print(f"  TabMReg: {len(A):,} satır, {A.shape[1]} sayısal, {len(self.cats)} kategorik | hazırlık {time.time() - t0:.0f} sn",
              flush=True)
        y = np.asarray(y, float)
        self.modeller = [egit_tahmin(A, C, y, None, None, self.kard, self.epok, s, None, self.cihaz, "tabm",
                                     tabm_k=self.tabm_k, tabm_d=self.tabm_d, amp=self.amp, bs=self.bs, delta=self.delta)
                         for s in self.tohumlar]
        return self

    def predict(self, X):
        A, C = self._donustur(X)
        if not self.cats:
            C = np.zeros((len(A), 1), np.int64)
        return np.mean([model_tahmin(m, A, C, "tabm", self.amp, self.cihaz) for m in self.modeller], axis=0)


def calis(ad, tr, ap, opt):
    y = tr[config.TARGET].to_numpy(float)
    if opt.get("hedef") == "taksi":  # 10.10: doğrudan taksi hedefi (MVT − BLOCK = NM − ofset); dönüş yine ofset birimi
        assert "_tx" in ad, "hedef=taksi çıktısının adında _tx olmalı"
        # NM farkı v126 özelliklerinde yok (EXCLUDE_V126; ofset hedefi ona çapalı) → taksi hedefinde girdi olarak eklenir
        p = genel_tahmin(tr, ap, F.feature_list(tr.columns) + ["NM_TAXI_DIFF_SEC"], tr["NM_TAXI_DIFF_SEC"].to_numpy(float) - y, opt)
        return ap["NM_TAXI_DIFF_SEC"].to_numpy(float) - p
    return genel_tahmin(tr, ap, F.feature_list(tr.columns), y, opt)


def genel_tahmin(tr, ap, feats, y, opt, cats=None, agirlik=None):
    """10.10: hedef ve özellik listesi dışarıdan (Model B μ için; scripts/b_mu_nn.py). y hedef birimiyle; dönüş aynı birim."""
    t0 = time.time()
    A, Ca, B, Cb, kard = hazirla(tr, ap, feats, 50 if opt.get("model") == "tabm" else MIN_SAYI, cats=cats)
    print(f"  hazırlık: {A.shape[1]} sayısal (eksiklik bayrakları dahil), {len(kard)} kategorik | {time.time() - t0:.0f} sn", flush=True)
    cihaz = opt.get("cihaz") or ("cuda" if torch.cuda.is_available() else "cpu")
    if cihaz == "cpu":
        torch.set_num_threads(int(opt.get("is_parcacigi", 12)))
    epok = int(opt.get("epok", 6))
    gizli = [int(h) for h in opt.get("gizli", "512,256,128").split(",")]
    tohumlar = [int(s) for s in opt.get("tohum", "42").split(",")]
    tur = opt.get("model", "mlp")
    if tur == "tabm":  # parçalı doğrusal gömme: kutu sınırları için örneklemde ≥ 2 farklı değeri olan sütunlar
        i = np.random.default_rng(0).choice(len(A), size=min(len(A), 200_000), replace=False)
        tut = np.array([len(np.unique(A[i, j])) > 1 for j in range(A.shape[1])])
        A, B = np.ascontiguousarray(A[:, tut]), np.ascontiguousarray(B[:, tut])
        print(f"  TabM: {int((~tut).sum())} sabit sütun atıldı, {int(tut.sum())} sayısal kaldı", flush=True)
    ek = dict(tabm_k=int(opt.get("tabm_k", 32)), tabm_d=int(opt.get("tabm_d", 512)), amp=opt.get("amp") == "1",
              bs=int(opt.get("bs", 2048)), delta=float(opt.get("delta", DELTA)))  # 11.10: delta (B μ varyantları)
    if agirlik is not None:  # 11.10: satır ağırlıkları (B μ'de b_agirlik=…)
        ek["agirlik"] = agirlik
    p = np.mean([egit_tahmin(A, Ca, y, B, Cb, kard, epok, s, gizli, cihaz, tur, **ek) for s in tohumlar], axis=0)
    return p


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ad = sys.argv[1]
    F.kuyruk_denetle(ad)
    opt = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
    hedefler = [a for a in sys.argv[2:] if "=" not in a]
    for h in hedefler:
        t0 = time.time()
        if h == "tam":
            tr, ap = F.prepare_full()
            p = calis(ad, tr, ap, opt)
            nm = ap["NM_TAXI_DIFF_SEC"].to_numpy(float)
            out = pd.DataFrame({"MVT_ID_mvt": ap["MVT_ID_mvt"].to_numpy(), "NM_TAXI_DIFF_SEC": nm, "nn": p,
                                "taksi": np.maximum(V.TAXI_FLOOR, nm - p)})
            out.to_parquet(PRED / f"TEST_nn_{ad}.parquet", index=False)
            print(f"[tam] {len(out):,} test satırı | ort. ofset {p.mean():.1f} | {time.time() - t0:.0f} sn", flush=True)
            continue
        tr, va, _ = F.prepare_split(h)
        p = calis(ad, tr, va, opt)
        pd.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"].to_numpy(), "nn": p}).to_parquet(PRED / f"NN_{ad}_{h}.parquet", index=False)
        nm, y = va["NM_TAXI_DIFF_SEC"].to_numpy(float), va[config.TARGET].to_numpy(float)
        pt, yt = np.maximum(V.TAXI_FLOOR, nm - p), nm - y
        ay = va["MVT_MONTH"].to_numpy()
        print(f"[{h}] taksi RMSE: " + ", ".join(f"ay {m} {np.sqrt(np.mean((pt[ay == m] - yt[ay == m]) ** 2)):.2f}"
                                                  for m in sorted(set(ay.tolist()))) + f" | {time.time() - t0:.0f} sn", flush=True)


if __name__ == "__main__":
    main()
