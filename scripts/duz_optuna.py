"""ADS-B düzelticisinin hiperparametre araması (05.10). Optuna TPE, sınırlı deneme.

Düzenek adsb_duzeltici.py cv ile aynı (üretim ayarları: havalimani=0 kayip=huber oz=4 tip=1 ek_aylar=1 kuyruk=1 iz5=1
seyrek=s30,s10; taban PRC_HARMAN): Ocak ve Temmuz 2025 sırayla dışarıda, eğitim diğer tüm aylar + seyreltilmiş iz kopyaları.
Veri bir kez hazırlanır; her denemede iki kat LightGBM eğitilir, her 100 turda Oca/Tem ağırlıklı RMSE (k = 1, 60 sn alt sınır)
ölçülür, denemenin değeri eğrinin en düşük noktası (tur da böylece seçilir). İlk deneme üretimdeki ayarlar (referans).
Kullanım: python scripts/duz_optuna.py <deneme_sayısı> [süre_sn]
Çıktı: kutuk/duz_optuna.csv (her deneme: ayarlar, en iyi tur, Oca / Tem / ağırlıklı RMSE)."""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_duzeltici as Dz  # noqa: E402
from src import validation as V  # noqa: E402

OPT = dict(havalimani="0", kayip="huber", oz="4", tip="1", ek_aylar="1", kuyruk="1", iz5="1", seyrek="s30,s10")
CIKTI = Dz.config.ROOT / "kutuk" / "duz_optuna.csv"


def hazirla():
    Dz.YENI_OZ, Dz.PARK_OZ, Dz.TIP_OZ, Dz.KUYRUK_OZ, Dz.IZ5 = True, True, True, True, True
    d = Dz.veri(True)
    d, _ = Dz.damga_ekle(d)
    es = d["ADSB_ESLESME"].fillna(0).to_numpy() > 0
    X = Dz.matris(d, False)
    r = (d["y_taksi"] - d["taban"]).to_numpy(float)
    ay = d["MVT_MONTH"].to_numpy()
    kop = Dz.kopyalar(OPT, False, X)
    katlar = []
    for b in (7, 1):
        tr, te = es & (ay != b), es & (ay == b)
        Xe, re_ = Dz.ekle_kopya(X[tr].reset_index(drop=True), r[tr], kop, set(ay[tr].tolist()))
        katlar.append((Xe, re_, X[te], te))
    return d, katlar, d["y_taksi"].to_numpy(float), d["taban"].to_numpy(float), ay


def main(n, sure):
    t0 = time.time()
    d, katlar, y, base, ay = hazirla()
    w = V.month_weights((1, 7))
    print(f"veri hazır: {len(d):,} satır, {time.time() - t0:.0f} sn", flush=True)
    kayit = []

    def amac(tr):
        prm = dict(objective="huber", learning_rate=tr.suggest_float("learning_rate", 0.01, 0.08, log=True),
                   num_leaves=tr.suggest_int("num_leaves", 63, 511, log=True),
                   min_data_in_leaf=tr.suggest_int("min_data_in_leaf", 50, 1000, log=True),
                   feature_fraction=tr.suggest_float("feature_fraction", 0.5, 1.0),
                   lambda_l2=tr.suggest_float("lambda_l2", 0.5, 50.0, log=True),
                   alpha=tr.suggest_float("alpha", 300.0, 1500.0, log=True),
                   bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=42, num_threads=12)
        tur_ust = tr.suggest_int("tur_ust", 600, 2000, step=200)
        rk = {}
        ta = time.time()
        for Xe, re_, Xte, te in katlar:
            m = lgb.train(prm, lgb.Dataset(Xe, re_), tur_ust)
            for k in range(100, tur_ust + 1, 100):
                rk.setdefault(k, np.zeros(len(d)))[te] = m.predict(Xte, num_iteration=k)
        sonuc = {}
        for k, v in rk.items():
            p = np.maximum(60.0, base + v)
            e = {a: np.mean((p[ay == a] - y[ay == a]) ** 2) for a in (1, 7)}
            sonuc[k] = (np.sqrt(w[1] * e[1] + w[7] * e[7]), np.sqrt(e[1]), np.sqrt(e[7]))
        k_en = min(sonuc, key=lambda k: sonuc[k][0])
        g, o, t = sonuc[k_en]
        kayit.append({**tr.params, "tur_en": k_en, "agirlikli": g, "oca": o, "tem": t, "sure_sn": round(time.time() - ta)})
        pd.DataFrame(kayit).to_csv(CIKTI, index=False)
        print(f"deneme {tr.number}: {g:.3f} (Oca {o:.2f}, Tem {t:.2f}) tur {k_en} | {tr.params} | {time.time() - ta:.0f} sn", flush=True)
        return g

    s = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5))
    # referans: üretim ayarları (PARAM + p:num_leaves=255, Huber δ = 600, tur 800 → eğri 1.000'e kadar)
    s.enqueue_trial(dict(learning_rate=0.03, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8, lambda_l2=10.0,
                         alpha=600.0, tur_ust=1000))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s.optimize(amac, n_trials=n, timeout=sure)
    print("en iyi:", round(s.best_value, 3), s.best_params, flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else None)
