"""Global derin CatBoost (GCB) girdisi ve eğitimi — v126 sayısalları + kategorikler (D3).

Varsayılan (A2 … v148): ADEP + v126'nın 5 kategoriği; sayısallar v126 gibi (metin → 0, NaN → 0, ±1e6 kırpma).
kat="ek5": v126'da sayısala zorlanıp 0'a dönen 5 metin sütunu kategorik olarak eklenir (sayısal kısımdan çıkarılır).
nan=True: sayısallarda eksik değer NaN kalır (CatBoost'un varsayılan nan_mode'u Min); ±inf yine ±1e6.
Ek CatBoost ayarları komut satırından "cb:<ad>=<değer>" biçiminde verilir (ör. cb:border_count=254,
cb:grow_policy=Depthwise); cb_params() bunları sayı / metin olarak çözer.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import features_v126 as F

KAT_EK = {"": [], "ek5": ["ADES_mvt", "MARKET_SEGMENT_flt", "STAND_TYPE", "AIRLINE_LETTERS", "ROUTE_PAIR"]}


def cats(kat=""):
    return ["ADEP_mvt"] + F.CAT_V126 + KAT_EK[kat]


def frame(df, feats, kat="", nan=False):
    fn = [c for c in feats if c not in KAT_EK[kat]]
    if nan:
        num = [c for c in fn if c not in F.CAT_V126]
        a = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(np.float32) for c in num])
        a = np.clip(np.nan_to_num(a, nan=np.nan, posinf=1e6, neginf=-1e6), -1e6, 1e6)
    else:
        a, num = F.numeric_array(F.matrix(df, fn), fn)
    c = cats(kat)
    return pd.concat([pd.DataFrame(a, columns=num), df[c].fillna("nan").astype(str).reset_index(drop=True)], axis=1)


def cb_params(args):
    out = {}
    for a in args:
        if a.startswith("cb:"):
            k, v = a[3:].split("=", 1)
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            out[k] = v
    return out


def fit(params, d_tr, y, cat_features, eval_set=None):
    """GPU hatasında (bellek) max_ctr_complexity=1 ile bir kez yeniden dener; bu params'a yazılır ve raporlanır."""
    try:
        return CatBoostRegressor(**params).fit(Pool(d_tr, y, cat_features=cat_features), eval_set=eval_set)
    except Exception as e:  # noqa: BLE001 — raporlanır; bilinçli ve kayıtlı yeniden deneme (A2 ile aynı)
        print(f"  CatBoost HATA (varsayılan max_ctr_complexity): {type(e).__name__}: {str(e)[:200]}", flush=True)
        params["max_ctr_complexity"] = 1
        return CatBoostRegressor(**params).fit(Pool(d_tr, y, cat_features=cat_features), eval_set=eval_set)
