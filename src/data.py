"""Model A verisinin yüklenmesi. Girdi dosyaları yalnızca okunur."""
import numpy as np
import pandas as pd

from . import config

KEYS = ["MVT_ID_mvt", "ADEP_mvt", "MVT_MONTH", "MVT_DAY", "NM_TAXI_DIFF_SEC", config.TARGET]


def load_model_a(columns=(), path=config.DEP_TRAIN, year=2025):
    """Eğitim tablosundan Model A satırlarını (hedefi dolu) istenen sütunlarla yükler.

    Satırlar temizlenmez (doğrulama ham satırlarla yapılır). DAY_ID (yıl içindeki gün) eklenir.
    Anahtar sütunlar dışındaki float64 sütunlar float32'ye indirilir.
    """
    cols = list(dict.fromkeys(KEYS + list(columns)))
    df = pd.read_parquet(path, columns=cols)
    df = df[df[config.TARGET].notna()].reset_index(drop=True)
    for c in df.columns:
        if c not in KEYS and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    dates = pd.to_datetime(pd.DataFrame({"year": year, "month": df["MVT_MONTH"], "day": df["MVT_DAY"]}))
    df["DAY_ID"] = dates.dt.dayofyear.astype(np.int16)
    return df
